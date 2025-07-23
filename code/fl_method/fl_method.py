import os
import json
import time
import re
import requests
import argparse
from threading import Lock
from openai import OpenAI, OpenAIError
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
# === 配置路径 ===
function_summary_path = "../../results/function_summary.json"
standard_output1_path = "../prompt/func_fl_standard_1.txt"
standard_output2_path = "../prompt/func_fl_standard_2.txt"
from dotenv import load_dotenv
flkernel_path = os.getenv('FLKERNEL')

# 检查环境变量是否存在
if not flkernel_path or not os.path.isdir(flkernel_path):
    raise ValueError("错误: 请先设置有效的 FLKERNEL 环境变量, 例如: export FLKERNEL=/path/to/your/flkernel")

# 构建并加载 .env 文件
dotenv_path = os.path.join(flkernel_path, '.env')
load_dotenv(dotenv_path)

# load_dotenv('/root/FLKERNEL/.env')
# === 初始化 OpenAI 客户端 ===
#client = OpenAI(
#    api_key="",  # 替换为你的 API Key
#    base_url="",
#)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
# === 提取最外层 JSON（支持 {} 或 []） ===
def extract_json_from_text(text):
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    return match.group(1) if match else None
def get_functions_by_case_id(case_id, file_path):
    funcs_list = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_case_id = None

    for line in lines:
        line = line.strip()

        # 检查是否是 case_id 行
        if line.startswith("func for case"):
            if current_case_id == case_id and funcs_list:
                break  # 找到目标 case_id，停止读取
            current_case_id = line.split(" ")[3]  # 提取 case_id
            funcs_list = []  # 重置函数列表
        elif line and current_case_id == case_id:
            funcs_list.append(line)
    return funcs_list
# === 文件读取函数 ===
def read_file(file_path, max_length=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content[:max_length] if max_length else content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return ""
def replace_fl_placeholders(content,repro_content, mutate_content1, mutate_content2,mutate_content3,mutate_content4,report_content,filelist):
    """替换占位符"""
    content = content.replace("{TEST_PROGRAM_HERE}", repro_content or "N/A")
    content = content.replace("{MUTATE1_HERE}", mutate_content1 or "N/A")
    content = content.replace("{MUTATE2_HERE}", mutate_content2 or "N/A")
    content = content.replace("{MUTATE3_HERE}", mutate_content3 or "N/A")
    content = content.replace("{MUTATE4_HERE}", mutate_content4 or "N/A")
    content = content.replace("{FUNCTION_LIST_HERE}", filelist or "N/A")
    content = content.replace("{BUG_REPORT_HERE}", report_content or "N/A")
    return  content
def process_syz_file(file_path, output_path=None):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        if line.startswith("syz_mount_image$"):
            line = re.sub(r"(syz_mount_image\$\w+).*", r"\1()", line)
            #print(line) # 替换为 syz_mount_image$xxx()
        processed_lines.append(line)

    result = "".join(processed_lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
    return result
def parse_mutations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 字典用于存储结构：{case_id: [(original_line, mutated_line), ...]}
    mutations_map = defaultdict(list)
    current_case_id = None

    for i, line in enumerate(lines):
        # 检查是否是新 case id 的起始行
        if line.startswith("Response for"):
            current_case_id = line.split()[2]  # 获取case ID
            continue

        # 检查是否为变异的原始代码行标记
        if "// Original:" in line and current_case_id:
            original_line = line.split("// Original:")[1].strip()
            mutated_line = lines[i + 1].strip()  # 下一行是变异代码
            mutations_map[current_case_id].append((original_line, mutated_line))

    return dict(mutations_map)
def extract_mutation(mutations_map, case_id, mutate_id):
    if case_id == 0:
        return None
    if case_id in mutations_map:
        mutations = mutations_map[case_id]
        if 1 <= int(mutate_id) <= len(mutations):
            original_line, mutated_line = mutations[int(mutate_id) - 1]
            return f"{original_line} -> {mutated_line}"
    return None
# === 模型调用函数 ===
def call_model(client, messages, retries=3):
    result = ""
    for retry in range(retries):
        try:
            completion = client.chat.completions.create(
                temperature=0,
                model="deepseek-v3",
                messages=messages,
                response_format={'type': 'json_object'},
                stream=True,
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    result += chunk.choices[0].delta.content
            return result
        except (requests.exceptions.Timeout, OpenAIError) as e:
            print(f"Retry {retry + 1}/{retries} failed: {e}")
            time.sleep(2 ** retry)
    return None

# === 加载第一轮已有总结 ===
try:
    with open(function_summary_path, 'r', encoding='utf-8') as f:
        func_summaries = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    func_summaries = {}
summary_lock = Lock()

# === 读取标准输出提示词 ===
standard_output1 = read_file(standard_output1_path)
standard_output2 = read_file(standard_output2_path)

# === 用于处理每个 syzid 的函数 ===
def process_case(entry, report_folder, testcase_folder, base_prompt1, base_prompt2, mutate_result_dir,candidate_func):
    syzid = entry[0]
    mutate_num1=entry[1]
    mutate_num2=entry[2]
    mutate_num3=entry[3]
    mutate_num4=entry[4]
    mutate_map=parse_mutations(mutate_result_dir)
    report_path = os.path.join(report_folder, f"{syzid}.txt")
    syz_path = os.path.join(testcase_folder, f"{syzid}.syz")
    report_content = read_file(report_path)
    syz_content = process_syz_file(syz_path)
    funclist=get_functions_by_case_id(syzid,candidate_func)
    mutate_content1= extract_mutation(mutate_map,syzid, mutate_num1)
    mutate_content2= extract_mutation(mutate_map,syzid, mutate_num2)
    mutate_content3= extract_mutation(mutate_map,syzid, mutate_num3)
    mutate_content4= extract_mutation(mutate_map,syzid, mutate_num4)
    funclist='\n'.join(funclist)
    prompt1=replace_fl_placeholders(base_prompt1, syz_content,mutate_content1, mutate_content2, mutate_content3, mutate_content4, report_content, funclist)
    prompt2=replace_fl_placeholders(base_prompt2, syz_content,mutate_content1, mutate_content2, mutate_content3, mutate_content4, report_content, funclist)
    print(f"\n=== Running {syzid} ===")

    # === 第一轮：总结函数功能 ===
    if syzid in func_summaries:
        print(f"Skip first round for {syzid}, already processed.")
    else:
        messages = [
            {'role': 'assistant', 'content': standard_output1},
            {'role': 'user', 'content': prompt1}
        ]
        result = call_model(client, messages)
        if not result:
            return None

        json_str = extract_json_from_text(result)
        if not json_str:
            print(f"First round: No valid JSON detected for {syzid}")
            return None

        try:
            json_data = json.loads(json_str)
            with summary_lock:
                func_summaries[syzid] = json_data
                with open(function_summary_path, 'w', encoding='utf-8') as f:
                    json.dump(func_summaries, f, indent=2)
        except json.JSONDecodeError:
            print(f"First round JSON decode error for {syzid}")
            return None

    second_prompt_full = prompt2 + "\n\n=== Function Summary ===\n" + json.dumps(func_summaries[syzid], indent=2)
    messages = [
        {'role': 'assistant', 'content': standard_output2},
        {'role': 'user', 'content': second_prompt_full}
    ]
    result = call_model(client, messages)
    if not result:
        return None

    json_str = extract_json_from_text(result)
    if not json_str:
        print(f"Second round: No valid JSON detected for {syzid}")
        return None

    try:
        json_data = json.loads(json_str)
        if "standard_output" not in json_data:
            print(f"Second round: Missing 'standard_output' in JSON for {syzid}")
            return None
        return syzid, json_data
    except json.JSONDecodeError:
        print(f"Second round JSON decode error for {syzid}")
        return None

# === 主执行入口 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fl for method level")
    parser.add_argument('--filelist', required=True, help='Path to the case list')
    parser.add_argument('--mutation_output', required=True, help='Path to parse mutation result')
    parser.add_argument('--report_dir', required=True, help='Directory containing reports')
    parser.add_argument('--testcase_dir', required=True, help='Directory containing test cases')
    parser.add_argument('--func_list', required=True, help='Path to save filtered candidate functions')
    parser.add_argument('--output_dir', required=True, help='Path to save output')
    args = parser.parse_args()

    file_list_dir = args.filelist

    with open(file_list_dir, 'r', encoding="utf-8") as f:
        entries = [line.strip().split('/') for line in f]
    base_prompt1=read_file("../prompt/func_fl_prompt1.txt")
    base_prompt2=read_file("../prompt/func_fl_prompt_2.txt")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_case, entry,args.report_dir,args.testcase_dir,base_prompt1,base_prompt2,args.mutation_output,args.func_list): entry[0] for entry in entries}
        for future in as_completed(futures):
            syzid = futures[future]
            try:
                result = future.result()
                if result:
                    syzid, json_data = result
                    with open(args.output_dir, "a", encoding="utf-8") as f:
                        f.write(f"response for {syzid}:\n{json.dumps(json_data, indent=2)}\n")
                    print(f"\n{syzid} done.")
                else:
                    print(f" {syzid} skipped or failed.")
            except Exception as e:
                print(f"{syzid} encountered error: {e}")
