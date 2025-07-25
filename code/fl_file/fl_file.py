import os
import json
import time
import threading
import argparse
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, OpenAIError
import requests
from collections import defaultdict
from dotenv import load_dotenv

# load_dotenv('/root/FLKERNEL/.env')
flkernel_path = os.getenv('FLKERNEL')

# 检查环境变量是否存在
if not flkernel_path or not os.path.isdir(flkernel_path):
    raise ValueError("错误: 请先设置有效的 FLKERNEL 环境变量, 例如: export FLKERNEL=/path/to/your/flkernel")

# 构建并加载 .env 文件
dotenv_path = os.path.join(flkernel_path, '.env')
load_dotenv(dotenv_path)

MAX_RETRIES = 3
MAX_INPUT_LENGTH = 61000
MAX_THREADS = 20
MAX_REQUESTS_PER_MINUTE = 40
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE

output_lock = threading.Lock()
rate_limit_lock = threading.Lock()
last_request_time = [0.0]

#client = OpenAI(
#    api_key="",  # 替换为你的 API Key
#    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

def wait_for_rate_limit():
    with rate_limit_lock:
        now = time.time()
        elapsed = now - last_request_time[0]
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        last_request_time[0] = time.time()

def read_file(file_path, max_length=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content[:max_length] if max_length else content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return ""



def replace_extend_placeholders(content, bug_report, code_line):
    content = content.replace("{BUG_REPORT_HERE}", bug_report or "N/A")
    content = content.replace("{SYZ_HERE}", code_line or "N/A")
    return content

def replace_fl_placeholders(content,repro_content, mutate_content1, mutate_content2,mutate_content3,mutate_content4,report_content,filelist):
    """替换占位符"""
    content = content.replace("{TEST_PROGRAM_HERE}", repro_content or "N/A")
    content = content.replace("{MUTATE1_HERE}", mutate_content1 or "N/A")
    content = content.replace("{MUTATE2_HERE}", mutate_content2 or "N/A")
    content = content.replace("{MUTATE3_HERE}", mutate_content3 or "N/A")
    content = content.replace("{MUTATE4_HERE}", mutate_content4 or "N/A")
    content = content.replace("{FILE_LIST_HERE}", filelist or "N/A")
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
def process_extend_entry(entry, report_folder, testcase_folder, base_prompt, standard_output, output_file):
    syzid = entry[0]
    report_path = os.path.join(report_folder, f"{syzid}.txt")
    syz_path = os.path.join(testcase_folder, f"{syzid}.syz")
    report_content = read_file(report_path)
    syz_content = process_syz_file(syz_path)

    prompt = replace_extend_placeholders(base_prompt, report_content, syz_content)
    print(prompt)
    retries = 0
    while retries < MAX_RETRIES:
        try:
            wait_for_rate_limit()

            response = client.chat.completions.create(
                model="deepseek-v3",
                temperature=0,
                messages=[
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': standard_output}
                ],
            )

            result = response.choices[0].message.content if response.choices else ""
            with output_lock:
                with open(output_file, "a", encoding="utf-8") as output:
                    output.write(f"response for {syzid}:\n{result}\n")
            print(f"\n====== Finish case {syzid} ======\n")
            return
        except (requests.exceptions.Timeout, OpenAIError) as e:
            retries += 1
            print(f"Warning: {syzid} API 请求失败 (尝试 {retries}/{MAX_RETRIES})，错误: {e}")
            time.sleep(2 ** retries)
        except Exception as e:
            print(f"Error: 处理 {syzid} 时发生未知错误: {e}")
            return
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
def extract_files_by_regex(filepath, target_id):
    with open(filepath, 'r') as f:
        content = f.read()

    # 提取指定 ID 的 response 块
    pattern = rf'response for {target_id}:\s*(\{{.*?)(?=^response for|\Z)'
    match = re.search(pattern, content, flags=re.DOTALL | re.MULTILINE)

    if not match:
        raise ValueError(f"未找到 ID: {target_id}")

    block = match.group(1)

    # 匹配所有形如 "some/path.c" 的字符串
    file_pattern = r'"([^"]+\.[chS])"'  # 包含 .c .h .S 文件
    files = re.findall(file_pattern, block)

    return sorted(set(files))
def process_fl_entry(entry, report_folder, testcase_folder, base_prompt, standard_output, output_file,mutate_result_dir,extend_output):
        syzid = entry[0]
        mutate_num1=entry[1]
        mutate_num2=entry[2]
        mutate_num3=entry[3]
        mutate_num4=entry[4]
        mutate_map=parse_mutations(mutate_result_dir)
        #print(mutate_map)
        report_path = os.path.join(report_folder, f"{syzid}.txt")
        syz_path = os.path.join(testcase_folder, f"{syzid}.syz")
        report_content = read_file(report_path)
        syz_content = process_syz_file(syz_path)
        mutate_content1= extract_mutation(mutate_map,syzid, mutate_num1)
        mutate_content2= extract_mutation(mutate_map,syzid, mutate_num2)
        mutate_content3= extract_mutation(mutate_map,syzid, mutate_num3)
        mutate_content4= extract_mutation(mutate_map,syzid, mutate_num4)
        filelist=extract_files_by_regex(extend_output,syzid)
        filelist= '\n'.join(filelist)
        prompt = replace_fl_placeholders(base_prompt, syz_content,mutate_content1, mutate_content2, mutate_content3, mutate_content4, report_content, filelist)
        #print(prompt)
        retries = 0
        while retries < MAX_RETRIES:
            try:
                wait_for_rate_limit()

                response = client.chat.completions.create(
                    model="deepseek-v3",
                    temperature=0,
                    messages=[
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': standard_output}
                    ],
                )

                result = response.choices[0].message.content if response.choices else ""
                with output_lock:
                    with open(output_file, "a", encoding="utf-8") as output:
                        output.write(f"response for {syzid}:\n{result}\n")
                print(f"\n====== Finish case {syzid} ======\n")
                return
            except (requests.exceptions.Timeout, OpenAIError) as e:
                retries += 1
                print(f"Warning: {syzid} API 请求失败 (尝试 {retries}/{MAX_RETRIES})，错误: {e}")
                time.sleep(2 ** retries)
            except Exception as e:
                print(f"Error: 处理 {syzid} 时发生未知错误: {e}")
                return
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_list', required=True, help='Path to case list')
    parser.add_argument('--report_dir', required=True, help='Directory containing reports')
    parser.add_argument('--testcase_dir', required=True, help='Directory containing test cases')
    parser.add_argument('--output_extend', required=True, help='Path to save file list extend output')
    parser.add_argument('--mutation_output', required=True, help='Path to parse mutation result')
    parser.add_argument('--localize_output', required=True, help='Path to save localize result')
    args = parser.parse_args()
    with open(args.case_list, 'r', encoding="utf-8") as f:
        entries = [line.strip().split('/') for line in f]
    print(entries)

    base_prompt_extend = read_file("../prompt/extend_scope_prompt.txt", MAX_INPUT_LENGTH)
    standard_output_extend = read_file("../prompt/extend_scope_standard_output.txt", MAX_INPUT_LENGTH)
    base_prompt_fl = read_file("../fl_file_prompt.txt", MAX_INPUT_LENGTH)
    standard_output_fl = read_file("../prompt/fl_file_standard_output.txt", MAX_INPUT_LENGTH)
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [
            executor.submit(process_extend_entry, entry, args.report_dir, args.testcase_dir, base_prompt_extend, standard_output_extend, args.output_extend)
            for entry in entries
        ]
    for future in as_completed(futures):
        try:
            future.result()  # 显式等待任务完成，可捕获异常
        except Exception as e:
            print(f"Error during processing: {e}")


    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [
            executor.submit(process_fl_entry, entry, args.report_dir, args.testcase_dir, base_prompt_fl, standard_output_fl,args.localize_output,args.mutation_output, args.output_extend)
            for entry in entries
        ]
    for future in as_completed(futures):
        try:
            future.result()  # 显式等待任务完成，可捕获异常
        except Exception as e:
            print(f"Error during processing: {e}")
if __name__ == "__main__":
    main()
