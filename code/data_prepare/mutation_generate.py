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
MAX_RETRIES = 3
MAX_INPUT_LENGTH = 61000
MAX_THREADS = 20
MAX_REQUESTS_PER_MINUTE = 40
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE

output_lock = threading.Lock()
rate_limit_lock = threading.Lock()
last_request_time = [0.0]
from dotenv import load_dotenv
flkernel_path = os.getenv('FLKERNEL')

# 检查环境变量是否存在
if not flkernel_path or not os.path.isdir(flkernel_path):
    raise ValueError("错误: 请先设置有效的 FLKERNEL 环境变量, 例如: export FLKERNEL=/path/to/your/flkernel")

# 构建并加载 .env 文件
dotenv_path = os.path.join(flkernel_path, '.env')
load_dotenv(dotenv_path)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
# load_dotenv('/root/FLKERNEL/.env')
#client = OpenAI(
#    api_key="sk-",  # 替换为你的 API Key
#    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#)
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

def extract_syscalls(syz_content: str):
    syscalls = []
    pattern = re.compile(r"\b([a-zA-Z0-9_$]+)\(")
    for line in syz_content.splitlines():
        match = pattern.search(line)
        if match:
            syscall_name = match.group(1)
            if syscall_name not in syscalls:
                syscalls.append(syscall_name)
    return syscalls

def replace_placeholders(content, bug_report, code_line, sys_msg):
    content = content.replace("{BUG_REPORT_HERE}", bug_report or "N/A")
    content = content.replace("{SYZ_HERE}", code_line or "N/A")
    content = content.replace("{SYSCALL_HERE}", sys_msg or "N/A")
    return content
def replace_image_strings(text: str, threshold: int = 200):
    #for syz_content has huge image, replace the IMAGE before using LLM.
    if os.path.exists("image_path.json"):
        with open("image_path.json", "r") as f:
            image_map = json.load(f)
    else:
        image_map = {}
    counter = 0
    existing_vars = [int(k.replace("$IMAGE_", "")) for k in image_map if k.startswith("$IMAGE_")]
    if existing_vars:
        counter = max(existing_vars) + 1
    pattern = re.compile(r'(= ?\(?.*?)("([^"]{%d,})")' % threshold)
    def replacer(match):
        nonlocal counter
        prefix = match.group(1)
        full_string = match.group(2)
        raw_string = match.group(3)

        var_name = f'$IMAGE_{counter}'
        image_map[var_name] = full_string
        counter += 1

        return f'{prefix}{var_name}'

    new_text = pattern.sub(replacer, text)

    with open("image_path.json", "w") as f:
        json.dump(image_map, f, indent=2)
    print("done")
    return new_text
def replace_string_images(content: str) -> str:
    with open("image_path.json", "r") as f:
        image_map = json.load(f)

    pattern = re.compile(r'\$IMAGE_\d+')

    def replacer(match):
        var = match.group(0)
        return image_map.get(var, var)

    restored_content = pattern.sub(replacer, content)

    return restored_content

def extract_mutations_from_content(content):

    mutation_pattern = re.compile(
        r'\{\s*"origin"\s*:\s*"(.*?)"\s*,\s*"mutated"\s*:\s*"(.*?)"\s*\}', re.DOTALL
    )
    mutations = mutation_pattern.findall(content)
    output = ""
    for idx, (origin_line, mutated_line) in enumerate(mutations, start=1):
        output += f"// Mutation {idx}\n"
        output += f"// Original: {origin_line.strip()}\n"
        output += f"{mutated_line.strip()}\n\n"
    return output
def parse_mutations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    case_id=[]
    mutations_map = defaultdict(list)
    current_case_id = None

    for i, line in enumerate(lines):
        if line.startswith("Response for"):
            current_case_id = line.split()[2]
            #print(current_case_id)
            case_id.append(current_case_id)
            continue

        if "// Original:" in line and current_case_id:
            original_line = line.split("// Original:")[1].strip()
            mutated_line = lines[i + 1].strip()
            mutations_map[current_case_id].append((original_line, mutated_line))

    return dict(mutations_map),case_id

def apply_individual_mutations(mutations_map, case_paths, output_dir):
    for case_id, mutation_list in mutations_map.items():
        file_path = case_paths.get(case_id)
        print(file_path)
        if not file_path:
            print(f"No path found for case ID: {case_id}")
            continue

        # 读取文件内容
        try:
            with open(file_path, 'r') as file:
                original_lines = file.readlines()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

        for idx, (original_line, mutated_line) in enumerate(mutation_list, start=1):
            modified_lines = original_lines[:]
            flag=False
            for i, line in enumerate(modified_lines):
                if original_line in line:
                    modified_lines[i] = line.replace(original_line, mutated_line)
                    #print("succ",case_id,idx)
                    flag=True
                    break
            # 创建输出目录
            case_output_dir = os.path.join(output_dir, case_id)
            os.makedirs(case_output_dir, exist_ok=True)

            # 保存修改后的文件
            output_file_path = os.path.join(case_output_dir, f"{case_id}_{idx}.syz")
            with open(output_file_path, 'w') as output_file:
                output_file.writelines(modified_lines)
def extract_and_write_by_id(file_path, syzid):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    extracting = False
    syscall_description = []
    target_header = f"response for {syzid}"

    for line in lines:
        if line.strip() == target_header:
            if extracting:
                # 第二次出现，说明提取结束
                break
            else:
                extracting = True
                continue
        elif extracting and line.strip().startswith("response for "):
            # 下一个 response 开始了，提前结束
            break
        elif extracting:
            syscall_description.append(line)

    return "".join(syscall_description)
def process_entry(entry, report_folder, testcase_folder, base_prompt, standard_output, output_file,output_file_4case):
    syzid = entry[0]
    report_path = os.path.join(report_folder, f"{syzid}.txt")
    syz_path = os.path.join(testcase_folder, f"{syzid}.syz")
    report_content = read_file(report_path)
    syz_content = read_file(syz_path)
    syz_content=replace_image_strings(syz_content)
    print(syz_content)
    syscall_list = extract_syscalls(syz_content)

    syscall_description =extract_and_write_by_id("../../datasets/syscall_description.txt",syzid)
    #Here, to avoid introducing new dependencies, we have prepared the syscall descriptions in advance.
    #If you wish to generate them yourself, please refer to the link: https://github.com/google/syzkaller/tree/master/sys/linux .
    prompt = replace_placeholders(base_prompt, report_content, syz_content, syscall_description)
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
            mutation_result_extract=extract_mutations_from_content(result)
            mutation_result=replace_string_images(mutation_result_extract)
            with output_lock:
                with open(output_file, "a", encoding="utf-8") as output:
                    output.write(f"Response for {syzid}:\n{mutation_result_extract}\n")
                with open(output_file_4case, "a", encoding="utf-8") as output:
                    output.write(f"Response for {syzid}:\n{mutation_result}\n")
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
    parser.add_argument('--output_file', required=True, help='Path to save outputs')
    parser.add_argument('--mutation_case_dir', required=True, help='Path to save mutation testcase')
    args = parser.parse_args()
    output_file_4case="../data_prepare/tmp.txt"
    with open(args.case_list, 'r', encoding="utf-8") as f:
        entries = [line.strip().split('/') for line in f]
    #print(entries)
    base_prompt = read_file("../prompt/mutation_prompt.txt", MAX_INPUT_LENGTH)
    standard_output = read_file("../prompt/mutation_standard_output.txt", MAX_INPUT_LENGTH)

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [
            executor.submit(process_entry, entry, args.report_dir, args.testcase_dir, base_prompt, standard_output, args.output_file, output_file_4case)
            for entry in entries
        ]
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error during processing: {e}")
    case_paths = {}
    mutations,caseid = parse_mutations(output_file_4case)
    for entry in entries:
        syzid = entry[0]
        case_path = os.path.join(args.testcase_dir, f"{syzid}.syz")
        case_paths[syzid] = case_path
    apply_individual_mutations(mutations, case_paths, args.mutation_case_dir)
if __name__ == "__main__":
    main()
