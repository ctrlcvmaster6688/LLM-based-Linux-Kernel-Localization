import re
import json
from typing import List
import torch
import torch.nn as nn
import os
import subprocess
import concurrent.futures
import argparse
from openai import OpenAI, OpenAIError

from collections import defaultdict
from get_cg_level import analyze_function_hierarchy, get_call_counts
from get_complexcity import calculate_cyclomatic_complexity, halstead_effort
from get_duringtime import calculate_function_time_interval
from tool_test import extract_function_bodies, count_local_global_variables, estimate_max_nesting_depth
from dotenv import load_dotenv
flkernel_path = os.getenv('FLKERNEL')

# 检查环境变量是否存在
if not flkernel_path or not os.path.isdir(flkernel_path):
    raise ValueError("错误: 请先设置有效的 FLKERNEL 环境变量, 例如: export FLKERNEL=/path/to/your/flkernel")

# 构建并加载 .env 文件
dotenv_path = os.path.join(flkernel_path, '.env')
load_dotenv(dotenv_path)

def get_file_names_for_syzid(input_file , syzid) :
    responses = {}
    with open(input_file, "r") as f:
        content = f.read()
    pattern = re.compile(
        r"response for (\w+).*?\"standard_output\":\s*\[\s*(.*?)\s*\]",
        re.DOTALL
    )
    matches = pattern.findall(content)
    for case_id, file_list in matches:
        responses[case_id] = []
        files = re.findall(r"\"(.*?)\"", file_list)
        for file in files:
            if ':' in file:
                _, func_part = file.split(':', 1)
                func_match = re.match(r'(\w+)', func_part.strip())
                if func_match:
                    file = func_match.group(1)
            responses[case_id].append(file)
    return responses[syzid]
def parse_syzid_commit_file(file_path):
    result = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                syzid, commit = line.strip().split(':', 1)
                result[syzid.strip()] = commit.strip()
    return result
def run_command(command, cwd=None):
    subprocess.run(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
class MultiTaskRankNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.rank_head = nn.Linear(128, 1)
        self.class_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        score = self.rank_head(x).squeeze(-1)
        prob = torch.sigmoid(self.class_head(x).squeeze(-1))
        return score, prob

def load_model(model_path, device):
    model = MultiTaskRankNet(input_dim=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def score_functions(model, functions, device):
    func_names = [name for name, feats in functions]
    features = torch.tensor([feats for _, feats in functions], dtype=torch.float32).to(device)
    with torch.no_grad():
        scores, _ = model(features)
        scores = scores.cpu().numpy()
    return sorted(zip(func_names, scores), key=lambda x: -x[1])
def compute_feature(funcname_params_body_tuple, file_rel_path, commit, level_map, call_degree_map,work_dir):
    funcname, (para, param_num, body) = funcname_params_body_tuple
    try:
        c1 = calculate_cyclomatic_complexity(body)
        c2 = halstead_effort(body)
        t = calculate_function_time_interval(file_rel_path, funcname, commit,work_dir)
        t = t if t is not None else 0
        l = level_map.get(funcname, 0)
        in_degree, out_degree = call_degree_map.get(funcname, (0, 0))
        static_var, global_var = count_local_global_variables(body)
        max_nest = estimate_max_nesting_depth(body)
        return (funcname, [param_num, c1, c2, t, l, in_degree, out_degree, static_var, global_var, max_nest])
    except Exception as e:
        print(f"Error computing feature for {funcname}: {e}")
        return None
def read_file( file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
def replace_placeholders(content,bug_report,filename,repro_content,mutate_content1,mutate_content2,mutate_content3,mutate_content4,funlist):
    """替换占位符"""
    content = content.replace("{BUG_REPORT}", bug_report or "N/A")
    content = content.replace("{FUNC_LIST_HERE}", funlist or "N/A")
    content = content.replace("{FILE_NAME}", filename or "N/A")
    content = content.replace("{TSER_PROGRAM_HERE}", repro_content or "N/A")
    content = content.replace("{MUTATE1_HERE}", mutate_content1 or "N/A")
    content = content.replace("{MUTATE2_HERE}", mutate_content2 or "N/A")
    content = content.replace("{MUTATE3_HERE}", mutate_content3 or "N/A")
    content = content.replace("{MUTATE4_HERE}", mutate_content4 or "N/A")

    #content = content.replace("{config_content_here}", config)
    return content
def extract_function_names_from_result(result):
    try:
        function_name_pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
        function_names = [name for name in re.findall(function_name_pattern, result) if '_' in name]
        return function_names
    except Exception as e:
        print(f"Unexpected error in extract_function_names_from_result: {e}")
        return []
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
if __name__ == "__main__":
    model_path="filter_model.pt"
    commit_path="../../datasets/case_commit.txt"
    parser = argparse.ArgumentParser(description="Evaluate Fault Localization Metrics")
    parser.add_argument('--filelist', required=True, help='Path to the case list ')
    parser.add_argument('--result_file', required=True, help='Path to the model result output file')
    parser.add_argument('--work_dir', required=True, help='Path to linux kernel')
    parser.add_argument('--output_dir', required=True, help='Path to save output')
    parser.add_argument('--mutation_output', required=True, help='Path to parse mutation result')
    parser.add_argument('--report_dir', required=True, help='Directory containing reports')
    parser.add_argument('--testcase_dir', required=True, help='Directory containing test cases')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    commit_map = parse_syzid_commit_file(commit_path)
    standard_output1=read_file("../prompt/func_fliter_standard_1.txt")
    standard_output2=read_file("../prompt/func_fliter_standard_2.txt")
    standard_output3=read_file("../prompt/func_fliter_standard_3.txt")
    base_prompt1=read_file("../prompt/func_fliter_prompt_1.txt")
    base_prompt2=read_file("../prompt/func_fliter_prompt_2.txt")
    base_prompt3=read_file("../prompt/func_fliter_prompt_3.txt")
    with open(args.filelist, "r") as f:
        entries = [line.strip().split('/') for line in f]

    for entry in entries:
        syzid = entry[0]
        print(f"Processing {syzid}")
        commit = commit_map.get(syzid)
        if not commit:
            print("commit not found.")
            continue
        run_command(f"git checkout -f {commit}", cwd=args.work_dir)
        filelist_for_case = get_file_names_for_syzid(args.result_file, syzid)
        index=0
        mutate_num1=entry[1]
        mutate_num2=entry[2]
        mutate_num3=entry[3]
        mutate_num4=entry[4]
        mutate_map=parse_mutations(args.mutation_output)
        #print(mutate_map)
        report_path = os.path.join(args.report_dir, f"{syzid}.txt")
        syz_path = os.path.join(args.testcase_dir, f"{syzid}.syz")
        report_content = read_file(report_path)
        syz_content = process_syz_file(syz_path)
        mutate_content1= extract_mutation(mutate_map,syzid, mutate_num1)
        mutate_content2= extract_mutation(mutate_map,syzid, mutate_num2)
        mutate_content3= extract_mutation(mutate_map,syzid, mutate_num3)
        mutate_content4= extract_mutation(mutate_map,syzid, mutate_num4)
        with open(args.output_dir,"a")as output:
            output.write(f"func for case {syzid}\n")
        for file_rel_path in filelist_for_case:
            index+=1
            print(f"File: {file_rel_path}：{index}")
            file_abs_path = os.path.join(args.work_dir, file_rel_path)
            if not os.path.exists(file_abs_path):
                print(f" not found: {file_abs_path}")
                continue
            funclist = extract_function_bodies(file_abs_path)
            if not funclist:
                print(" No functions extracted")
                continue

            filenames = list(funclist.keys())
            level_map = analyze_function_hierarchy(file_abs_path, filenames)
            call_degree_map = get_call_counts(file_abs_path, filenames)

            features = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = [
                    executor.submit(compute_feature, item, file_rel_path, commit, level_map, call_degree_map,args.work_dir)
                    for item in funclist.items()
                ]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res is not None:
                        features.append(res)
            if not features:
                continue
            scored = score_functions(model, features, device)
            total = len(scored)
            raw_bottom_func = scored[:int(total * 0.8)] # Hyperparameters ,here set 0.8 means filtering out the last twenty percent of the function
            bottom_func = []
            for funcname, score in raw_bottom_func:
                if funcname in funclist:
                    para = funclist[funcname][0]
                    full_signature = f"{funcname}({para})"
                    bottom_func.append(full_signature)
            #print(bottom_func)
            result_str='\n'.join(bottom_func)
            if index <=3:
                base_prompt=base_prompt1
                standard_output=standard_output1
                print(file_rel_path,index)
            if index>3 and index <=7:
                base_prompt=base_prompt2
                standard_output=standard_output2
                print(file_rel_path,index)
            if index>7:
                base_prompt=base_prompt3
                standard_output3=standard_output3
                print(file_rel_path,index)
#            client = OpenAI(
#                api_key="",  # 替换为你的 API Key
#                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#            )
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
            prompt=replace_placeholders(base_prompt,report_content,file_rel_path,syz_content,mutate_content1,mutate_content2,mutate_content3,mutate_content4,result_str)
            system_content=f'''You are an experienced AI software engineer specializing in identifying functions that are more likely to trigger bugs described in a bug report.
                                Your task is to analyze the provided bug report, file name, and list of functions within the file to determine which functions are most relevant to the bug. 
                                You need to understand the functionality of each function and the details of the bug report to make this determination. 
                                Your output should be a list of function names, ranked by their likelihood of being related to the bug in json format.
            '''
            completion = client.chat.completions.create(
                model="deepseek-v3" ,# 此处以 deepseek-r1 为例，可按需更换模型名称。
                messages=[
                    {'role': 'system', 'content':system_content},
                    {'role': 'assistant','content':standard_output},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0,
                response_format={'type': 'json_object'}
            )
            result=completion.choices[0].message.content
            print(result)
            if result == []:
                result=completion.choices[0].message.content
            results=extract_function_names_from_result(result)
            for func in results:
                with open(args.output_dir,"a")as output:
                    output.write(f"{file_rel_path}:{func}\n")
