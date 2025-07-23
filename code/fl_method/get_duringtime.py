import subprocess
from datetime import datetime
import re
import json
import os
# 获取指定版本提交的时间（使用 commit hash）
def get_commit_time(commit_hash,work_dir):
    result = subprocess.run(
        ['git', 'log', '-1', '--format=%ci', commit_hash],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/home/pangyesong/linux"
    )
    # 返回日期部分
    #print(result)
    return result.stdout.strip()
def extract_first_commit_hash(stdout):
    match = re.search(r"^commit\s+([0-9a-f]{40})", stdout, re.MULTILINE)
    if match:
        return match.group(1)
    return None
# 获取函数首次修改的提交时间
def get_function_first_commit_time(func_name, file_path, before_commit_hash,work_dir):
    result = subprocess.run(
        ['git', 'log', '-S', func_name, '--', file_path, '--before', before_commit_hash, '--reverse', '--date=iso'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',  # 关键修改点
        env={**os.environ, 'LANG': 'C.UTF-8'} ,
        cwd=work_dir
    )

    commit=extract_first_commit_hash(result.stdout)
    return commit

# 计算时间间隔
def calculate_days_between(time1_str, time2_str):
    format_str = "%Y-%m-%d %H:%M:%S %z"
    time1 = datetime.strptime(time1_str, format_str)
    time2 = datetime.strptime(time2_str, format_str)
    
    delta = abs(time1 - time2)
    return delta.days

# 主函数，接受文件路径、函数名和commit hash
def calculate_function_time_interval(file_path, func_name, commit_hash,work_dir):
    # 获取版本提交时间
    version_commit_time = get_commit_time(commit_hash,work_dir)
    #print(version_commit_time)
    
    # 获取函数首次修改时间
    first_commit = get_function_first_commit_time(func_name, file_path, version_commit_time,work_dir)
    if not first_commit:
        print(f"函数 {func_name} 在版本 {commit_hash} 之前没有修改记录")
        return
    first_commit_time=get_commit_time(first_commit,work_dir)
    #print(first_commit_time)

    # 计算时间间隔
    time_diff = calculate_days_between(first_commit_time, version_commit_time)
    return time_diff
def parse_oracle_file(oracle_file_path):
    """
    Parse the oracle file into a dictionary with case IDs as keys 
    and oracle function names as values.
    """
    oracle = {}
    with open(oracle_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line:
            # Split the line into case ID and oracle function names
            case_id, oracle_functions = line.split(":", 1)
            oracle[case_id] = oracle_functions.split(",")  # Split function names by comma
    return oracle
def parse_json_to_map(json_file_path):
    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 创建一个字典，将每个键值对映射为新的 map
    branch_to_files_map = {branch: files for branch, files in data.items()}
    
    return branch_to_files_map



