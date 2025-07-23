import subprocess
import re
from collections import defaultdict
from typing import List, Dict, Tuple


def analyze_function_hierarchy(file_path: str, target_funcs: List[str]) -> Dict[str, int]:
    # 调用 cflow 并启用 print-level
    try:
        result = subprocess.run(
            ["cflow", "--all", "--print-level", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        cflow_output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"cflow 执行失败: {e}")
        return {func: 0 for func in target_funcs}

    reverse_call_graph = defaultdict(list)  # callee -> list of callers
    level_func_list = []  # 存储 (level, function) 顺序信息

    level_func_pattern = re.compile(r"^\{\s*(\d+)\}\s+([\w\d_]+)")

    # 解析 cflow 输出，提取函数名和其调用层级
    for line in cflow_output.splitlines():
        match = level_func_pattern.match(line)
        if not match:
            continue
        level = int(match.group(1))
        func = match.group(2)
        level_func_list.append((level, func))

    # 构建调用图（反向：callee -> [callers]）
    call_stack = []
    for level, func in level_func_list:
        if level == 0:
            call_stack = [func]
        else:
            # 确保调用栈正确截断再添加
            call_stack = call_stack[:level]
            if level >= 1:
                caller = call_stack[level - 1]
                reverse_call_graph[func].append(caller)
            call_stack.append(func)

    # 计算每个函数的层级（递归时为0）
    level_cache = {}
    visiting = set()

    def get_func_level(func: str) -> int:
        if func in visiting:
            return 0  # 递归调用，返回0
        if func in level_cache:
            return level_cache[func]
        callers = reverse_call_graph.get(func, [])
        if not callers:
            level_cache[func] = 1
            return 1
        visiting.add(func)
        max_parent_level = 0
        for caller in callers:
            caller_level = get_func_level(caller)
            if caller_level == 0:
                max_parent_level = 0
                break
            max_parent_level = max(max_parent_level, caller_level)
        visiting.remove(func)
        level_cache[func] = max_parent_level + 1 if max_parent_level > 0 else 0
        return level_cache[func]

    # 处理目标函数
    all_funcs = {func for _, func in level_func_list}
    result = {}
    for func in target_funcs:
        if func not in all_funcs:
            result[func] = 0  # 函数不在调用图中
        else:
            result[func] = get_func_level(func)

    return result
def get_called_count_from_reverse_cflow(lines_reverse: List[str], func: str) -> int:
    """
    根据 cflow --reverse 输出，计算 func 被调用的次数（即被多少个函数调用）。
    """
    try:
        called_count = 0
        func_line_idx = None

        # 找到缩进为0且匹配函数名的那一行
        for i, line in enumerate(lines_reverse):
            if line.lstrip() == line and re.search(rf'\b{re.escape(func)}\b', line):
                func_line_idx = i
                break

        if func_line_idx is None:
            return 0  # 没有找到该函数

        base_indent = 0

        # 向下遍历直到遇到下一个缩进为0的函数定义行
        for j in range(func_line_idx + 1, len(lines_reverse)):
            line = lines_reverse[j]
            indent = len(line) - len(line.lstrip())
            if indent == base_indent:
                break  # 到达下一个函数块
            if indent > base_indent:
                called_count += 1

        return called_count
    except Exception:
        return 0

def get_call_counts(filepath: str, function_names: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    计算每个函数在给定 C 文件中的出度（调用其他函数数量）和入度（被其他函数调用的数量）。

    :param filepath: C 文件路径
    :param function_names: 函数名列表
    :return: 字典 {函数名: (调用其他函数数量, 被调用数量)}
    """
    try:
        proc_forward = subprocess.run(['cflow', filepath], capture_output=True, text=True, check=True)
        lines_forward = proc_forward.stdout.splitlines()
    except subprocess.CalledProcessError:
        lines_forward = []

    try:
        proc_reverse = subprocess.run(['cflow', '--reverse', filepath], capture_output=True, text=True, check=True)
        lines_reverse = proc_reverse.stdout.splitlines()
    except subprocess.CalledProcessError:
        lines_reverse = []

    results = {}

    for func in function_names:
        try:
            # 计算出度（调用其他函数数量）
            call_count = 0
            found = False
            base_indent = None
            pattern_func = re.compile(rf"^\s*\d*\s*{re.escape(func)}\(\)")

            for line in lines_forward:
                if not found:
                    if pattern_func.search(line):
                        found = True
                        base_indent = len(line) - len(line.lstrip())
                        continue
                else:
                    indent = len(line) - len(line.lstrip())
                    if indent > base_indent:
                        if re.search(r'\b\w+\(\)', line.strip()):
                            call_count += 1
                    else:
                        break
        except Exception:
            call_count = 0

        # 计算入度（被调用数量）
        called_count = get_called_count_from_reverse_cflow(lines_reverse, func)

        results[func] = (call_count, called_count)

    return results
def evaluate_cohesion_static(code_str: str, out_degree: int) -> float:
  #内聚度（0-1）
  #基于启发式规则1. 函数行数 > 50 行 2. 超过 3 个 if / switch 3.调用超过 5 个不同函数（传入的出度） 规则4：同时包含 copy_to_user 和 copy_from_user# 规则5：包含多个资源操作关键词
    penalty = 0

    # 规则1：函数行数 > 50 行
    lines = code_str.strip().split('\n')
    if len(lines) > 50:
        penalty += 2

    # 规则2：超过 3 个 if / switch
    control_flow_count = len(re.findall(r'\b(if|switch)\b', code_str))
    if control_flow_count > 3:
        penalty += 2

    # 规则3：调用超过 5 个不同函数（传入的出度）
    if out_degree > 5:
        penalty += 2

    # 规则4：同时包含 copy_to_user 和 copy_from_user
    if "copy_to_user" in code_str and "copy_from_user" in code_str:
        penalty += 2

    # 规则5：包含多个资源操作关键词
    resource_keywords = ['alloc', 'free', 'lock', 'unlock']
    count = sum(code_str.count(k) for k in resource_keywords)
    if count > 2:
        penalty += 2

    # 内聚度 = 1 - 惩罚系数 * 0.1（最多 5 项惩罚，总惩罚为 10，最低分为 0.0）
    cohesion = max(0.0, 1.0 - penalty * 0.1)
    return cohesion

