import re
import json
import re
import subprocess
from tree_sitter import Language, Parser
import os
import matplotlib.patches as patches
import seaborn as sns
import math
import matplotlib.pyplot as plt
import numpy as np
# 初始化 Tree-sitter
Language.build_library(
    '/home/pangyesong/fault_localization/ir_approach/build/my-languages.so',
    [
        '/home/duyiheng/projects/kernel-tcp/parser/tree-sitter-c',  # 替换为实际路径
    ]
)

C_LANGUAGE = Language('/home/pangyesong/fault_localization/build/my-languages.so', 'c')
c_parser = Parser()
c_parser.set_language(C_LANGUAGE)
def get_file_names_for_syzid(file_path, syzid):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 初始化标志和结果列表
    found = False
    file_names = []

    # 遍历文件内容
    for line in lines:
        line = line.strip()
        if line.startswith("Response for case"):
            # 检查当前行是否匹配 syzid
            if line.split()[-1] == syzid:
                found = True
            else:
                found = False
        elif found and line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")):
            # 提取文件名
            file_name = line.split(maxsplit=1)[1]
            file_names.append(file_name)

    return file_names
def extract_function_bodies(file_path):
    with open(file_path, 'r') as f:
        code = f.read()

    tree = c_parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    function_map = {}  # 改为字典存储 {函数名: 函数体}

    def walk(node):
        stack = [node]
        while stack:
            current_node = stack.pop()
            if current_node.type == 'function_definition':
                # 提取函数名
                declarator = current_node.child_by_field_name('declarator')
                function_name_node = None
                temp_stack = [declarator]
                while temp_stack:
                    cur_node = temp_stack.pop()
                    for ch in cur_node.children:
                        if ch.type == 'identifier':
                            function_name_node = ch
                            temp_stack.clear()
                            break
                        if ch.end_byte <= current_node.end_byte:
                            temp_stack.append(ch)

                if function_name_node:
                    function_name = function_name_node.text.decode('utf-8')

                    # 提取函数体
                    function_body = current_node.child_by_field_name('body')
                    if function_body:
                        body_code = code[function_body.start_byte:function_body.end_byte]
                        num_lines = body_code.count('\n')

                        # 条件：函数体 >=5 行且函数名包含下划线
                        if num_lines >= 5 and '_' in function_name:
                            function_map[function_name] = body_code  # 存入字典

            stack.extend(current_node.children)

    walk(root_node)
    return function_map  # 返回 {函数名: 函数体} 的字典
def calculate_cyclomatic_complexity(code: str) -> int:
    """
    计算 C 语言代码的圈复杂度（Cyclomatic Complexity）。
    
    :param code: C 语言函数的字符串内容
    :return: 圈复杂度（整数）
    """
    # 初始化复杂度（最少是 1）
    complexity = 1

    # 关键控制流语句
    keywords = [
        r'\bif\b', r'\belse\s+if\b', r'\bfor\b', r'\bwhile\b', r'\bdo\b',
        r'\bcase\b', r'\bcatch\b', r'\bgoto\b'
    ]

    # 逻辑运算符（&&, ||）
    logical_operators = [r'&&', r'\|\|']

    # 统计关键控制流语句
    for keyword in keywords:
        matches = re.findall(keyword, code)
        complexity += len(matches)

    # 统计逻辑运算符
    for op in logical_operators:
        matches = re.findall(op, code)
        complexity += len(matches)

    return complexity
def halstead_effort(code):
    tokens = re.findall(r'\b\w+\b|[-+*/=(){}[\],.;]', code)

    operators = set(["+", "-", "*", "/", "=", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "!", "(", ")", "{", "}", "[", "]", ";", ","])
    operands = set()

    N1, N2 = 0, 0
    n1_set, n2_set = set(), set()

    for token in tokens:
        if token in operators:
            N1 += 1
            n1_set.add(token)
        else:
            N2 += 1
            n2_set.add(token)

    n1 = len(n1_set)
    n2 = len(n2_set)
    N = N1 + N2
    n = n1 + n2

    if n2 == 0 or n == 0:
        return 0  # 避免除零错误

    V = N * math.log2(n)  # 计算体积
    D = (n1 / 2) * (N2 / n2)  # 计算难度
    E = D * V  # 计算工作量（Effort）

    return E