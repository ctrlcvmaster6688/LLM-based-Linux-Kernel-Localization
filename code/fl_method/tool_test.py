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
    '../my-languages.so',
    [
        '../tree-sitter-c',
    ]
)
C_LANGUAGE = Language('../my-languages.so', 'c')
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
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    tree = c_parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    function_map = {}  # {函数名: (参数字符串, 参数个数, 函数体)}

    def walk(node):
        stack = [node]
        while stack:
            current_node = stack.pop()
            if current_node.type == 'function_definition':
                declarator = current_node.child_by_field_name('declarator')

                # 提取函数名
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

                    # 提取参数字符串和参数个数
                    param_node = declarator.child_by_field_name('parameters')
                    if param_node:
                        params_text = code[param_node.start_byte:param_node.end_byte]
                        params_text = params_text.strip()[1:-1] if params_text.strip().startswith('(') else params_text
                        params_text = ' '.join(params_text.replace('\n', ' ').split())
                        param_count = len([p for p in params_text.split(',') if p.strip()])
                    else:
                        params_text = ""
                        param_count = 0

                    # 提取函数体
                    function_body = current_node.child_by_field_name('body')
                    if function_body:
                        body_code = code[function_body.start_byte:function_body.end_byte]
                        num_lines = body_code.count('\n')
                        if num_lines >= 5 and '_' in function_name:
                            function_map[function_name] = (params_text, param_count, body_code)

            stack.extend(current_node.children)

    walk(root_node)
    return function_map
def estimate_max_nesting_depth(code: str) -> int:
    max_depth = 0
    current_depth = 0
    in_comment = False
    in_string = False
    i = 0
    while i < len(code):
        c = code[i]
        # 跳过注释
        if code[i:i+2] == '/*':
            in_comment = True
            i += 2
            continue
        elif code[i:i+2] == '*/' and in_comment:
            in_comment = False
            i += 2
            continue
        elif in_comment:
            i += 1
            continue

        # 跳过字符串常量
        if c == '"' and not in_comment:
            in_string = not in_string
            i += 1
            continue
        elif in_string:
            if c == '\\' and i+1 < len(code):  # skip escaped chars
                i += 2
            else:
                i += 1
            continue

        # 计数大括号嵌套
        if c == '{':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif c == '}':
            current_depth -= 1

        i += 1

    return max_depth
def count_local_global_variables(code: str):
    lines = code.splitlines()
    in_function = False
    brace_count = 0
    local_vars = 0
    global_vars = 0

    # 正则：匹配一般变量定义行（允许 static、const、struct 等）
    var_decl_regex = re.compile(
        r'^\s*(static\s+)?(const\s+)?(volatile\s+)?(struct|union|enum)?\s*[a-zA-Z_]\w*\s+[*\s]*[a-zA-Z_]\w*(\s*=\s*[^;]+)?\s*;'
    )

    # 排除的声明前缀
    exclude_prefixes = ('typedef', 'extern', 'register', '#', '//', '/*')

    for line in lines:
        stripped = line.strip()

        # 跳过空行或注释或排除前缀
        if not stripped or any(stripped.startswith(p) for p in exclude_prefixes):
            continue

        # 进入函数体（带括号和左大括号的行）
        if re.match(r'.*\w+\s+\w+\s*\([^)]*\)\s*{', stripped):
            in_function = True
            brace_count = 1
            continue

        if in_function:
            # 在函数体内部
            brace_count += line.count('{')
            brace_count -= line.count('}')
            if brace_count == 0:
                in_function = False
                continue

            if var_decl_regex.match(stripped):
                local_vars += 1
        else:
            # 在函数外部（全局变量）
            if var_decl_regex.match(stripped):
                global_vars += 1

    return local_vars, global_vars
