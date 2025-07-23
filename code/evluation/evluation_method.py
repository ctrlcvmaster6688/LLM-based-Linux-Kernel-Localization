import json
from typing import Dict, List
import re
import  argparse
def process_output_to_responses(input_file: str) -> Dict[str, List[str]]:
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

    return responses

def calculate_metrics(responses, oracle, caseid_list, top_n):

    correct_counts = {n: 0 for n in top_n}
    reciprocal_rank_sum = 0.0
    all_oracle_ranks = []
    first_match_ranks = []
    valid_case_count = 0

    valid_caseids = sorted(set(caseid_list) & set(oracle.keys()) & set(responses.keys()))
    total_cases = len(valid_caseids)

    case_rankings = {}

    for case_id in valid_caseids:
        predicted_files = responses[case_id]
        oracle_files = set(oracle[case_id])
        max_rank = 11
        matched_ranks = [i+1 for i, file in enumerate(predicted_files) if file in oracle_files]
        missed_count = len(oracle_files) - len(matched_ranks)

        current_mar_ranks = matched_ranks + [max_rank] * missed_count
        all_oracle_ranks.extend(current_mar_ranks)

        if matched_ranks:
            min_rank = min(matched_ranks)
            case_rankings[case_id] = min_rank
            reciprocal_rank_sum += 1.0 / min_rank
            first_match_ranks.append(min_rank)
        else:
            case_rankings[case_id] = max_rank
            first_match_ranks.append(max_rank)

        for n in top_n:
            if matched_ranks and min(matched_ranks) <= n:
                correct_counts[n] += 1

        valid_case_count += 1

    accuracies = {n: correct_counts[n] / total_cases if total_cases > 0 else 0.0 for n in top_n}
    mrr = reciprocal_rank_sum / total_cases if total_cases > 0 else 0.0
    mar = sum(all_oracle_ranks) / len(all_oracle_ranks) if all_oracle_ranks else 0.0
    mfr = sum(first_match_ranks) / total_cases if total_cases > 0 else 0.0

    missing_in_responses = set(caseid_list) - set(responses.keys())
    missing_in_oracle = set(caseid_list) - set(oracle.keys())
    if missing_in_responses:
        print(f"警告：{len(missing_in_responses)} 个 caseid 在结果文件中不存在")
    if missing_in_oracle:
        print(f"警告：{len(missing_in_oracle)} 个 caseid 在 oracle 文件中不存在")

    return accuracies, mrr, mar, mfr, case_rankings


def read_caseid_list(filelist_path):
    with open(filelist_path, 'r') as f:
        return {line.strip().split('/')[0] for line in f if line.strip()}
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
            case_id, oracle_functions = line.split(":", 1)
            oracle[case_id] = [f.strip() for f in oracle_functions.split(",")]
    return oracle
def main():
    parser = argparse.ArgumentParser(description="Evaluate Fault Localization Metrics")
    parser.add_argument('--filelist', required=True, help='Path to the case list ')
    parser.add_argument('--result_file', required=True, help='Path to the model result output file')
    parser.add_argument('--oracle_file', required=True, help='Path to the buggy file in json format')
    args = parser.parse_args()
    # Read inputs
    responses = process_output_to_responses(args.result_file)
    oracle=parse_oracle_file(args.oracle_file)
    caseid_list = read_caseid_list(args.filelist)
    # Evaluate
    top_n = [1,3,5,10]
    accuracy, mrr, mar, mfr, case_rankings = calculate_metrics(responses, oracle, caseid_list, top_n)

    for n in top_n:
        print(f"Top-{n}: {accuracy[n]:.2%}")
    print(f"MRR: {mrr:.4f}")
    print(f"MAR: {mar:.4f}")
    print(f"MFR: {mfr:.4f}")
main()