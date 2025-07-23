
# LLM-based Linux Kernel Fault Localization

This repository contains the implementation of **COHIKER** and the dataset used in our experiments for fault localization in the Linux kernel using large language models (LLMs). The entire fault localization process includes three major phases:

1. **Data Preparation**: Generates contrastive mutated test cases and reproduces crashes in the kernel.
2. **File-level Fault Localization**: Identifies potentially faulty files.
3. **Function-level Fault Localization**: Further narrows down to faulty functions within the suspected files.

---

## 📁 Dataset Structure

The `datasets/` folder contains the data used throughout the experiments:

- `bug_config/`: Kernel configuration files that can trigger known bugs.
- `bug_report/`: Crash and bug reports.
- `testcases/`: Test programs that reproduce the kernel crashes.
- Other auxiliary information such as kernel versions, buggy files, and buggy methods.

---

## 🧠 Code Overview

All source code is located under the `code/` directory. The subdirectories and their responsibilities are as follows:

### `data_prepare/` — Data Preparation

- `mutation_generate.py`: Generates contrastive mutated test programs for each case.
- `testcase_reproduce.py`: Validates whether the mutations avoid triggering the original crash.

### `fl_file/` — File-level Fault Localization

- `fl_file.py`: Performs fault localization at the file level using LLMs.

### `fl_method/` — Function-level Fault Localization

- `fliter_model.py`: Trains a defect prediction model to filter low-risk functions.
- `func_fliter.py`: Filters and selects candidate functions based on the file-level results and defect prediction.
- Additional utility scripts for LLM-based function ranking.

### `prompt/` — Prompt Engineering

- Contains default prompts and reference output formats used in both localization phases.

---

## 🚀 Workflow

### 1. Contrastive Mutation Generation (Optional)

```bash
python mutation_generate.py       # Generates 10 mutated versions per test case
python testcase_reproduce.py      # Verifies whether mutations avoid crashing
```

> Outputs:
> - Mutated test cases in the `output/` folder
> - Mutation verification results in `mutation_result_all.txt` and `case_all.txt`

Alternatively, you can **skip this step** and directly use the prepared results in `/results`.
---

### 2. File-level Fault Localization

```bash
python fl_file.py
```
---
### 3. Function-level Fault Localization

```bash
# Step 1: Extract candidate functions
python func_fliter.py

# Step 2: Rank functions using LLM
python fl_method.py
```
---
### 4. Evaluation

```bash
# File-level localization evaluation
python evaluation_file.py

# Function-level localization evaluation
python evaluation_method.py
```

---

## ⚡ Quick Start

To skip mutation generation and use the provided results for fault localization:

### Step 0:Set up your own path
export FLKERNEL=/path/to/your/project/FLKERNEL
for example:export FLKERNEL=/root/FLKERNEL

### Step 1: File-Level Localization

```bash
python $FLKERNEL/code/fl_file/fl_file.py \
  --case_list=$FLKERNEL/results/case_all.txt \
  --report_dir=$FLKERNEL/datasets/bug_report \
  --testcase_dir=$FLKERNEL/datasets/testcases \
  --output_extend=$FLKERNEL/results/extend_file_result.txt \
  --mutation_output=$FLKERNEL/results/mutation_result_all.txt \
  --localize_output=$FLKERNEL/results/fl_file_result.txt
```

### Step 2: Function-Level Localization

#### Generate Candidate Functions

```bash
python $FLKERNEL/code/fl_method/func_fliter.py \
  --filelist=$FLKERNEL/results/case_all.txt \
  --report_dir=$FLKERNEL/datasets/bug_report \
  --testcase_dir=$FLKERNEL/datasets/testcases \
  --mutation_output=$FLKERNEL/results/mutation_result_all.txt \
  --work_dir=</path/to/linux/source> \
  --result_file=$FLKERNEL/results/fl_file_result.txt \
  --output_dir=$FLKERNEL/results/candidate_functions.txt
```

#### Perform Localization

```bash
python $FLKERNEL/code/fl_method/fl_method.py \
  --filelist=$FLKERNEL/results/case_all.txt \
  --report_dir=$FLKERNEL/datasets/bug_report \
  --testcase_dir=$FLKERNEL/datasets/testcases \
  --func_list=$FLKERNEL/results/candidate_functions.txt \
  --mutation_output=$FLKERNEL/results/mutation_result_all.txt \
  --output_dir=$FLKERNEL/results/fl_method_result.txt
```

---

## 📬 Contact

If you have questions or encounter issues, feel free to open an issue or contact the maintainer.
