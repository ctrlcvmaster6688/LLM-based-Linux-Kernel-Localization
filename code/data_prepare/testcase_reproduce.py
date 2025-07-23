import os
import subprocess
import time
import logging
import re
import argparse
import random

logging.basicConfig(filename='repro.log', level=logging.INFO, format='%(asctime)s %(message)s')
oracle_file_path='../../datasets/oracle_content.txt'
syzkaller_path=""#path to syzkaller forlder
def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error(f"Command failed: {command}")
        logging.error(result.stdout.decode())
        logging.error(result.stderr.decode())
    else:
        logging.info(f"Command succeeded: {command}")
    return result

def execute_cmd_vm(cmd,image_path,log_path):
    ssh_command = f'ssh -i {image_path}bullseye.id_rsa -p 10022 -o "StrictHostKeyChecking no" -t root@localhost "{cmd}" >> {log_path} '
    res = run_command(ssh_command)
    logging.info(f"虚拟机命令执行-{cmd}")
    return res
def run_make_commands(directory):
    try:
        run_command('make olddefconfig', cwd=directory)
        logging.info(f"开始编译内核: {directory}")
        start_time = time.time()
        run_command(f'make -j{os.cpu_count()}', cwd=directory)
        end_time = time.time()
        logging.info(f"内核编译完成: {directory}, 耗时 {end_time - start_time:.2f} 秒")
    except subprocess.CalledProcessError as e:
        logging.error(f"运行 make 命令时出错: {directory}: {e}")
def switch_to_branch(branch, directory):
    try:
        run_command(f'git checkout -f {branch}', cwd=directory)
        logging.info(f"已切换到 '{branch}' 分支: {directory}")
    except subprocess.CalledProcessError as e:
        logging.error(f"切换分支时出错: {directory}: {e}")
def write_configs(file_path, configs):
    with open(file_path, 'a') as f:
        for config in configs:
            f.write(config + "\n")
    logging.info(f"已将配置项写入 '{file_path}' 文件中.")

def start_qemu(folder_name,image_path):
    vm_log_path =f"{folder_name}/vm.log"
    if os.path.exists(vm_log_path):
        os.remove(vm_log_path)
    kernel_path = f"{folder_name}/arch/x86/boot/bzImage"
    qemu_command = [
        "qemu-system-x86_64",
        "-m", "4G",
        "-smp", "2",
        "-kernel", kernel_path,
        "-append", "console=ttyS0 root=/dev/sda earlyprintk=serial net.ifnames=0" ,
        "-drive", f"file={image_path}/bullseye.img,format=raw",
        "-net", "user,host=10.0.2.10,hostfwd=tcp:127.0.0.1:10022-:22",
        "-net", "nic,model=e1000",
        "-enable-kvm",
        "-nographic",
        "-pidfile", f"{folder_name}/vm.pid"
    ]

    log_file = vm_log_path
    with open(log_file, 'w') as log:
        subprocess.Popen(qemu_command, stdout=log, stderr=subprocess.STDOUT)
    logging.info(f"QEMU 启动中")
    time.sleep(60)
    print("qemu 启动成功")
def trans_and_excute(repro_exe_path,image_path,vm_log_path):
    if os.path.exists(repro_exe_path):
        scp_command = (
            f"scp -i {image_path}/bullseye.id_rsa -P 10022 -o \"StrictHostKeyChecking no\" {repro_exe_path} root@localhost:/root/repro.txt"
        )
        res= run_command(scp_command)
        if res.returncode != 0:
            logging.error("SCP 传输文件失败")
            return False
        logging.info(f"复现文件已传输到虚拟机中-{repro_exe_path}")
    else:
        logging.error("syz文件不存在.")
        return False
    tool_scp_command1 = (
        f"scp -i {image_path}/bullseye.id_rsa -P 10022 -o \"StrictHostKeyChecking no\" {syzkaller_path}/bin/linux_amd64/syz-execprog root@localhost:/root"
    )
    res= run_command(tool_scp_command1)
    tool_scp_command2 = (
        f"scp -i {image_path}/bullseye.id_rsa -P 10022 -o \"StrictHostKeyChecking no\" {syzkaller_path}/bin/linux_amd64/syz-executor root@localhost:/root"
    )
    res= run_command(tool_scp_command2)
    time.sleep(5)
    run_cmd = f"./syz-execprog -repeat 10 repro.txt"
    runssh_command = f'ssh -i {image_path}/bullseye.id_rsa -p 10022 -o "StrictHostKeyChecking no" -t root@localhost "{run_cmd}" >> {vm_log_path}'
    subprocess.Popen(runssh_command,shell=True, text=True)
    time.sleep(60)
    logging.info("repro执行完毕")
    rm_cmd=f"rm repro.txt"
    execute_cmd_vm(rm_cmd,image_path,vm_log_path)

def check_crash_in_log(folder_name,vm_log_path):

    def load_oracle(oracle_file_path):
        oracle_data = {}
        if not os.path.exists(oracle_file_path):
            logging.error(f"oracle文件不存在: {oracle_file_path}")
            return oracle_data

        with open(oracle_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    fn, keyword = parts
                    oracle_data[fn.strip()] = keyword.strip()
        return oracle_data
    if not os.path.exists(vm_log_path):
        logging.error(f"虚拟机日志文件不存在: {vm_log_path}")
        return False
    oracle_data = load_oracle(oracle_file_path)
    if not oracle_data:
        logging.error("未能加载oracle数据")
        return False

    if folder_name not in oracle_data:
        logging.error(f"oracle数据中不存在文件夹名: {folder_name}")
        return False
    keyword = oracle_data[folder_name]
    with open(vm_log_path, 'r', errors='ignore') as log_file:
        log_content = log_file.read()
        if keyword in log_content:
            logging.info(f"在日志中找到匹配的关键词: {keyword}")
            updated_content = log_content.replace(keyword, "")
            with open(vm_log_path,"w") as file:
                file.write(updated_content)
            return True
    return False
def kill_qemu_process(folder_name):
    pid_file = f"{folder_name}/vm.pid"
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            pid = f.read().strip()
            run_command(f"kill -9 {pid}")
            logging.info(f"虚拟机已终止 (PID: {pid}): {folder_name}")
            os.remove(pid_file)
            logging.info(f"PID 文件已删除: {pid_file}")
    else:
        logging.error(f"虚拟机 PID 文件不存在: {folder_name}")

def parse_syzid_commit_file(file_path):
    result = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            parts = line.split(':', 1)
            syzid = parts[0].strip()
            commit = parts[1].strip()
            result[syzid] = commit
    return result
def process_syz_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    processed_lines = []
    for line in lines:
        line = line.rstrip()
        if line.startswith("#"):
            processed_lines.append(line)
        else:
            processed_lines.append(line + "#")
    while processed_lines and processed_lines[-1] == "":
        processed_lines.pop()

    with open(input_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(processed_lines) + "\n")
def main():
    commit_map=parse_syzid_commit_file("../../datasets/case_commit.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_list', required=True, help='Path to case list')
    parser.add_argument('--mutation_case_dir', required=True, help='Path to save mutation testcase')
    parser.add_argument('--image_path', required=True, help='Path to qemu image')
    parser.add_argument('--output_file', required=True, help='Path to save outputs')
    parser.add_argument('--linux_dir', required=True, help='Path to linux kernel forlder')
    parser.add_argument('--bug_config_dir', required=True, help='Path to bug config forlder')
    args = parser.parse_args()
    work_dir = args.linux_dir
    image_path = args.image_path
    bug_config_dir = args.bug_config_dir
    config_dir = f"{work_dir}/.config"
    testcase_path=args.mutation_case_dir
    vm_log_path=f"{work_dir}/vm.log"
    with open(args.case_list,"r") as f:
        entries = [line.strip().split('/') for line in f]
    for entry in entries:
        syzid = entry[0]
        logging.info(f"Processing syzid: {syzid}")
        kill_qemu_process(work_dir)
        branch = commit_map[syzid]
        if branch == None:
            logging.info(f"{syzid} is not upstream pass!")
            continue
        switch_to_branch(branch, work_dir)
        bugconfig_dir = os.path.join(bug_config_dir, f"{syzid}.txt")
        if not os.path.exists(bugconfig_dir):
            continue
        with open(bugconfig_dir, "r") as file:
            bug_config = file.read()
            bug_config = re.sub('CONFIG_DEBUG_INFO=y', '# CONFIG_DEBUG_INFO is not set', bug_config)
            bug_config = re.sub('CONFIG_DEBUG_INFO_DWARF4=y', '# CONFIG_DEBUG_INFO_DWARF4 is not set', bug_config)
            bug_config = re.sub('CONFIG_KASAN=y', '# CONFIG_KASAN is not set', bug_config)
            bug_config = re.sub('CONFIG_KCOV=y', '# CONFIG_KCOV is not set', bug_config)
        with open(config_dir, "w") as target:
            target.write(bug_config)
        run_make_commands(work_dir)
        logging.info("start repro mutate case")
        succ=[]
        fail=[]
        for i in range(1,11):
            logging.info(f"processing-{i}")
            start_qemu(work_dir,image_path)
            with open(vm_log_path,"r") as f:
                log_content=f.read()
            if "Rebooting" in log_content:
                result=False
                fail.append(i)
                continue
            if not "syzkaller login:" in log_content:
                kill_qemu_process(work_dir)
                start_qemu(work_dir,image_path)
            syz_path=os.path.join(testcase_path,syzid,f"{syzid}_{i}.syz")
            process_syz_file(syz_path)
            trans_and_excute(syz_path,image_path,vm_log_path)
            result=check_crash_in_log(syzid,vm_log_path)
            if result:
                succ.append(i)
            else:
                fail.append(i)
            kill_qemu_process(work_dir)
        kill_qemu_process(work_dir)
        succ_sample = random.sample(succ, min(len(succ), 3)) + [0] * (3 - len(succ))
        fail_sample = random.choice(fail) if fail else 0
        with open(args.output_file,"a") as f:
            f.write(f"{syzid}/")
            f.write("/".join(map(str, succ_sample + [fail_sample])))
            f.write("\n")
if __name__ == "__main__":
    main()





