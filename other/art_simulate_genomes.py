import subprocess
import os
import time

from multiprocessing import Process

# 定义路径常量（使用原始字符串处理Windows路径）
ART_PATH = r"/home/lys/gh/softwares/art_bin_MountRainier/art_illumina"
INPUT_DIR = "/home/lys/gh/DBFiles/dmpdata/labeled_genome_genus/"
OUTPUT_DIR = "/home/lys/gh/DBFiles/dmpdata/art_output/"
NUM_WORKERS = 4


def process_files(file_list: list[str]):
    # 遍历输入目录下的所有文件
    for file in file_list:
        # 解析基因标识（使用Path对象处理文件名）
        splits = file.split(".")
        out_prefix = "_".join([splits[0],"pari_end"])

        # 构造art命令参数列表
        input_path = INPUT_DIR + file
        output_path = OUTPUT_DIR + out_prefix
        cmd_args = [
            ART_PATH,
            "-ss",
            "HS25",
            "-i",
            input_path,  # 转换为绝对路径
            "-l",
            "150",
            "-na",
            "-f",
            "5",
            "-p",
            "-s",
            "50",
            "-m",
            "400",
            "-o",
            output_path,
        ]

        try:
            # 同步执行命令（shell=False更安全）
            p = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,  # 不显示控制台窗口
            )
            p.wait()
            # 输出执行结果
            print(f"处理完成: {out_prefix}")
        except subprocess.CalledProcessError as e:
            print(f"错误：处理 {out_prefix} 失败，返回码 {e.returncode}")
            print(f"错误日志:\n{e.stderr[:500]}...")
        except Exception as e:
            print(f"意外错误: {str(e)}")


if __name__ == "__main__":
    file_list = os.listdir(INPUT_DIR)
    num_files = len(file_list)
    step = int(num_files / 4)
    files4process = []
    for i in range(NUM_WORKERS):
        start, end = int(i * step), int((i + 1) * step)
        if i != NUM_WORKERS - 1:
            files4process.append(file_list[start:end])
        else:
            files4process.append(file_list[start:])
    workers = []
    for i in range(NUM_WORKERS):
        workers.append(Process(target=process_files, args=(list(files4process[i]),)))
        workers[i].start()
