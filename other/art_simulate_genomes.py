import subprocess
import os
import time

from multiprocessing import Process, Pool

# 定义路径常量（使用原始字符串处理Windows路径）
ART_PATH = r"/home/lys/gh/softwares/art_bin_MountRainier/art_illumina"
INPUT_DIR = "/home/lys/gh/DBFiles/dmpdata/labeled_genome_genus/"
OUTPUT_DIR = "/home/lys/gh/DBFiles/dmpdata/art_output/"
NUM_WORKERS = os.cpu_count()

def process_single_file(file: str):
    """处理单个文件的函数（适配进程池）"""
    # 生成输出前缀（示例：将文件名拆分为"genus_pari_end"）
    splits = file.split(".")
    out_prefix = "_".join([splits[0], "pari_end"])

    # 构造输入输出路径
    input_path = os.path.join(INPUT_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, out_prefix)

    # ART命令参数
    cmd_args = [
        ART_PATH,
        "-ss",
        "HS25",
        "-i",
        input_path,
        "-l",
        "150",
        "-na",  # 不生成ALN文件（节省I/O）
        "-f",
        "5",
        "-p",  # 启用配对端模式
        "-s",
        "50",
        "-m",
        "400",
        "-o",
        output_path,
    ]

    try:
        # 执行命令（禁用shell以提升安全性）
        result = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,  # 自动检查返回码
        )
        print(f"处理成功: {out_prefix}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {out_prefix} 失败 (返回码 {e.returncode})")
        print(f"错误日志（截断）: {e.stderr[:200]}...")
        return False
    except Exception as e:
        print(f"意外错误: {out_prefix} - {str(e)}")
        return False


if __name__ == "__main__":
    file_list = os.listdir(INPUT_DIR)
    if NUM_WORKERS is None:
        NUM_WORKERS = 8
    else:
        NUM_WORKERS = NUM_WORKERS // 4
    with Pool(NUM_WORKERS) as pool:
        results = pool.imap_unordered(process_single_file, file_list, chunksize=5)
        # 可选：统计成功/失败数量
        success = 0
        for result in results:
            if result:
                success += 1
        print(f"完成！成功处理 {success}/{len(file_list)} 个文件")
