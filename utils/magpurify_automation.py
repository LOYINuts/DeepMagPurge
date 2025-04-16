import os
import subprocess
import multiprocessing


def run_magpurify_commands(input_file, output_dir):
    commands = [
        f"magpurify phylo-markers {input_file} {output_dir}",
        f"magpurify clade-markers {input_file} {output_dir}",
        f"magpurify tetra-freq {input_file} {output_dir}",
        f"magpurify gc-content {input_file} {output_dir}",
        f"magpurify known-contam {input_file} {output_dir}",
        f"magpurify clean-bin {input_file} {output_dir} {os.path.splitext(input_file)[0]}_cleaned.fna",
    ]

    for command in commands:
        try:
            print(f"正在执行命令: {command}")
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"执行命令 {command} 时出错: {e}")


def process_single_fna_file(input_file):
    output_dir = os.path.join(os.path.dirname(input_file), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    run_magpurify_commands(input_file, output_dir)


def process_fna_files(input_dir):
    fna_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".fna") or file.endswith(".fa") or file.endswith(".fasta"):
                fna_files.append(os.path.join(root, file))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(process_single_fna_file, fna_files)
    pool.close()
    pool.join()


if __name__ == "__main__":
    input_directory = "example"  # 请替换为实际包含 .fna 文件的目录
    process_fna_files(input_directory)
