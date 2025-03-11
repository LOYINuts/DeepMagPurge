import multiprocessing
import os
from Bio import SeqIO

FILE_PATH = "/home/lys/gh/DBFiles/dmpdata/art_output/"
OUTPUT_PATH = "/home/lys/gh/DBFiles/dmpdata/all_concat_seq_data.fa"
NUM_WORKERS = os.cpu_count()

def process_file(file):
    """处理单个文件，返回FASTA数据和文件名"""
    file_path = os.path.join(FILE_PATH, file)
    fasta_data = []
    try:
        with open(file_path, "r") as handle:
            for rec in SeqIO.parse(handle, "fastq"):
                fasta_data.append(rec.format("fasta"))
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return "", file
    return "".join(fasta_data), file


if __name__ == "__main__":
    file_list = os.listdir(FILE_PATH)
    # 创建进程池（默认使用所有CPU核心）
    if NUM_WORKERS is None:
        NUM_WORKERS = 8
    else:
        NUM_WORKERS = NUM_WORKERS // 4
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        # 使用imap_unordered提升处理效率（不保证顺序）
        results = list(pool.imap_unordered(process_file, file_list, chunksize=6))
        success = 0
        with open(OUTPUT_PATH, "w") as fout:
            for data, file in results:
                success += 1
                if data:  # 忽略空数据（处理失败的情况）
                    fout.write(data)
                print(f"Complete fastq: {file}")

        print(f"完成！成功处理 {success}/{len(file_list)} 个文件")
