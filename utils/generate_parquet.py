import multiprocessing
import os
from Bio import SeqIO
import polars as pl

FILE_PATH = "/home/lys/gh/DBFiles/dmpdata/art_output/"
OUTPUT_PATH = "/home/lys/gh/DBFiles/dmpdata/all_concat_seq_data.parquet"

def Read_Parser(record):
    """将 SeqIO.parse 的返回的一个 rec 获取其序列和 label

    Args:
        record : SeqIO.parse 的返回的一个 rec
    """
    seq = str(record.seq)
    identifier = record.id
    label_id = int(identifier.split("|")[1])
    return seq, label_id


def process_file(file):
    full_path = os.path.join(FILE_PATH, file)
    data_list = []
    with open(full_path, "r") as handle:
        for rec in SeqIO.parse(handle, "fastq"):
            seq, label = Read_Parser(rec)
            data_list.append([label, seq])
    print(f"{file} complete")
    return data_list


if __name__ == "__main__":
    files = os.listdir(FILE_PATH)
    with multiprocessing.Pool() as pool:
        results = pool.map(process_file, files)

    flat_results = [item for sublist in results for item in sublist]
    data = pl.DataFrame(flat_results,schema=["label","seq"],orient="row")
    data.write_parquet(OUTPUT_PATH)
