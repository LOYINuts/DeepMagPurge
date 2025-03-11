import os
from Bio import SeqIO

FILE_PATH = "/home/lys/gh/DBFiles/dmpdata/art_output/"
OUTPUT_PATH = "/home/lys/gh/DBFiles/dmpdata/all_concat_seq_data.fa"

file_list = os.listdir(FILE_PATH)
with open(OUTPUT_PATH, "w") as fout:
    for file in file_list:
        file_path = os.path.join(FILE_PATH, file)
        with open(file_path, "r", buffering=2048) as handle:
            for rec in SeqIO.parse(handle, "fastq"):
                fout.write(rec.format("fasta"))
        print("Complete fastq:", file)
