import os
from Bio import SeqIO

FILE_PATH = "./DeepMicrobeGenusArt3"
OUTPUT_PATH = "./all_concat_seq_data3.fasta"

file_list = os.listdir(FILE_PATH)
with open(OUTPUT_PATH, "w") as fout:
    for file in file_list:
        file_path = os.path.join(FILE_PATH, file)
        with open(file_path, "r", buffering=2048) as handle:
            for rec in SeqIO.parse(handle, "fastq"):
                fout.write(rec.format("fasta"))
        print("Complete fastq:", file)
