import os
from Bio import SeqIO


def Label_One_Genomes(genome_path: str, label: str, output_path: str):
    with open(genome_path, "r") as handle:
        with open(output_path, "w") as fout:
            for seq in SeqIO.parse(handle, "fastq"):
                seq.id = "label|" + label + "|" + seq.id
                fout.write(seq.format("fasta"))
        print("Complete " + output_path)


def Label_All_Genomes(gene2label_path: str, input_path: str, output_path: str):
    with open(gene2label_path, "r") as f1:
        for line in f1:
            line = line.strip()
            splits = line.split()
            gene, label = splits[0], splits[1]
            gene_path = os.path.join(input_path, gene + ".fq")
            out_file_path = os.path.join(output_path, gene + "_label_" + label + ".fa")
            Label_One_Genomes(gene_path, label, out_file_path)


def Taxon2Labels(all_species_path: str):
    with open(all_species_path, "r") as fin:
        with open("taxon2label.txt", "a") as fout:
            index = 0
            for line in fin:
                line = line.strip()
                fout.write(line + "\t" + str(index) + "\n")
                index += 1


def Genome2Label(gene2spec_map: str, taxon2idx: dict):
    with open(gene2spec_map, "r") as f1:
        with open("gene2label.txt", "a") as fout:
            for line in f1:
                line = line.strip()
                splits = line.split()
                genome, taxon = splits[0], splits[1]
                idx = taxon2idx[taxon]
                fout.write(genome + "\t" + str(idx) + "\n")


if __name__ == "__main__":
    # taxon2labels("./data/species.txt")
    # taxon2idx = DataProcess.TransferTaxon2Idx("./data/taxon2label.txt")
    # Genome2Label("./data/gene2species.txt", taxon2idx)
    Label_All_Genomes(
        "./data/gene2label.txt",
        "E:/StudyData/DMPdata/art_process",
        "E:/StudyData/DMPdata/label_data",
    )
    pass
