from torch.utils.data.dataset import Dataset
from utils import DataProcess
import os
from Bio import SeqIO
import random
from tqdm import tqdm
import torch


def Reservoir_sample(records, k: int):
    """蓄水池抽样

    Args:
        records : SeqIO.parse返回的迭代器
        k (int): 采样数
    """
    samples = []
    for i, item in enumerate(records):
        if i < k:
            samples.append(item)
        else:
            # 生成0到i的随机整数
            j = random.randint(0, i)
            if j < k:
                samples[j] = item
        # 如果迭代器元素不足k个，返回全部
    return samples[:k] if k <= len(samples) else samples


def Read_Parser(record: any):
    """将SeqIO.parse的返回的一个rec获取其序列和label

    Args:
        record (any): SeqIO.parse的返回的一个rec
    """
    seq = str(record.seq)
    identifier = record.id
    label_id = int(identifier.split("|")[1])
    return seq, label_id


class Dictionary:
    def __init__(self, kmer_file_path: str, taxon_file_path: str):
        self.kmer2idx = {}
        self.taxon2idx = {}
        self.kmer2idx = DataProcess.TransferKmer2Idx(kmer_file_path)
        self.taxon2idx = DataProcess.TransferTaxon2Idx(taxon_file_path)


class SeqDataset(Dataset):
    def __init__(
        self,
        input_path: str,
        max_len: int,
        all_dict: Dictionary,
        samples_perfile: int,
        k: int,
    ):
        self.max_len = max_len
        self.mydict = all_dict
        self.k = k
        self.samples = samples_perfile
        self.Data, self.Label = self.Convert_files2data(input_path)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label

    def Convert_files2data(
        self,
        input_path: str,
    ):
        file_list = os.listdir(input_path)
        DataTensor = []
        Labels = []
        print("Processing files.....")
        processBar = tqdm(file_list, "处理进度")
        for _, file in enumerate(processBar):
            full_path = os.path.join(input_path, file)
            with open(full_path, "r") as handle:
                records = SeqIO.parse(handle, "fasta")
                samples = Reservoir_sample(records, self.samples)
                for rec in samples:
                    seq, label_id = Read_Parser(rec)
                    kmer_tensor = DataProcess.seq2kmer(
                        seq, self.k, self.mydict.kmer2idx, self.max_len
                    )
                    DataTensor.append(kmer_tensor)
                    Labels.append(label_id)

        DataTensor = torch.stack(DataTensor)
        Labels = torch.tensor(Labels)
        return DataTensor, Labels
