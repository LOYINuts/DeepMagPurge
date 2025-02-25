from torch.utils.data.dataset import Dataset
from utils import DataProcess
import os
from Bio import SeqIO
import random
from tqdm import tqdm
import torch

def ReservoirSample(records, k: int):
    """蓄水池抽样

    Args:
        records : SeqIO.parse返回的迭代器
        k (int): 采样数
    """
    train_samples = []
    test_samples = []
    tk = int(k/2)
    for i, item in enumerate(records):
        if i < k:
            train_samples.append(item)
        if i < tk:
            test_samples.append(item)
        if i > k:
            # 生成0到i的随机整数
            j = random.randint(0, i)
            if j < k:
                train_samples[j] = item
        if i > tk:
            j = random.randint(0,i)
            if j < tk:
                test_samples[j] = item
        # 如果迭代器元素不足k个，返回全部
    return train_samples,test_samples

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


class AllDataset:
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
        train_data, train_label, test_data, test_label = self.Convert_files2data(
            input_path
        )
        self.train_dataset = SeqDataset(train_data, train_label)
        self.test_dataset = SeqDataset(test_data, test_label)

    def Convert_files2data(
        self,
        input_path: str,
    ):
        file_list = os.listdir(input_path)
        Train_DataTensor = []
        Train_Labels = []
        Test_DataTensor = []
        Test_Labels = []
        print("Processing files.....")
        processBar = tqdm(file_list, "处理进度")
        for _, file in enumerate(processBar):
            full_path = os.path.join(input_path, file)
            with open(full_path, "r") as handle:
                records = SeqIO.parse(handle, "fasta")
                train_samples,test_samples = ReservoirSample(records, k=self.samples)
                # 训练集
                for rec in train_samples:
                    seq, label_id = Read_Parser(rec)
                    kmer_tensor = DataProcess.seq2kmer(
                        seq, self.k, self.mydict.kmer2idx, self.max_len
                    )
                    Train_DataTensor.append(kmer_tensor)
                    Train_Labels.append(label_id)
                # 测试集
                for rec in test_samples:
                    seq, label_id = Read_Parser(rec)
                    kmer_tensor = DataProcess.seq2kmer(
                        seq, self.k, self.mydict.kmer2idx, self.max_len
                    )
                    Test_DataTensor.append(kmer_tensor)
                    Test_Labels.append(label_id)

        Train_DataTensor = torch.stack(Train_DataTensor)
        Test_DataTensor = torch.stack(Test_DataTensor)
        Train_Labels = torch.tensor(Train_Labels)
        Test_Labels = torch.tensor(Test_Labels)
        return Train_DataTensor, Train_Labels, Test_DataTensor, Test_Labels


class SeqDataset(Dataset):
    def __init__(
        self,
        data,
        label,
    ):
        self.Data = data
        self.Label = label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label
