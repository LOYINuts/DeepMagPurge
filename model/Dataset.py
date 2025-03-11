from torch.utils.data.dataset import Dataset
from utils import DataProcess
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
import torch


def read_file2data(filepath: str, k: int, word2idx: dict, max_len: int):
    """读取文件将数据提取出来

    Args:
        filepath (str): _description_
        k (int): _description_
        word2idx (dict): _description_
        max_len (int): _description_

    Returns:
        _type_: _description_
    """
    DataTensor = []
    Labels = []
    with open(filepath, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        processBar = tqdm(records, desc="转换数据")
        for rec in processBar:
            seq, label_id = Read_Parser(rec)
            kmer_tensor = DataProcess.seq2kmer(seq, k, word2idx, max_len)
            DataTensor.append(kmer_tensor)
            Labels.append(label_id)
    return DataTensor, Labels


def Read_Parser(record):
    """将SeqIO.parse的返回的一个rec获取其序列和label

    Args:
        record : SeqIO.parse的返回的一个rec
    """
    seq = str(record.seq)
    identifier = record.id
    label_id = int(identifier.split("|")[1])
    return seq, label_id


class Dictionary:
    def __init__(self, kmer_file_path: str, taxon_file_path: str):
        self.kmer2idx = DataProcess.TransferKmer2Idx(kmer_file_path)
        self.taxon2idx = DataProcess.TransferTaxon2Idx(taxon_file_path)


# class AllDataset:
#     def __init__(
#         self,
#         input_path: str,
#         test_path: str,
#         max_len: int,
#         all_dict: Dictionary,
#         k: int,
#     ):
#         self.max_len = max_len
#         self.mydict = all_dict
#         self.k = k
#         train_data, train_label, valid_data, valid_label, test_data, test_label = (
#             self.ConvertFile2Data(input_path, test_path)
#         )
#         self.train_dataset = SeqDataset(train_data, train_label)
#         self.valid_dataset = SeqDataset(valid_data, valid_label)
#         self.test_dataset = SeqDataset(test_data, test_label)

#     def ConvertFile2Data(self, input_path: str, test_data_path: str):
#         print("Extracting train and valid data......")
#         Train_DataTensor, Train_Labels = read_file2data(
#             input_path, self.k, self.mydict.kmer2idx, self.max_len
#         )

#         print("Extracting test data......")
#         Test_DataTensor, Test_Labels = read_file2data(
#             test_data_path, self.k, self.mydict.kmer2idx, self.max_len
#         )

#         valid_index = np.random.randint(
#             low=0,
#             high=len(Train_DataTensor),
#             size=int(
#                 len(Train_DataTensor) / 10,
#             ),
#         )
#         valid_index = torch.tensor(valid_index)

#         Train_DataTensor = torch.stack(Train_DataTensor)
#         Train_Labels = torch.tensor(Train_Labels)
#         Test_DataTensor = torch.stack(Test_DataTensor)
#         Test_Labels = torch.tensor(Test_Labels)
#         Valid_DataTensor = Train_DataTensor[valid_index]
#         Valid_Labels = Train_Labels[valid_index]
#         return (
#             Train_DataTensor,
#             Train_Labels,
#             Valid_DataTensor,
#             Valid_Labels,
#             Test_DataTensor,
#             Test_Labels,
#         )



class SeqDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        input_path: str,
        all_dict: Dictionary,
        k: int,
    ):
        self.Data, self.Label = read_file2data(
            input_path, k, all_dict.kmer2idx, max_len
        )
        self.Data = torch.stack(self.Data)
        self.Label = torch.tensor(self.Label)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label
