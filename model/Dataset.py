from torch.utils.data.dataset import Dataset
from utils import DataProcess
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
import torch
from multiprocessing import Pool


def process_record(args):
    rec, k, word2idx, max_len = args
    seq, label_id = Read_Parser(rec)
    kmer_tensor = DataProcess.seq2kmer(seq, k, word2idx, max_len)
    return kmer_tensor, label_id


def read_file2data(
    filepath: str, k: int, word2idx: dict, max_len: int, num_workers: int = 8
):
    with open(filepath, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))

    args_list = [(rec, k, word2idx, max_len) for rec in records]
    with Pool(num_workers) as pool:
        # 使用imap并手动管理进度条
        results = []
        with tqdm(total=len(args_list), desc="转换数据") as pbar:
            for result in pool.imap(process_record, args_list):
                results.append(result)
                pbar.update(1)

    # 分离数据和标签
    DataTensor, Labels = zip(*results)
    return list(DataTensor), list(Labels)


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
        num_workers: int,
    ):
        self.Data, self.Label = read_file2data(
            input_path, k, all_dict.kmer2idx, max_len, num_workers=num_workers
        )
        self.Data = torch.stack(self.Data)
        self.Label = torch.tensor(self.Label)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label
