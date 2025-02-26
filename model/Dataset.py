from torch.utils.data.dataset import Dataset
from utils import DataProcess
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
import torch


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
        k: int,
    ):
        self.max_len = max_len
        self.mydict = all_dict
        self.k = k
        train_data, train_label, test_data, test_label = self.ConvertFile2Data(
            input_path
        )
        self.train_dataset = SeqDataset(train_data, train_label)
        self.test_dataset = SeqDataset(test_data, test_label)

    def ConvertFile2Data(self, input_path: str):
        Train_DataTensor = []
        Train_Labels = []
        Test_DataTensor = []
        Test_Labels = []
        print("Extracting data......")
        with open(input_path, "r") as handle:
            records = list(SeqIO.parse(handle, "fasta"))
            processBar = tqdm(records, desc="转换序列")
            for rec in processBar:
                seq, label_id = Read_Parser(rec)
                kmer_tensor = DataProcess.seq2kmer(
                    seq, self.k, self.mydict.kmer2idx, self.max_len
                )
                Train_DataTensor.append(kmer_tensor)
                Train_Labels.append(label_id)

        test_index = np.random.randint(
            low=0, high=len(Train_DataTensor), size=int(len(Train_DataTensor) / 10)
        )
        Test_DataTensor = Train_DataTensor[test_index]
        Test_Labels = Test_Labels[test_index]
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
