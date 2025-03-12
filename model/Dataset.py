from torch.utils.data.dataset import Dataset
from utils import DataProcess
from Bio import SeqIO
from tqdm import tqdm
import torch


def read_file2data(filepath: str, k: int, word2idx: dict, max_len: int, mode: str):
    """
    读取文件将数据提取出来

    Args:
        filepath (str): 要读取的文件路径
        k (int): k-mer的长度
        word2idx (dict): 单词到索引的映射字典
        max_len (int): 序列的最大长度
        mode (str): 处理模式，例如 "train" 或 "test"

    Returns:
        tuple: 包含数据张量列表和标签列表的元组
    """
    DataTensor = []
    Labels = []
    trim = mode == "train"

    with open(filepath, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))

    pbar = tqdm(records, desc=f"Processing {mode} data")
    for rec in pbar:
        seq, label_id = Read_Parser(rec)
        kmer_tensor = DataProcess.seq2kmer(seq, k, word2idx, max_len, trim)
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


class SeqDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        input_path: str,
        all_dict: Dictionary,
        k: int,
        mode: str,
    ):
        self.Data, self.Label = read_file2data(
            input_path, k, all_dict.kmer2idx, max_len, mode
        )
        self.Data = torch.stack(self.Data)
        self.Label = torch.tensor(self.Label)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label
