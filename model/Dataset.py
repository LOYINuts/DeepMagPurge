from torch.utils.data.dataset import Dataset
from utils import DataProcess, config
from Bio import SeqIO
from tqdm import tqdm
import torch
import multiprocessing as mp


def process_batch(args):
    batch_records, k, word2idx, max_len, trim = args
    batch_tensors = []
    batch_labels = []
    for rec in batch_records:
        seq, label_id = Read_Parser(rec)
        tensor = DataProcess.seq2kmer(seq, k, word2idx, max_len, trim)
        batch_tensors.append(tensor)
        batch_labels.append(label_id)
    return batch_tensors, batch_labels


def read_file2data_mp(
    filepath: str, k: int, word2idx: dict, max_len: int, mode: str, batch_size=1000
):
    mp.set_start_method("spawn")
    trim = mode == "train"
    with open(filepath, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]

    with mp.Pool(processes=4) as pool:
        # 使用imap+进度条代替map
        results = list(tqdm(
            pool.imap(process_batch, [(batch, k, word2idx, max_len, trim) for batch in batches]),
            total=len(batches),
            desc="Processing batches"
        ))

    DataTensor = []
    Labels = []
    for tensors, labels in results:
        DataTensor.extend(tensors)
        Labels.extend(labels)
    return DataTensor, Labels


def read_file2data(filepath: str, k: int, word2idx: dict, max_len: int, mode: str):
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
    trim = mode == "train"

    with open(filepath, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))

    pbar = tqdm(records, desc=f"Processing {mode} data")
    for rec in pbar:
        seq, label_id = Read_Parser(rec)  # 调整 Read_Parser 处理 pyfastx 对象
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
        self.Data, self.Label = read_file2data_mp(
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
