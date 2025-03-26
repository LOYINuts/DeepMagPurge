from torch.utils.data.dataset import Dataset
from utils import DataProcess
from Bio import SeqIO
from tqdm import tqdm
import torch
import numpy as np
import polars as pl


def file2data(filepath: str) -> tuple[list[torch.Tensor], list[int]]:
    """
    读取文件将数据提取出来

    Args:
        filepath (str): 要读取的文件路径

    Returns:
        tuple: 包含数据张量列表和标签列表的元组
    """
    DataTensor = []
    Labels = []
    pldata = pl.read_parquet(filepath)
    pbar = tqdm(pldata.iter_rows(),desc="Processing parquet", total=len(pldata))
    for row in pbar:
        label, kmer = int(row[0]), torch.as_tensor(row[1])
        DataTensor.append(kmer)
        Labels.append(label)
    return DataTensor, Labels


class Dictionary:
    def __init__(self, kmer_file_path: str, taxon_file_path: str):
        self.kmer2idx = DataProcess.TransferKmer2Idx(kmer_file_path)
        self.taxon2idx = DataProcess.TransferTaxon2Idx(taxon_file_path)


class PQSeqDataset(Dataset):
    def __init__(
        self,
        input_path: str,
    ):
        self.Data, self.Label = file2data(input_path)
        self.Data = torch.stack(self.Data)
        self.Label = torch.tensor(self.Label)
        print("Dataset complete")

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label


class PredictSeqDataset(Dataset):
    def __init__(
        self,
        k: int,
        all_dict: Dictionary,
        record,
        sub_seq_len=150,
        step=75,
        max_samples=100,
        threshold=5000,
    ) -> None:
        self.k = k
        self.name = record.id
        self.sub_seq_len = sub_seq_len
        self.step = step
        self.all_dict = all_dict
        self.threshold = threshold
        self.Data = []
        seq = str(record.seq)

        # 序列必须比送入模型的子序列长
        if len(seq) >= sub_seq_len:
            self.Data = self._generate_windows(seq, max_samples)
            self.Data = torch.stack(self.Data)
        else:
            raise ValueError(
                f"Sequence length ({len(seq)}) is shorter than {sub_seq_len}"
            )

    def _generate_windows(self, seq: str, max_samples: int) -> list[torch.Tensor]:
        """智能生成窗口策略"""
        total_possible = len(seq) - self.sub_seq_len + 1
        # 短序列模式：全量滑动窗口
        if len(seq) <= self.threshold:
            return [
                self._process_window(seq, i)
                for i in range(0, total_possible, self.step)
            ]

        # 长序列模式：随机采样
        sample_size = min(max_samples, total_possible)
        positions = np.random.choice(total_possible, size=sample_size, replace=False)
        return [self._process_window(seq, pos) for pos in sorted(positions)]

    def _process_window(self, seq: str, start_pos: int) -> torch.Tensor:
        """核心窗口处理函数"""
        end_pos = start_pos + self.sub_seq_len
        sub_seq = seq[start_pos:end_pos]
        return torch.tensor(
            DataProcess.seq2kmer(
                seq=sub_seq,
                k=self.k,
                word2idx=self.all_dict.kmer2idx,
                max_len=self.sub_seq_len - self.k + 1,
            )
        )

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        return self.Data[index]


class BenchmarkDataset(Dataset):
    """benchmark使用的dataset，同样可作为什么都不进行处理的数据集"""

    def __init__(
        self,
        k: int,
        file_path: str,
        all_dict: Dictionary,
        label: int,
        max_len: int = 150,
    ) -> None:
        self.Data = []
        self.Label = []
        with open(file=file_path, mode="r") as handle:
            for rec in SeqIO.parse(handle, "fasta"):
                seq = str(rec.seq)
                kmer_tensor = DataProcess.seq2kmer(
                    seq=seq, k=k, word2idx=all_dict.kmer2idx, max_len=max_len
                )
                self.Data.append(kmer_tensor)
                self.Label.append(label)

        self.Data = torch.stack(self.Data)
        self.Label = torch.tensor(self.Label)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        return self.Data[index], self.Label[index]
