from torch.utils.data.dataset import Dataset
from utils import DataProcess


class Dictionary:
    def __init__(self):
        self.kmer2idx = {}
        self.taxon2idx = {}

    def init_dict(self, kmer_file_path: str, taxon_file_path: str):
        self.kmer2idx = DataProcess.TransferKmer2Idx(kmer_file_path)
        self.taxon2idx = DataProcess.TransferTaxon2Idx(taxon_file_path)


class SeqDataset(Dataset):
    pass
