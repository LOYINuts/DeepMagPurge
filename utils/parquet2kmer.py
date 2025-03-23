from ..model import Dataset
from . import config, DataProcess
import polars as pl

if __name__ == "__main__":
    conf = config.load_config("./data/config.yaml")
    if conf is None:
        raise Exception("Error loading configuration")
    all_dict = Dataset.Dictionary(conf["KmerFilePath"], conf["TaxonFilePath"])
    trim = True
    data = pl.read_parquet(conf["TrainDataPath"])
    data_list = []
    for row in data.iter_rows():
        label, seq = int(row[0]), str(row[1])
        kmer_list = DataProcess.seq2kmer(
            seq=seq,
            k=conf["kmer"],
            word2idx=all_dict.kmer2idx,
            max_len=conf["max_len"],
            trim=True,
        )
        data_list.append([label, kmer_list])
    processed_data = pl.DataFrame(data_list, schema=["label", "kmer"], orient="row")
    processed_data.write_parquet(conf["TrainDataPath"])
