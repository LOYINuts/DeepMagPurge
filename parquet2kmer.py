import model.Dataset as Dataset
from utils import config, DataProcess
import polars as pl
from tqdm import tqdm
if __name__ == "__main__":
    conf = config.load_config("./data/config.yaml")
    if conf is None:
        raise Exception("Error loading configuration")
    all_dict = Dataset.Dictionary(conf["KmerFilePath"], conf["TaxonFilePath"])
    data = pl.read_parquet("E:\\GenomeData\\DMPdata\\all_concat_seq_data.parquet")
    print("parquet file loaded")
    data_list = []
    pbar = tqdm(data.iter_rows(),"processing")
    for row in pbar:
        label, seq = int(row[0]), str(row[1])
        kmer_list = DataProcess.seq2kmer(
            seq=seq,
            k=conf["kmer"],
            word2idx=all_dict.kmer2idx,
            max_len=conf["max_len"],
            trim=True,
        )
        data_list.append([label, kmer_list])
    print("seq2kmer complete")
    processed_data = pl.DataFrame(data_list, schema=["label", "kmer"], orient="row")
    processed_data.write_parquet(conf["TrainDataPath"])
