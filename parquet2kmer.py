import model.Dataset as Dataset
from utils import config, DataProcess
import polars as pl
from tqdm import tqdm
import os

if __name__ == "__main__":
    conf = config.load_config("./data/config.yaml")
    if conf is None:
        raise Exception("Error loading configuration")
    if os.path.exists(conf["TrainDataPath"]) is False:
        os.makedirs(conf["TrainDataPath"])
    all_dict = Dataset.Dictionary(conf["KmerFilePath"], conf["TaxonFilePath"])
    data = pl.read_parquet(conf["TempParquet"])
    data = data.sample(n=len(data),shuffle=True)
    print("parquet file loaded")
    # 分批次处理数据
    batch_size = 2048*100  # 可根据实际情况调整批次大小
    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

    for batch_idx in tqdm(range(num_batches), "Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]

        data_list = []
        for row in batch.iter_rows():
            label, seq = int(row[0]), str(row[1])
            kmer_list = DataProcess.seq2kmer(
                seq=seq,
                k=conf["kmer"],
                word2idx=all_dict.kmer2idx,
                max_len=conf["max_len"],
                trim=True,
            )
            data_list.append([label, kmer_list])

        processed_batch = pl.DataFrame(
            data_list, schema=["label", "kmer"], orient="row"
        )
        file_path = os.path.join(conf["TrainDataPath"],f"train_data_{batch_idx}.parquet")
        processed_batch.write_parquet(file_path)
    print("Processing complete")
