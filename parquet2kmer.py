import model.Dataset as Dataset
from utils import config, DataProcess
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm
import os

if __name__ == "__main__":
    conf = config.load_config("./data/config.yaml")
    if conf is None:
        raise Exception("Error loading configuration")
    if os.path.exists(conf["TrainDataPath"]) is True:
        print("文件存在！已移除")
        os.remove(conf["TrainDataPath"])
    all_dict = Dataset.Dictionary(conf["KmerFilePath"], conf["TaxonFilePath"])
    data = pl.read_parquet(conf["TempParquet"])
    print("parquet file loaded")
    pqwriter = None
    # 分批次处理数据
    batch_size = 100000  # 可根据实际情况调整批次大小
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

        # 转换为 Arrow 表格
        arrow_table = processed_batch.to_arrow()
        if batch_idx == 0:
            pqwriter = pq.ParquetWriter(conf["TrainDataPath"],schema=arrow_table.schema)
        # 逐块写入文件
        pqwriter.write_table(arrow_table)
    pqwriter.close()
    print("Processing complete")
