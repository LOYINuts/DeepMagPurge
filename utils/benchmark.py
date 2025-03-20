from . import config
import os
from model import TaxonClassifier, Dataset
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import tqdm
import logging


def benchmark(
    dataloader: DataLoader,
    model: nn.Module,
    label: int,
    device: torch.device,
    logger: logging.Logger,
    threshold: float = 0.5,
):
    lossF = nn.CrossEntropyLoss()
    model.eval()
    the_label = torch.tensor([label]).unsqueeze(0)
    total_loss = 0
    total_acc= 0
    total_conf_acc =0
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for step, data in enumerate(pbar):
            data = data.to(device)
            outputs = model(data)
            loss = lossF(outputs, the_label)
            total_loss += loss


def benchmark_one_file(
    file: str,
    label: int,
    conf,
    model: nn.Module,
    all_dict: Dataset.Dictionary,
    logger: logging.Logger,
):
    root_path = conf["BenchmarkDataPath"]
    file_path = os.path.join(root_path, file)
    bm_dataset = Dataset.BenchmarkDataset(
        k=conf["kmer"], file_path=file_path, all_dict=all_dict
    )
    bm_dataloader = DataLoader(bm_dataset, batch_size=1)
    device = torch.device(conf["eval_device"])
    logger.info("Start benchmark......")
    logger.info("-" * 80)
    benchmark(
        dataloader=bm_dataloader, model=model, label=label, device=device, logger=logger
    )


if __name__ == "__main__":
    # 设置logger
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler("logs/benchmark.log")
    handler.setFormatter(formatter)

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    conf = config.load_config("./data/config.yaml")
    if conf is None:
        raise Exception("Load yaml configuration error!")

    model_path = os.path.join(conf["save_path"], "checkpoint.pt")
    device = torch.device(conf["eval_device"])
    model = TaxonClassifier.TaxonModel(
        vocab_size=conf["vocab_size"],
        embedding_size=conf["embedding_size"],
        hidden_size=conf["hidden_size"],
        device=device,
        max_len=conf["max_len"],
        num_layers=conf["num_layers"],
        num_class=conf["num_class"],
        drop_out=conf["drop_prob"],
    )
    if os.path.exists(model_path) is True:
        logger.info("Loading existing model state_dict......")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        raise Exception("Can't run benchmark without existing model!")

    logger.info("Loading dict files......")
    all_dict = Dataset.Dictionary(conf["KmerFilePath"], conf["TaxonFilePath"])
    logger.info("Loading files2taxon.txt")
    file_path = conf["BenchmarkFile2Taxon"]
    file2taxon = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            splits = line.split()
            file, taxon = splits[0], splits[1]
            label = all_dict.taxon2idx[taxon]
            file2taxon[file] = label
    for file, label in file2taxon:
        benchmark_one_file(
            file=file,
            label=label,
            conf=conf,
            model=model,
            all_dict=all_dict,
            logger=logger,
        )
