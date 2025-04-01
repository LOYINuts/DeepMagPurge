from utils import config
import os
from model import TaxonClassifier, Dataset
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import tqdm
import logging

TOTAL_LOSS = 0
TOTAL_ACC = 0
TOTAL_SENSITIVITY = 0


def benchmark(
    file_name: str,
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    logger: logging.Logger,
    num_sample: int,
    threshold: float = 0.5,
):
    lossF = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    total_acc = 0
    total_conf_acc = 0
    num_conf_samples = 0
    num_acc = 0
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for step, (seq, label) in enumerate(pbar):
            seq = seq.to(device)
            label = label.to(device)
            outputs = model(seq)
            predictions = torch.argmax(outputs, dim=1)
            acc = torch.sum(predictions == label) / label.shape[0]
            loss = lossF(outputs, label)
            total_loss += loss
            total_acc += acc
            # 计算置信度的准确率
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            conf_mask = max_probs >= threshold
            conf_acc = 0
            if conf_mask.sum() > 0:
                conf_preds = predictions[conf_mask]
                conf_labels = label[conf_mask]
                num_acc += torch.sum(conf_preds == conf_labels)
                conf_acc = torch.sum(conf_preds == conf_labels) / conf_labels.shape[0]
                total_conf_acc += conf_acc
                num_conf_samples += 1
        avg_loss = torch.as_tensor(total_loss / len(dataloader))
        avg_acc = torch.as_tensor(total_acc / len(dataloader))
        if num_conf_samples > 0:
            conf_avg_acc = total_conf_acc / num_conf_samples
        else:
            conf_avg_acc = 0
        conf_avg_acc = torch.as_tensor(conf_avg_acc)
        sensitivity = torch.as_tensor(num_acc / num_sample)
        logger.info(
            f"{file_name}:Avg Loss: {avg_loss.item():.4f}, Avg Acc: {avg_acc.item():.4f}, Avg Conf Acc: {conf_avg_acc.item():.4f} Sensitivity: {sensitivity.item():.4f}"
        )
        global TOTAL_LOSS, TOTAL_ACC, TOTAL_SENSITIVITY
        TOTAL_LOSS += avg_loss.item()
        TOTAL_ACC += avg_acc.item()
        TOTAL_SENSITIVITY += sensitivity.item()


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
        k=conf["kmer"], file_path=file_path, all_dict=all_dict, label=int(label)
    )
    bm_dataloader = DataLoader(bm_dataset, batch_size=64)
    device = torch.device(conf["eval_device"])
    logger.info(f"Start benchmark {file}")
    logger.info("-" * 80)
    benchmark(
        file_name=file,
        dataloader=bm_dataloader,
        model=model,
        device=device,
        num_sample=len(bm_dataset),
        logger=logger,
    )


def benchmark_main():
    # 设置logger
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler("logs/benchmark.log")
    handler.setFormatter(formatter)

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    conf = config.load_config("./data/config.toml")
    if conf is None:
        raise Exception("Load toml configuration error!")

    model_path = os.path.join(conf["filepath"]["save_path"], "checkpoint.pt")
    device = torch.device(conf["model"]["eval_device"])
    model = TaxonClassifier.TaxonModel(
        vocab_size=conf["model"]["vocab_size"],
        embedding_size=conf["model"]["embedding_size"],
        hidden_size=conf["model"]["hidden_size"],
        device=device,
        max_len=conf["model"]["max_len"],
        num_layers=conf["model"]["num_layers"],
        num_class=conf["model"]["num_class"],
        drop_out=conf["model"]["drop_prob"],
    )
    if os.path.exists(model_path) is True:
        logger.info("Loading existing model state_dict......")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        raise Exception("Can't run benchmark without existing model!")

    logger.info("Loading dict files......")
    all_dict = Dataset.Dictionary(conf["filepath"]["KmerFilePath"], conf["filepath"]["TaxonFilePath"])
    logger.info("Loading files2taxon.txt")
    file_path = conf["filepath"]["BenchmarkFile2Taxon"]
    file2taxon = {}
    num_files = 0
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            splits = line.split()
            file, taxon = splits[0], splits[1]
            label = all_dict.taxon2idx[taxon]
            file2taxon[file] = label
            num_files += 1
    for file, label in file2taxon.items():
        benchmark_one_file(
            file=file,
            label=label,
            conf=conf,
            model=model,
            all_dict=all_dict,
            logger=logger,
        )
    avg_loss = TOTAL_LOSS / num_files
    avg_acc = TOTAL_ACC / num_files
    avg_sen = TOTAL_SENSITIVITY / num_files
    logger.info("BenchMark Results")
    logger.info("-" * 80)
    logger.info(f"Avg Loss: {avg_loss:.4f}")
    logger.info(f"Avg Acc: {avg_acc:.4f}")
    logger.info(f"Avg Sentivity: {avg_sen:.4f}")
