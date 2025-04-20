from utils import config, benchmark
import torch
import os
import argparse
from torch.utils.data.dataloader import DataLoader
from model import TaxonClassifier, Dataset
import logging
from Bio import SeqIO
import numpy as np
import random
import polars as pl
from collections import defaultdict
from tqdm import tqdm


class PredOutput:
    def __init__(
        self,
        taxon: int = -1,
        confidence: bool = False,
        threshold: float = 0.5,
        top3taxon: list = [],
    ):
        self.Taxon = taxon
        self.Confidence = confidence
        self.Threshold = threshold
        self.Top3Taxon = top3taxon


def setup_model(conf, device):
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
    model = model.to(device=device)
    return model


def load_model(
    model_path: str,
    model: torch.nn.Module,
    device: torch.device,
    logger: logging.Logger,
):
    """加载预训练模型的函数。

    :param model_path: 模型文件的路径
    :param model: 要加载权重的模型实例
    :param device: 计算设备，如CPU或GPU
    :param logger: 日志记录器，用于记录加载过程中的信息
    """
    if os.path.exists(model_path) is True:
        logger.info("Loading existing model state_dict......")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        return True
    else:
        logger.info("No existing model state......")
        return False


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """设置日志记录器的函数。

    :param name: 日志记录器的名称
    :param log_file: 日志文件的路径
    :param level: 日志记录的级别，默认为INFO
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M"
    )
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def train_setup(conf, logger: logging.Logger):
    """训练前的准备工作，数据加载、模型初始化等。

    :param conf: 配置信息对象
    :param logger: 日志记录器，用于记录准备过程中的信息
    """
    logger.info("#" * 80)
    logger.info("#" * 80)
    model_path = os.path.join(conf["filepath"]["save_path"], "checkpoint.pt")
    train_device = torch.device(conf["model"]["train_device"])
    model = setup_model(conf=conf["model"], device=train_device)
    # 打乱文件顺序
    files = os.listdir(conf["filepath"]["TrainDataPath"])
    random.shuffle(files)

    lossF = torch.nn.CrossEntropyLoss()
    load_model(model_path=model_path, model=model, device=train_device, logger=logger)
    optimizer = torch.optim.NAdam(model.parameters(), lr=conf["model"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(files) * 100)

    logger.info("Start Training")
    logger.info("#" * 80)

    for epoch in range(conf["model"]["epoch"]):
        logger.info(msg="-" * 30 + f"EPOCH {epoch}" + "-" * 30)
        idx = 0
        for i, file in enumerate(files):
            full_path = os.path.join(conf["filepath"]["TrainDataPath"], file)
            train_dataset = Dataset.PQSeqDataset(
                input_path=full_path,
            )
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=conf["model"]["batch_size"],
                shuffle=True,
                num_workers=16,
            )
            logger.info(f"Using {file} to train...")
            train(
                epochs=1,
                net=model,
                trainDataLoader=train_dataloader,
                device=train_device,
                lossF=lossF,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
            )
            if i % 10 == 0:
                logger.info(f"Saving model as checkpoint_{idx}.pt")
                model_save_path = os.path.join(
                    conf["filepath"]["save_path"], f"checkpoint_{idx}.pt"
                )
                with open(model_save_path, "wb") as f:
                    torch.save(model.state_dict(), f)
                idx = (idx + 1) % 2
            logger.info("-" * 80)
    logger.info("End Training")
    logger.info("#" * 80)


def train(
    epochs: int,
    net: torch.nn.Module,
    trainDataLoader: DataLoader,
    device,
    lossF: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    logger: logging.Logger,
    threshold: float = 0.5,
):
    """训练函数。

    :param epochs: 训练的轮数
    :param net: 要训练的模型实例
    :param trainDataLoader: 训练数据加载器，用于批量加载训练数据
    :param device: 计算设备，如CPU或GPU
    :param lossF: 损失函数，用于计算模型预测结果与真实标签之间的损失
    :param optimizer: 优化器，用于更新模型的参数
    :param scheduler: 学习率调度器，用于动态调整学习率
    :param logger: 日志记录器，用于记录训练过程中的信息
    :param threshold: 置信度，默认为0.5
    """
    scaler = torch.amp.GradScaler(device=device)  # type: ignore
    for epoch in range(1, epochs + 1):
        net.train(True)
        total_train_loss = 0
        total_train_acc = 0
        total_conf_acc = 0  # 累计置信度为 50% 的准确率
        num_conf_samples = 0  # 记录置信度大于 50% 的样本数量
        log_interval = 50
        data_length = len(trainDataLoader)
        for step, (train_seq, train_labels) in enumerate(trainDataLoader):
            train_seq = train_seq.to(device)
            train_labels = train_labels.to(device)
            optimizer.zero_grad()

            # 自动混合精度上下文
            with torch.amp.autocast("cuda"):  # type: ignore
                outputs = net(train_seq)
                loss = lossF(outputs, train_labels)

            scaler.scale(loss).backward()  # 缩放损失反向传播
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放器
            scheduler.step()

            total_train_loss += loss
            # 普通准确率计算(置信度为0)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == train_labels) / train_labels.shape[0]
            total_train_acc += accuracy
            # 计算置信度为 50% 的准确率
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            conf_mask = max_probs >= threshold
            conf_acc = 0
            if conf_mask.sum() > 0:
                conf_preds = predictions[conf_mask]
                conf_labels = train_labels[conf_mask]
                conf_acc = torch.sum(conf_preds == conf_labels) / conf_labels.shape[0]
                total_conf_acc += conf_acc
                num_conf_samples += 1

            conf_acc = torch.as_tensor(conf_acc)
            # 每隔 log_interval 个 step 记录一次日志
            if step % log_interval == 0:
                logger.info(
                    "Epoch:[%d/%d] Step:[%d/%d] Loss: %.4f, Acc: %.4f, Conf Acc: %.4f"
                    % (
                        epoch,
                        epochs,
                        step,
                        data_length,
                        loss.item(),
                        accuracy.item(),
                        conf_acc.item(),
                    )
                )
            if step == data_length - 1:
                # 显示置信度为50%的准确率
                if num_conf_samples > 0:
                    conf_avg_acc = total_conf_acc / num_conf_samples
                else:
                    conf_avg_acc = 0
                train_avg_loss = torch.as_tensor(total_train_loss / data_length)
                train_avg_acc = torch.as_tensor(total_train_acc / data_length)
                conf_avg_acc = torch.as_tensor(conf_avg_acc)
                logger.info(
                    "[%d/%d] Avg Loss: %.4f, Avg Acc: %.4f, Avg Conf Acc: %.4f"
                    % (
                        epoch,
                        epochs,
                        train_avg_loss.item(),
                        train_avg_acc.item(),
                        conf_avg_acc.item(),
                    )
                )


def predict_one_record(
    model: torch.nn.Module,
    seq_dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> PredOutput:
    model.eval()
    total_probs = 0
    num_samples = 0
    with torch.no_grad():
        for seq in seq_dataloader:
            seq = seq.to(device)
            logits = model(seq)
            probs = torch.softmax(logits, dim=1)
            total_probs += torch.sum(probs, dim=0)  # 直接累加概率
            num_samples += probs.size(0)  # 累加样本数量

    avg_probs = torch.as_tensor(total_probs / num_samples).cpu().numpy()

    # 置信度决策逻辑
    max_prob_idx = np.argmax(avg_probs)
    max_prob = avg_probs[max_prob_idx]
    # 取预测可能性最高的3个结果
    top3_indices = np.argsort(avg_probs)[-3:][::-1]
    top3_probs = avg_probs[top3_indices]
    return PredOutput(
        taxon=int(max_prob_idx),
        confidence=(max_prob >= threshold),
        threshold=threshold,
        top3taxon=[
            {"taxon": int(idx), "prob": float(prob)}
            for idx, prob in zip(top3_indices, top3_probs)
        ],
    )


def predict_one_file(
    model: torch.nn.Module,
    file: str,
    conf,
    all_dict: Dataset.Dictionary,
    device: torch.device,
    logger: logging.Logger,
):
    outputs = []
    # 使用defaultdict按Taxon分组存储SeqRecord对象和预测信息
    taxon_info = defaultdict(lambda: {"records": [], "count": 0})
    file_path = os.path.join(conf["predict"]["PredictFilePath"], file)
    csv_output_path = os.path.join(
        conf["predict"]["OutputPath"],
        "predict",
        f"{os.path.splitext(file)[0]}.csv",
    )
    os.makedirs(os.path.join(conf["predict"]["OutputPath"], "predict"), exist_ok=True)
    fillter_output_path = os.path.join(conf["predict"]["OutputPath"], "filtered")
    os.makedirs(fillter_output_path, exist_ok=True)
    # counts = 0
    with open(file_path, "r") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            # counts += 1
            try:
                rec_dataset = Dataset.PredictSeqDataset(
                    k=conf["model"]["kmer"],
                    all_dict=all_dict,
                    record=rec,
                    sub_seq_len=conf["model"]["max_len"],
                )
            except Exception as e:
                logger.info(e)
                continue
            rec_dataloader = DataLoader(rec_dataset, batch_size=64, shuffle=False)
            pred = predict_one_record(
                model=model,
                seq_dataloader=rec_dataloader,
                device=device,
                threshold=conf["predict"]["threshold"],
            )
            top3 = pred.Top3Taxon
            taxonomy = all_dict.idx2taxon[str(pred.Taxon)]
            row = {
                "contig": rec.id,
                "Taxon_id": pred.Taxon,
                "Taxonomy": taxonomy,
                "Confidence": pred.Confidence,
                "Threshold": pred.Threshold,
                "Top1_Taxon": top3[0]["taxon"],
                "Top1_Prob": top3[0]["prob"],
                "Top2_Taxon": top3[1]["taxon"],
                "Top2_Prob": top3[1]["prob"],
                "Top3_Taxon": top3[2]["taxon"],
                "Top3_Prob": top3[2]["prob"],
            }
            taxon_info[pred.Taxon]["records"].append((rec, pred))
            taxon_info[pred.Taxon]["count"] += 1
            outputs.append(row)

    df = pl.DataFrame(outputs)
    df.write_csv(csv_output_path)
    # 1. 确定出现次数最多的taxon
    if taxon_info:
        main_taxon = max(taxon_info.keys(), key=lambda x: taxon_info[x]["count"])
    else:
        main_taxon = None

    # 2. 筛选记录
    filtered_records = []
    for taxon, info in taxon_info.items():
        for rec, pred in info["records"]:
            # 保留条件：属于主要taxon 或 不属于但confidence为False
            if taxon == main_taxon or not pred.Confidence:
                filtered_records.append(rec)

    # 写入筛选后的fasta文件
    if main_taxon is not None:
        output_path = os.path.join(
            fillter_output_path, f"{os.path.splitext(file)[0]}_filtered.fa"
        )
        with open(output_path, "w") as output_handle:
            SeqIO.write(filtered_records, output_handle, "fasta")


def predict_files(conf, logger: logging.Logger):
    model_path = os.path.join(conf["filepath"]["save_path"], "checkpoint.pt")
    device = torch.device(conf["model"]["eval_device"])
    logger.info("setup model......")
    model = setup_model(conf["model"], device)
    logger.info("loading dict files......")
    all_dict = Dataset.Dictionary(
        conf["filepath"]["KmerFilePath"], conf["filepath"]["TaxonFilePath"]
    )
    ok = load_model(model_path=model_path, model=model, device=device, logger=logger)
    if ok is False:
        raise Exception("Can't run predict due to there is no existing model!")
    files = [
        f
        for f in os.listdir(conf["predict"]["PredictFilePath"])
        if os.path.isfile(os.path.join(conf["predict"]["PredictFilePath"], f))
    ]
    pbar = tqdm(files, desc="Processing")
    for file in pbar:
        predict_one_file(
            model=model,
            file=file,
            conf=conf,
            all_dict=all_dict,
            device=device,
            logger=logger,
        )
        logger.info(f"predict {file} complete.")


def main():
    parser = argparse.ArgumentParser(description="DeepMAGPurge")
    parser.add_argument(
        "--config",
        type=str,
        default="./data/config.toml",
        help="path to config toml file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="mode to run DeepMAGPurge,choose train, predict and benchmark",
    )
    args = parser.parse_args()

    conf = config.load_config(args.config)
    run_mode = args.mode

    os.makedirs("./logs", exist_ok=True)
    if run_mode == "train":
        train_logger = setup_logger("train_logger", "logs/train.log")
        train_setup(conf=conf, logger=train_logger)
    elif run_mode == "predict":
        predict_logger = setup_logger("predict_logger", "logs/predict.log")
        predict_files(conf=conf, logger=predict_logger)
    elif run_mode == "benchmark":
        benchmark.benchmark_main()
    else:
        logging.error("非法的运行模式(mode)!,请在 train, eval, predict 中选择")
        raise Exception("非法的运行模式(mode)!,请在 train, eval, predict 中选择")


if __name__ == "__main__":
    main()
