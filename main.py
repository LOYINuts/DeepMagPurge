import torch.optim.nadam
from utils import config
from tqdm import tqdm
import torch
import os
import argparse
from torch.utils.data.dataloader import DataLoader
from model import TaxonClassifier, Dataset
import logging
from Bio import SeqIO
import numpy as np


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
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
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
    logger.info("Parsed YAML configuration:")
    for key, value in conf.items():
        logger.info(f"{key}: {value}")

    model_path = os.path.join(conf["save_path"], "checkpoint.pt")
    train_device = torch.device(conf["device"])
    model = setup_model(conf=conf, device=train_device)
    lossF = torch.nn.CrossEntropyLoss()
    load_model(model_path=model_path, model=model, device=train_device, logger=logger)
    optimizer = torch.optim.NAdam(model.parameters(), lr=conf["lr"])

    logger.info("Loading Dict Files......")
    all_dict = Dataset.Dictionary(conf["KmerFilePath"], conf["TaxonFilePath"])
    logger.info("Allocating memory......")
    num_elements = 16 * 1024 * 1024 * 1024 // 4
    huge_tensor = torch.empty(num_elements, dtype=torch.float32).cuda()

    logger.info("Loading dataset......")
    train_dataset = Dataset.SeqDataset(
        max_len=conf["max_len"],
        input_path=conf["TrainDataPath"],
        all_dict=all_dict,
        k=conf["kmer"],
        mode="train",
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=16,
    )

    logger.info("Setting lr scheduler")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_dataloader)
    )
    del huge_tensor
    torch.cuda.empty_cache()
    logger.info("Start Training")
    logger.info("-" * 80)
    train(
        epochs=conf["epoch"],
        net=model,
        trainDataLoader=train_dataloader,
        device=train_device,
        lossF=lossF,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=conf["save_path"],
        logger=logger,
    )


def train(
    epochs: int,
    net: torch.nn.Module,
    trainDataLoader: DataLoader,
    device,
    lossF: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    save_path: str,
    logger: logging.Logger,
):
    """训练函数。

    :param epochs: 训练的轮数
    :param net: 要训练的模型实例
    :param trainDataLoader: 训练数据加载器，用于批量加载训练数据
    :param device: 计算设备，如CPU或GPU
    :param lossF: 损失函数，用于计算模型预测结果与真实标签之间的损失
    :param optimizer: 优化器，用于更新模型的参数
    :param scheduler: 学习率调度器，用于动态调整学习率
    :param save_path: 模型保存的路径
    :param logger: 日志记录器，用于记录训练过程中的信息
    """
    Best_loss = None
    scaler = torch.amp.GradScaler(device=device)  # type: ignore
    for epoch in range(1, epochs + 1):
        processBar = tqdm(trainDataLoader, unit="step")
        net.train(True)
        total_train_loss = 0
        total_train_acc = 0
        total_conf_50_acc = 0  # 累计置信度为 50% 的准确率
        num_conf_50_samples = 0  # 记录置信度大于 50% 的样本数量
        log_interval = 50
        msg = "-" * 30 + f"Epoch:{epoch}" + "-" * 30
        logger.info(msg)
        data_length = len(processBar)
        for step, (train_seq, train_labels) in enumerate(processBar):
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
            conf_50_mask = max_probs >= 0.5
            conf_50_acc = 0
            if conf_50_mask.sum() > 0:
                conf_50_preds = predictions[conf_50_mask]
                conf_50_labels = train_labels[conf_50_mask]
                conf_50_acc = (
                    torch.sum(conf_50_preds == conf_50_labels) / conf_50_labels.shape[0]
                )
                total_conf_50_acc += conf_50_acc
                num_conf_50_samples += 1

            conf_50_acc = torch.as_tensor(conf_50_acc)
            processBar.set_description(
                "[%d/%d] Loss: %.4f, Acc: %.4f Conf50 Acc: %.4f"
                % (epoch, epochs, loss.item(), accuracy.item(), conf_50_acc.item())
            )
            # 每隔 log_interval 个 step 记录一次日志
            if step % log_interval == 0:
                logger.info(
                    "Epoch:[%d/%d] Step:[%d/%d] Loss: %.4f, Acc: %.4f, Conf50 Acc: %.4f"
                    % (
                        epoch,
                        epochs,
                        step,
                        data_length,
                        loss.item(),
                        accuracy.item(),
                        conf_50_acc.item(),
                    )
                )
            if step % (data_length // 10) == 0:
                model_save_path = os.path.join(save_path, "checkpoint.pt")
                with open(model_save_path, "wb") as f:
                    torch.save(net.state_dict(), f)
            if step == data_length - 1:
                # 显示置信度为50%的准确率
                if num_conf_50_samples > 0:
                    conf_50_avg_acc = total_conf_50_acc / num_conf_50_samples
                else:
                    conf_50_avg_acc = 0
                train_avg_loss = torch.as_tensor(total_train_loss / data_length)
                train_avg_acc = torch.as_tensor(total_train_acc / data_length)
                conf_50_avg_acc = torch.as_tensor(conf_50_avg_acc)
                processBar.set_description(
                    "[%d/%d] Avg Loss: %.4f, Avg Acc: %.4f, Avg Conf50 Acc: %.4f"
                    % (
                        epoch,
                        epochs,
                        train_avg_loss.item(),
                        train_avg_acc.item(),
                        conf_50_avg_acc.item(),
                    )
                )
                logger.info(
                    "[%d/%d] Avg Loss: %.4f, Avg Acc: %.4f, Avg Conf50 Acc: %.4f"
                    % (
                        epoch,
                        epochs,
                        train_avg_loss.item(),
                        train_avg_acc.item(),
                        conf_50_avg_acc.item(),
                    )
                )
                if not Best_loss or train_avg_loss < Best_loss:
                    Best_loss = train_avg_loss
                    model_save_path = os.path.join(save_path, "checkpoint.pt")
                    with open(model_save_path, "wb") as f:
                        torch.save(net.state_dict(), f)

        processBar.close()
    # 结束添加分割线
    logger.info("Training Finished")
    logger.info("-" * 80)


def evaluate_setup(conf, logger: logging.Logger):
    """评估前的准备工作，如加载测试数据等。

    :param conf: 配置信息对象
    :param logger: 日志记录器，用于记录准备过程中的信息
    """
    logger.info("Parsed YAML configuration:")
    for key, value in conf.items():
        logger.info(f"{key}: {value}")

    eval_device = torch.device("cpu")
    torch.set_num_threads(conf["num_workers"])
    model_path = os.path.join(conf["save_path"], "checkpoint.pt")
    model = setup_model(conf, eval_device)
    lossF = torch.nn.CrossEntropyLoss()
    loaded = load_model(
        model_path=model_path, model=model, device=eval_device, logger=logger
    )
    if not loaded:
        logger.info("You can't run eval mode without an existing model!")
        return

    logger.info("Loading Dict Files......")
    all_dict = Dataset.Dictionary(conf["KmerFilePath"], conf["TaxonFilePath"])

    logger.info("Loading dataset......")
    test_dataset = Dataset.SeqDataset(
        max_len=conf["max_len"],
        input_path=conf["TestDataPath"],
        all_dict=all_dict,
        k=conf["kmer"],
        mode="eval",
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=16,
        pin_memory=False,
        persistent_workers=True,  # 保持worker进程存活
    )
    logger.info("Start evaluating......")
    logger.info("-" * 80)
    evaluate(net=model, testDataLoader=test_dataloader, lossF=lossF, logger=logger)


def evaluate(
    net: torch.nn.Module,
    testDataLoader: DataLoader,
    lossF: torch.nn.modules.loss._WeightedLoss,
    logger: logging.Logger,
):
    """评估模型的函数。

    :param net: 要评估的模型实例
    :param testDataLoader: 测试数据加载器，用于批量加载测试数据
    :param lossF: 损失函数，用于计算模型预测结果与真实标签之间的损失
    :param logger: 日志记录器，用于记录评估过程中的信息
    """
    net.eval()
    total_loss = 0
    total_acc = 0
    total_conf_50_acc = 0  # 累计置信度为 50% 的准确率
    num_conf_50_samples = 0  # 记录置信度大于 50% 的样本数量
    log_interval = 50
    with torch.no_grad():
        processBar = tqdm(testDataLoader, unit="step")
        for step, (test_seq, test_labels) in enumerate(processBar):
            outputs = net(test_seq)
            predictions = torch.argmax(outputs, dim=1)
            acc = torch.sum(predictions == test_labels) / test_labels.shape[0]
            loss = lossF(outputs, test_labels)
            total_loss += loss
            total_acc += acc
            # 计算置信度为 50% 的准确率
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            conf_50_mask = max_probs >= 0.5
            conf_50_acc = 0
            if conf_50_mask.sum() > 0:
                conf_50_preds = predictions[conf_50_mask]
                conf_50_labels = test_labels[conf_50_mask]
                conf_50_acc = (
                    torch.sum(conf_50_preds == conf_50_labels) / conf_50_labels.shape[0]
                )
                total_conf_50_acc += conf_50_acc
                num_conf_50_samples += 1
            conf_50_acc = torch.as_tensor(conf_50_acc)
            processBar.set_description(
                "Loss: %.4f, Acc: %.4f, Conf50 Acc: %.4f"
                % (loss.item(), acc.item(), conf_50_acc.item())
            )
            # 每隔 log_interval 个 step 记录一次日志
            if step % log_interval == 0:
                logger.info(
                    "[%d/%d] Loss: %.4f, Acc: %.4f, Conf50 Acc: %.4f"
                    % (
                        step,
                        len(processBar),
                        loss.item(),
                        acc.item(),
                        conf_50_acc.item(),
                    )
                )
        total_loss = torch.as_tensor(total_loss / len(testDataLoader))
        total_acc = torch.as_tensor(total_acc / len(testDataLoader))
        if num_conf_50_samples > 0:
            conf_50_avg_acc = total_conf_50_acc / num_conf_50_samples
        else:
            conf_50_avg_acc = 0
        conf_50_avg_acc = torch.as_tensor(conf_50_avg_acc)
        logger.info(
            f"Avg Test Loss: {total_loss.item():.4f}, Avg Test Acc: {total_acc.item():.4f}, Avg Conf50 Acc: {conf_50_avg_acc.item():.4f}"
        )
        logger.info("End evaluating")
        logger.info("-" * 80)


def predict_one_record(
    model: torch.nn.Module,
    seq_data: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    weighted_probs = 0
    total_weight = 0
    with torch.no_grad():
        for seq in seq_data:
            seq = seq.to(device)
            logits = model(seq)
            probs = torch.softmax(logits, dim=1)
            # 动态计算权重（基于每个样本的预测置信度）
            sample_weights, _ = torch.max(probs, dim=1)  # 获取每个样本的最大概率
            weighted_probs += torch.sum(probs * sample_weights.unsqueeze(1), dim=0)
            total_weight += torch.sum(sample_weights)

    avg_probs = torch.as_tensor(weighted_probs / total_weight).cpu().numpy()

    # 置信度决策逻辑
    max_prob_idx = np.argmax(avg_probs)
    max_prob = avg_probs[max_prob_idx]

    if max_prob >= threshold:
        return {
            "prediction": int(max_prob_idx),
            "confidence": float(max_prob),
            "status": "high_confidence",
        }
    else:
        top3_indices = np.argsort(avg_probs)[-3:][::-1]
        top3_probs = avg_probs[top3_indices]
        return {
            "top3": [
                {"class": int(idx), "prob": float(prob)}
                for idx, prob in zip(top3_indices, top3_probs)
            ],
            "confidence": float(max_prob),
            "status": "low_confidence",
        }


def predict_one_file(
    model: torch.nn.Module,
    file,
    conf,
    all_dict: Dataset.Dictionary,
    device: torch.device,
):
    with open(file, "r") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            rec_dataset = Dataset.RecordSeqDataset(
                k=conf["kmer"],
                all_dict=all_dict,
                record=rec,
            )
            rec_dataloader = DataLoader(rec_dataset, batch_size=128, shuffle=False)
            prediction = predict_one_record(
                model=model, seq_data=rec_dataloader, device=device
            )


def predict_files(conf, logger: logging.Logger):
    model_path = os.path.join(conf["save_path"], "checkpoint.pt")
    device = torch.device(conf["device"])
    model = setup_model(conf, device)
    all_dict = Dataset.Dictionary(conf["KmerFilePath"], conf["TaxonFilePath"])
    load_model(model_path=model_path, model=model, device=device, logger=logger)


def main():
    parser = argparse.ArgumentParser(description="DeepMAGPurge")
    parser.add_argument(
        "--config",
        type=str,
        default="./data/config.yaml",
        help="path to config yaml file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="mode to run DeepMAGPurge,choose train, eval or predict",
    )
    args = parser.parse_args()

    conf = config.load_config(args.config)
    run_mode = args.mode

    os.makedirs("./logs", exist_ok=True)
    if run_mode == "train":
        train_logger = setup_logger("train_logger", "logs/train.log")
        train_setup(conf=conf, logger=train_logger)
    elif run_mode == "eval":
        eval_logger = setup_logger("eval_logger", "logs/eval.log")
        evaluate_setup(conf=conf, logger=eval_logger)
    elif run_mode == "predict":
        predict_logger = setup_logger("predict_logger", "logs/predict.log")
        predict_files(conf=conf, logger=predict_logger)
    else:
        logging.error("非法的运行模式(mode)!,请在 train, eval, predict 中选择")
        raise Exception("非法的运行模式(mode)!,请在 train, eval, predict 中选择")


if __name__ == "__main__":
    main()
