import torch.optim.nadam
from utils import config
from tqdm import tqdm
import torch
import os
import argparse
from torch.utils.data.dataloader import DataLoader
from model import TaxonClassifier, Dataset


def train_setup(conf):
    model_path = os.path.join(conf["save_path"], "checkpoint.pt")
    train_device = torch.device(conf["device"])
    model = TaxonClassifier.TaxonModel(
        vocab_size=conf["vocab_size"],
        embedding_size=conf["embedding_size"],
        hidden_size=conf["hidden_size"],
        device=train_device,
        max_len=conf["max_len"],
        num_layers=conf["num_layers"],
        num_class=conf["num_class"],
        drop_out=conf["drop_prob"],
    )
    model = model.to(device=train_device)
    lossF = torch.nn.CrossEntropyLoss()
    if os.path.exists(model_path) is True:
        print("Loading existing model state_dict......")
        checkpoint = torch.load(
            model_path, map_location=train_device, weights_only=True
        )
        model.load_state_dict(checkpoint)
    else:
        print("No existing model state......")

    optimizer = torch.optim.NAdam(model.parameters(), lr=conf["lr"])

    print("Loading Dict Files......")
    all_dict = Dataset.Dictionary(conf["KmerFilePath"], conf["TaxonFilePath"])

    print("Loading dataset......")
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

    print("Setting lr scheduler")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_dataloader)
    )

    print("Start Training")
    train(
        epochs=conf["epoch"],
        net=model,
        trainDataLoader=train_dataloader,
        device=train_device,
        lossF=lossF,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=conf["save_path"],
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
):
    Best_loss = None
    for epoch in range(1, epochs + 1):
        processBar = tqdm(trainDataLoader, unit="step")
        net.train(True)
        total_train_loss = 0
        total_train_acc = 0
        total_conf_50_acc = 0  # 累计置信度为 50% 的准确率
        num_conf_50_samples = 0  # 记录置信度大于 50% 的样本数量
        for step, (train_seq, train_labels) in enumerate(processBar):
            train_seq = train_seq.to(device)
            train_labels = train_labels.to(device)
            optimizer.zero_grad()
            outputs = net(train_seq)
            loss = lossF(outputs, train_labels)
            loss.backward()
            optimizer.step()
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
            if step == len(processBar) - 1:
                # 显示置信度为50%的准确率
                if num_conf_50_samples > 0:
                    conf_50_avg_acc = total_conf_50_acc / num_conf_50_samples
                else:
                    conf_50_avg_acc = 0
                train_avg_loss = torch.as_tensor(total_train_loss / len(processBar))
                train_avg_acc = torch.as_tensor(total_train_acc / len(processBar))
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
                if not Best_loss or train_avg_loss < Best_loss:
                    Best_loss = train_avg_loss
                    model_save_path = os.path.join(save_path, "checkpoint.pt")
                    with open(model_save_path, "wb") as f:
                        torch.save(net.state_dict(), f)

        processBar.close()


def evaluate_setup(conf):
    torch.set_num_threads(conf["num_workers"])
    model_path = os.path.join(conf["save_path"], "checkpoint.pt")
    model = TaxonClassifier.TaxonModel(
        vocab_size=conf["vocab_size"],
        embedding_size=conf["embedding_size"],
        hidden_size=conf["hidden_size"],
        device=torch.device("cpu"),
        max_len=conf["max_len"],
        num_layers=conf["num_layers"],
        num_class=conf["num_class"],
        drop_out=conf["drop_prob"],
    )
    model = model.to(device=torch.device("cpu"))
    lossF = torch.nn.CrossEntropyLoss()

    if os.path.exists(model_path) is True:
        print("Loading existing model state_dict......")
        checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=True
        )
        model.load_state_dict(checkpoint)
    else:
        print("No existing model state......")
        return

    print("Loading Dict Files......")
    all_dict = Dataset.Dictionary(conf["KmerFilePath"], conf["TaxonFilePath"])

    print("Loading dataset......")
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
    print("Start evaluating......")
    evaluate(
        net=model,
        testDataLoader=test_dataloader,
        lossF=lossF,
    )


def evaluate(
    net: torch.nn.Module,
    testDataLoader: DataLoader,
    lossF: torch.nn.modules.loss._WeightedLoss,
):
    net.eval()
    total_loss = 0
    total_acc = 0
    total_conf_50_acc = 0  # 累计置信度为 50% 的准确率
    num_conf_50_samples = 0  # 记录置信度大于 50% 的样本数量
    with torch.no_grad():
        processBar = tqdm(testDataLoader, unit="step")
        for test_seq, test_labels in processBar:
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
        total_loss = torch.as_tensor(total_loss / len(testDataLoader))
        total_acc = torch.as_tensor(total_acc / len(testDataLoader))
        if num_conf_50_samples > 0:
            conf_50_avg_acc = total_conf_50_acc / num_conf_50_samples
        else:
            conf_50_avg_acc = 0
        conf_50_avg_acc = torch.as_tensor(conf_50_avg_acc)
        print(
            f"Avg Test Loss: {total_loss.item():.4f}, Avg Test Acc: {total_acc.item():.4f}, Avg Conf50 Acc: {conf_50_avg_acc.item()}"
        )


def predict():
    pass


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

    if run_mode == "train":
        train_setup(conf=conf)
    elif run_mode == "eval":
        evaluate_setup(conf=conf)
    elif run_mode == "predict":
        predict()
    else:
        raise Exception("非法的运行模式(mode)!,请在 train, eval, predict 中选择")


if __name__ == "__main__":
    main()
