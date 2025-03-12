import torch.optim.nadam
from utils import config
from tqdm import tqdm
import torch
import os
from torch.utils.data.dataloader import DataLoader
from model import TaxonClassifier, Dataset


def evaluate_cpu(
    net: torch.nn.Module,
    testDataLoader: DataLoader,
    lossF: torch.nn.modules.loss._WeightedLoss,
):
    torch.set_num_threads(config.AllConfig.num_workers)
    net.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        processBar = tqdm(testDataLoader, unit="step")
        for test_seq, test_labels in processBar:
            outputs = net(test_seq)
            predictions = torch.argmax(outputs, dim=1)
            # 累积结果用于最终统一计算
            all_preds.append(predictions)
            all_labels.append(test_labels)
        # 合并所有结果
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        # 统一计算指标
        total_acc = (all_preds == all_labels).float().mean()
        total_loss = lossF(
            torch.nn.functional.one_hot(all_preds).float(),  # 伪logits
            all_labels.float(),
        )
        print(f"Avg Test Loss: {total_loss.item():.4f}, Avg Test Acc: {total_acc.item():.4f}")


def main():
    conf = config.AllConfig
    conf.device = torch.device("cpu")

    model_path = os.path.join(conf.save_path, "checkpoint.pt")
    model = TaxonClassifier.TaxonModel(
        vocab_size=conf.vocab_size,
        embedding_size=conf.embedding_size,
        hidden_size=conf.hidden_size,
        device=conf.device,
        max_len=conf.max_len,
        num_layers=conf.num_layers,
        num_class=conf.num_class,
        drop_out=conf.drop_prob,
    )
    model = model.to(device=conf.device)
    lossF = torch.nn.CrossEntropyLoss()

    if os.path.exists(model_path) is True:
        print("Loading existing model state_dict......")
        checkpoint = torch.load(model_path, map_location=conf.device, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        print("No existing model state......")
        return

    print("Loading Dict Files......")
    all_dict = Dataset.Dictionary(conf.KmerFilePath, conf.TaxonFilePath)

    print("Loading dataset......")
    test_dataset = Dataset.SeqDataset(
        max_len=conf.max_len,
        input_path=conf.TestDataPath,
        all_dict=all_dict,
        k=conf.kmer,
        mode="eval",
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        num_workers=conf.num_workers,
        pin_memory=False,
        persistent_workers=True,  # 保持worker进程存活
    )
    print("Start evaluating......")
    evaluate_cpu(
        net=model,
        testDataLoader=test_dataloader,
        lossF=lossF,
    )


if __name__ == "__main__":
    main()
