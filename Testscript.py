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
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        processBar = tqdm(testDataLoader, unit="step")
        for test_seq, test_labels in processBar:
            outputs = net(test_seq)
            predictions = torch.argmax(outputs, dim=1)
            acc = torch.sum(predictions == test_labels) / test_labels.shape[0]
            loss = lossF(outputs,test_labels)
            total_loss += loss
            total_acc += acc
            processBar.set_description(
                "Loss: %.4f, Acc: %.4f"
                % ( loss.item(), acc.item())
            )
        total_loss = total_loss / len(testDataLoader)
        total_acc = total_acc / len(testDataLoader)
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
