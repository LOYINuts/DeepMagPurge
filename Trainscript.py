from utils import config
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from model import TaxonClassifier, Dataset


def train(
    epochs: int,
    batch_size: int,
    net: torch.nn.Module,
    trainDataLoader: DataLoader,
    testDataLoader: DataLoader,
    device: str,
    lossF: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
    save_path: str,
):
    best_test_loss = None
    history = {"Test Loss": [], "Test Accuracy": []}
    for epoch in range(1, epochs + 1):
        processBar = tqdm(trainDataLoader, unit="step")
        net.train(True)
        for step, (train_seq, train_labels) in enumerate(processBar):
            train_seq = train_seq.to(device)
            train_labels = train_labels.to(device)
            net.zero_grad()
            outputs = net(train_seq)
            loss = lossF(outputs, train_labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == train_labels) / train_labels.shape[0]
            loss.backward()
            optimizer.step()
            processBar.set_description(
                "[%d/%d] Loss: %.4f, Acc: %.4f"
                % (epoch, epochs, loss.item(), accuracy.item())
            )
            if step == len(processBar) - 1:
                correct, total_loss = 0, 0
                net.train(False)
                with torch.no_grad():
                    for test_seq, test_labels in testDataLoader:
                        test_seq = test_seq.to(device)
                        test_labels = test_labels.to(device)
                        test_out = net(test_seq)
                        tloss = lossF(test_out, test_labels)
                        predictions = torch.argmax(test_out, dim=1)
                        total_loss += tloss
                        correct += torch.sum(predictions == test_labels)
                test_acc = correct / (batch_size * len(testDataLoader))
                test_loss = total_loss / len(testDataLoader)
                history["Test Accuracy"].append(test_acc.item())
                history["Test Loss"].append(test_loss.item())
                processBar.set_description(
                    "[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f"
                    % (
                        epoch,
                        epochs,
                        loss.item(),
                        accuracy.item(),
                        test_loss.item(),
                        test_acc.item(),
                    )
                )
                if not best_test_loss or test_loss < best_test_loss:
                    model_save_path = os.path.join(save_path, "taxonclassifier.pth")
                    with open(model_save_path, "wb") as f:
                        torch.save(net.state_dict(), f)
                    best_test_loss = test_loss
        processBar.close()

    plt.plot(history["Test Loss"], label="Test Loss")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # 对验证集准确率进行可视化
    plt.plot(history["Test Accuracy"], color="red", label="Test Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def main():
    conf = config.AllConfig
    model_path = os.path.join(conf.save_path, "taxonclassifier.pth")
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
    if os.path.exists(model_path) is True:
        print("Loading existing model state......")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    else:
        print("No existing model state......")
    print("Loading Dict Files......")
    all_dict = Dataset.Dictionary(conf.KmerFilePath, conf.TaxonFilePath)
    print("Loading dataset......")
    all_dataset = Dataset.AllDataset(
        conf.DataPath, conf.max_len, all_dict, conf.samples, conf.kmer
    )
    train_dataloader = DataLoader(
        dataset=all_dataset.train_dataset, batch_size=conf.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=all_dataset.test_dataset, batch_size=conf.batch_size, shuffle=False
    )
    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    model = model.to(device=conf.device)
    print("Start Training")
    train(
        epochs=conf.epoch,
        batch_size=conf.batch_size,
        net=model,
        trainDataLoader=train_dataloader,
        testDataLoader=test_dataloader,
        device=conf.device,
        lossF=lossF,
        optimizer=optimizer,
        save_path=conf.save_path,
    )


if __name__ == "__main__":
    main()
