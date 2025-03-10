import torch.optim.nadam
from utils import config
from tqdm import tqdm
import torch
import os
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
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    save_path: str,
):
    Best_loss = None
    for epoch in range(1, epochs + 1):
        processBar = tqdm(trainDataLoader, unit="step")
        net.train(True)
        total_train_loss = 0
        total_train_acc = 0
        for step, (train_seq, train_labels) in enumerate(processBar):
            train_seq = train_seq.to(device)
            train_labels = train_labels.to(device)
            optimizer.zero_grad()
            outputs = net(train_seq)
            loss = lossF(outputs, train_labels)
            total_train_loss += loss
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == train_labels) / train_labels.shape[0]
            total_train_acc += accuracy
            loss.backward()
            optimizer.step()
            scheduler.step()
            processBar.set_description(
                "[%d/%d] Loss: %.4f, Acc: %.4f"
                % (epoch, epochs, loss.item(), accuracy.item())
            )
            if step == len(processBar) - 1:
                correct, total_test_loss = 0, 0
                net.train(False)
                with torch.no_grad():
                    for test_seq, test_labels in testDataLoader:
                        test_seq = test_seq.to(device)
                        test_labels = test_labels.to(device)
                        test_out = net(test_seq)
                        tloss = lossF(test_out, test_labels)
                        predictions = torch.argmax(test_out, dim=1)
                        total_test_loss += tloss
                        correct += torch.sum(predictions == test_labels)
                test_acc = correct / (batch_size * len(testDataLoader))
                test_loss = total_test_loss / len(testDataLoader)
                train_avg_loss = total_train_loss / len(processBar)
                train_avg_acc = total_train_acc / len(processBar)
                processBar.set_description(
                    "[%d/%d] Avg Loss: %.4f, Avg Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f"
                    % (
                        epoch,
                        epochs,
                        train_avg_loss.item(),
                        train_avg_acc.item(),
                        test_loss.item(),
                        test_acc.item(),
                    )
                )
                if not Best_loss or test_loss < Best_loss:
                    model_save_path = os.path.join(save_path, "checkpoint.pt")
                    with open(model_save_path, "wb") as f:
                        torch.save(net.state_dict(), f)

        processBar.close()


def main():
    conf = config.AllConfig
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
    optimizer = torch.optim.NAdam(model.parameters(), lr=conf.lr)
    if os.path.exists(model_path) is True:
        print("Loading existing model state_dict......")
        checkpoint = torch.load(model_path, map_location=conf.device, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        print("No existing model state......")
    print("Loading Dict Files......")
    all_dict = Dataset.Dictionary(conf.KmerFilePath, conf.TaxonFilePath)
    print("Loading dataset......")
    all_dataset = Dataset.AllDataset(conf.DataPath, conf.max_len, all_dict, conf.kmer)
    train_dataloader = DataLoader(
        dataset=all_dataset.train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        dataset=all_dataset.test_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
    )
    lossF = torch.nn.CrossEntropyLoss()
    print("Setting lr scheduler")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_dataloader)
    )
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
        scheduler=scheduler,
        save_path=conf.save_path,
    )


if __name__ == "__main__":
    main()
