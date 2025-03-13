import torch.optim.nadam
from utils import config
from tqdm import tqdm
import torch
import os
from torch.utils.data.dataloader import DataLoader
from model import TaxonClassifier, Dataset


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
                train_avg_loss = total_train_loss / len(processBar)
                train_avg_acc = total_train_acc / len(processBar)
                processBar.set_description(
                    "[%d/%d] Avg Loss: %.4f, Avg Acc: %.4f"
                    % (
                        epoch,
                        epochs,
                        train_avg_loss.item(),
                        train_avg_acc.item(),
                    )
                )
                if not Best_loss or train_avg_loss < Best_loss:
                    Best_loss = train_avg_loss
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
    lossF = torch.nn.CrossEntropyLoss()

    if os.path.exists(model_path) is True:
        print("Loading existing model state_dict......")
        checkpoint = torch.load(model_path, map_location=conf.device, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        print("No existing model state......")

    optimizer = torch.optim.NAdam(model.parameters(), lr=conf.lr)

    print("Loading Dict Files......")
    all_dict = Dataset.Dictionary(conf.KmerFilePath, conf.TaxonFilePath)

    print("Loading dataset......")
    train_dataset = Dataset.SeqDataset(
        max_len=conf.max_len,
        input_path=conf.TrainDataPath,
        all_dict=all_dict,
        k=conf.kmer,
        mode="train",
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=8,
    )

    print("Setting lr scheduler")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_dataloader)
    )

    print("Start Training")
    train(
        epochs=conf.epoch,
        net=model,
        trainDataLoader=train_dataloader,
        device=conf.device,
        lossF=lossF,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=conf.save_path,
    )


if __name__ == "__main__":
    main()
