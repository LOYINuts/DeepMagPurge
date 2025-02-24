from utils import config
import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from model import TaxonClassifier


def train(
    epochs: int,
    batch_size: int,
    net: torch.nn.Module,
    trainDataLoader: DataLoader,
    validDataLoader: DataLoader,
    device: str,
    lossF: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
    save_path: str,
):
    pass


def main():
    conf = config.AllConfig
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
    train(
        epochs=conf.epoch,
        batch_size=conf.batch_size,
    )


if __name__ == "__main__":
    main()
