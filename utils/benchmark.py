from . import config
import os
from model import TaxonClassifier,Dataset
import torch
from torch.utils.data.dataloader import DataLoader


if __name__ == "__main__":
    conf = config.load_config("./data/config.yaml")
    if conf is None:
        raise Exception("Load yaml configuration error!")

    benchmark_data_path = conf["BenchmarkDataPath"]
    model_path = os.path.join(conf["save_path"], "checkpoint.pt")
    device = torch.device(conf["eval_device"])
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
    if os.path.exists(model_path) is True:
        print("Loading existing model state_dict......")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        raise Exception("Can't run benchmark without existing model!")
    
    print("Loading dict files......")
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