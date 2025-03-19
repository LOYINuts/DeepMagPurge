from . import config
import os
from model import TaxonClassifier, Dataset
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn

# TODO
def benchmark():
    pass

# TODO
def benchmark_one_file(file:str,label:int,conf,model:nn.Module,all_dict:Dataset.Dictionary):
    root_path = conf["BenchmarkDataPath"]
    file_path = os.path.join(root_path,file)
    pass
         

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
    print("Loading files2taxon.txt")
    file_path = conf["BenchmarkFile2Taxon"]
    file2taxon = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            splits = line.split()
            file, taxon = splits[0], splits[1]
            label = all_dict.taxon2idx[taxon]
            file2taxon[file] = label
    for key,value in file2taxon:
        pass
    
