import torch


class AllConfig:
    SpeciesFilePath = "./data/taxonomy2label.txt"
    KmerPath = "./data/token_8mers.txt"
    save_path = "./checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 32897
    embedding_size = 32898
    hidden_size = 300
    num_layers = 2
    num_class = 2945
    drop_prob = 0.5
    max_len = 150
    lr = 0.001
    batch_size = 2048
    epoch = 4
