import torch


class AllConfig:
    TaxonFilePath = "./data/taxon2label.txt"
    KmerFilePath = "./data/tokens_8mers.txt"
    DataPath = "E:/StudyData/DMPdata/label_data"
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
    kmer = 8
    batch_size = 2048
    epoch = 4
    samples = 20
