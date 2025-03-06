import torch


class AllConfig:
    TaxonFilePath = "./data/taxon2label.txt"
    KmerFilePath = "./data/tokens_8mers.txt"
    DataPath = "E:/GenomeData/DMPdata/testdata.fa"
    save_path = "./checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 32898
    embedding_size = 200
    hidden_size = 300
    num_layers = 2
    num_class = 120
    drop_prob = 0.5
    max_len = 150
    d_model = 350
    row = 30
    lr = 0.002
    kmer = 8
    batch_size = 512
    epoch = 1
