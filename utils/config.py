import torch


class AllConfig:
    TaxonFilePath = "./data/taxon2label.txt"
    KmerFilePath = "./data/tokens_12mer.txt"
    TrainDataPath = "/home/lys/gh/DBFiles/dmpdata/all_concat_seq_data.fa"
    num_workers = 32
    TestDataPath = "/home/lys/gh/DBFiles/dmpdata/all_concat_seq_data.fa"
    save_path = "./checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 8390658
    embedding_size = 100
    hidden_size = 300
    num_layers = 2
    num_class = 120
    drop_prob = 0.1
    max_len = 150
    lr = 0.002
    kmer = 12
    batch_size = 2048
    epoch = 10
