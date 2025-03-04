import torch


def TransferKmer2Idx(word_file_path: str):
    """词典生成

    Args:
        word_file_path (str): 文件路径，即所有kmer的文件路径

    Returns:
        dict: 词典
    """
    with open(word_file_path, "r") as fin:
        word2idx = {}
        idx = 1
        for line in fin:
            word = line.strip()
            word2idx[word] = idx
            idx += 1
    return word2idx


def TransferTaxon2Idx(taxon_file_path: str) -> dict:
    """物种词典

    Args:
        taxon_file_path (str): 物种文件

    Returns:
        dict: 物种分类词典
    """
    with open(taxon_file_path, "r") as fin:
        taxon2idx = {}
        for line in fin:
            line = line.strip()
            splits = line.split()
            taxon, idx = splits[0], splits[1]
            taxon2idx[taxon] = idx

    return taxon2idx


def ReverseComplementSeq(seq: str):
    """将序列转为反向互补链，在生物中反向互补链就是序列本身，A<->T,C<->G,所有的模糊的碱基都转为N

    Args:
        seq (str): kmer的序列
    """
    translation_dict = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N",
        "K": "N",
        "M": "N",
        "R": "N",
        "Y": "N",
        "S": "N",
        "W": "N",
        "B": "N",
        "V": "N",
        "H": "N",
        "D": "N",
        "X": "N",
    }
    letters = list(seq)
    letters = [translation_dict[base] for base in letters]
    return "".join(letters)[::-1]


def kmer2index(k_mer: str, word2idx: dict):
    """Converts k-mer to index to the embedding layer"""
    if k_mer in word2idx:
        idx = word2idx[k_mer]
    elif ReverseComplementSeq(k_mer) in word2idx:
        idx = word2idx[ReverseComplementSeq(k_mer)]
    else:
        idx = word2idx["<unk>"]
    return idx


def seq2kmer(seq: str, k: int, word2idx: dict, max_len: int):
    """将序列转换为kmer列表

    Args:
        seq (str): 序列
        k (int): kmer的k值
        word2idx (dict): kmer词典
    """
    length = len(seq)
    kmer_list = []
    for i in range(0, length):
        if i + k >= length + 1:
            break
        k_mer = seq[i : i + k]
        idx = kmer2index(k_mer, word2idx)
        kmer_list.append(idx)
    # 不足长度进行padding
    while len(kmer_list) < max_len:
        kmer_list.append(0)
    kmer_list = kmer_list[:max_len]
    kmer_array = torch.tensor(kmer_list)
    return kmer_array
