import random

_TRANSLATION_DICT = {
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

# 创建翻译表
_TRANSLATION_TABLE = str.maketrans(_TRANSLATION_DICT)


def TransferKmer2Idx(word_file_path: str):
    """词典生成,反向互补序列与原kmer共享同一个idx

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
            rev_word = ReverseComplementSeq(word)
            word2idx[rev_word] = idx
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


def Trim_seq(seq: str, min_trim=0, max_trim=75, ori_seq_len=150):
    """裁剪序列

    :param seq: 输入序列
    :param min_trim: 最小裁剪长度
    :param max_trim: 最大裁剪长度
    :param ori_seq_len: 原始序列长度
    :return: 裁剪后的序列
    """
    trim_length = random.randint(min_trim, max_trim)
    return seq[: ori_seq_len - trim_length]


def ReverseComplementSeq(seq: str):
    """生成序列的反向互补序列

    :param seq: 输入序列
    :return: 反向互补序列
    """
    return seq[::-1].translate(_TRANSLATION_TABLE)


def kmer2index(k_mer: str, word2idx: dict):
    """将 k-mer 转换为索引

    :param k_mer: k-mer 序列
    :param word2idx: 词到索引的映射字典
    :return: k-mer 对应的索引
    """
    return word2idx.get(k_mer, 1)


def seq2kmer(seq: str, k: int, word2idx: dict, max_len: int, trim: bool = False):
    """将序列转换为 k-mer 索引列表

    :param seq: 输入序列
    :param k: k-mer 的长度
    :param word2idx: 词到索引的映射字典
    :param max_len: 最大长度
    :param trim: 是否裁剪序列
    :return: k-mer 索引列表的张量
    """
    if trim:
        seq = Trim_seq(seq)

    length = len(seq)
    max_i = length - k + 1
    kmer_list = [kmer2index(seq[i : i + k], word2idx) for i in range(0, max_i)]
    # 不足长度进行padding
    if len(kmer_list) < max_len:
        kmer_list += [0] * (max_len - len(kmer_list))
    else:
        # 否则截断
        kmer_list = kmer_list[:max_len]
    return kmer_list
