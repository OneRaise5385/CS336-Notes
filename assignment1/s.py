import os
import regex as re
from collections import defaultdict

def update_indices(
    indices: dict[tuple[int, ...], int], 
    pair: tuple[int, int], 
) -> defaultdict[tuple[int, ...], int]:
    '''
    更新 indices 序列的函数
    '''
    new_indices = defaultdict(int)
    for index in indices:
        new_index = []
        index_value = indices[index]
        i = 0
        while i < len(index):
            # i + 1 < len(index) 是用来保证 i 指向的是列表中的第二个 index
            # index[i] == pair[0] and index[i + 1] == pair[1] ：指定的 token 对 pair
            if i + 1 < len(index) and index[i] == pair[0] and index[i + 1] == pair[1]:
                new_index.append(pair[0] + pair[1])
                i += 2
            else:
                # 没有被指定 pair 对的时候，将原来index中的indice直接添加到new_index
                new_index.append(index[i])
                i += 1
        new_indices[tuple(new_index)] = index_value

    return new_indices


def pre_count_indices(
    content: str, 
    special_tokens: list
    ) -> defaultdict[tuple[int, ...], int]:
    '''
    对文档进行预分词
    '''
    # 按照特殊 tokens 分割文档
    texts = re.split("|".join(map(re.escape, special_tokens)), content)
    # 预分词规则：gpt2 的分词规则
    # 就这个正则化的错误让我改了一天的bug
    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+"""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # 对每个文档中的每个部分进行统计
    pre_indices = defaultdict(int)
    for text in texts:
        # 关于 re.finditer() 的使用在 `assignment.md` 中的 `其他` 部分有介绍
        pre_token_matches = re.finditer(PAT, text)
        # 统计分词出现的的次数
        for pre_token_matche in pre_token_matches:
            pre_indices_key = tuple([bytes([x]) for x in tuple(pre_token_matche.group().encode())])
            pre_indices[pre_indices_key] += 1
    return pre_indices


def max_pair(
    indices: defaultdict[tuple[int, ...], int]
    )-> tuple:
    '''
    找出出现次数最多的 pair
    '''
    counts = defaultdict(int)  # 用来计数的字典
    for index in indices:
        # 生成相邻两个 token 的组合（index1, index2）
        for index1, index2 in zip(index, index[1:]):
            counts[(index1, index2)] += indices[index]
    max_val = max(counts.values())  # 出现次数最多
    pair = max([k for k, v in counts.items() if v == max_val])  # 字典序最大
    
    return pair


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    给定输入语料的路径，训练一个 BPE 分词器，并输出其 vocab 和 merges 。
    参数：
        input_path (str | os.PathLike)：BPE 分词器训练数据的路径。
        vocab_size (int)：分词器词表的总大小（包括特殊 token）。
        special_tokens (list[str])：一个字符串列表，表示要加入词表的特殊 token。
            这些特殊 token 永远不会被拆分成多个 token，总是保持为一个整体。
            如果这些特殊 token 出现在 `input_path` 中，它们会被视作普通字符串处理。
    返回：
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]：
            vocab：
                训练得到的分词器词表，字典的 key 是 int 结构（词表中的 token ID），
                value 是 bytes（对应的 token 字节串）。
            merges：
                BPE 合并规则。列表中的每一项是一个 bytes 元组 (<token1>, <token2>)，
                表示 <token1> 和 <token2> 被合并为一个新 token。合并规则按创建顺序排列。
    """
    
    # 1. 初始化: 256个基础词、特殊 tokens
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    # 256个基础词，0-255，所以下一个是256
    next_token_id = 256
    # 将特殊 tokens 转换成 byte 格式并加入词表
    for special_token in special_tokens:
        # 注意这里所给的特殊 tokens 为 str 格式，vocab 中的是字节形式的
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    # 2. 预分词
    # 读取数据
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    indices = pre_count_indices(content, special_tokens)

    # 3. BPE 合并
    # 合并次数为词表大小减去初始化的词表
    num_merges = vocab_size - 256 - len(special_tokens)
    for i in range(num_merges):
        # 找出出现次数最多的 pair
        pair = max_pair(indices)
        # merges 更新
        merges.append(pair)
        # 词表更新。将两个字节相加：不是数值相加，是两个字节拼接到一起
        vocab[next_token_id] = pair[0] + pair[1]
        next_token_id += 1
        # indices 序列更新
        indices = update_indices(indices, pair)

    return vocab, merges