import os
import regex as re
from collections import defaultdict,Counter
from multiprocessing import Pool
from typing import BinaryIO
import cProfile
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_count_indices(
    content: str, 
    special_tokens: list
    ) -> defaultdict[tuple[int, ...], int]:
    '''
    对输入的文档进行预分词
    '''
    # 按照特殊 tokens 分割文档
    texts = re.split("|".join(map(re.escape, special_tokens)), content)
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

def multi_process_pre_token(input_path, num_processes, special_tokens):
    '''
    并行化预分词阶段
    '''
    # 首先将文本分段
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore").replace("\r\n", "\n")
            chunks.append(chunk)
            
    # 使用进程池并行处理
    with Pool() as pool:
        dicts = pool.starmap(pre_count_indices, [(chunk, special_tokens) for chunk in chunks])
    
    # 合并结果
    indices = Counter()
    for d in dicts:
        indices.update(d)
    # 转回普通字典
    indices = dict(indices)

    return indices

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

def max_pair(
    indices: defaultdict[tuple[int, ...], int]
    )-> tuple:
    '''
    找出出现次数最多的 pair
    '''
    counts = defaultdict(int)  # 用来计数的字典
    for index in indices:
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
    next_token_id = 256
    # 将特殊 tokens 转换成 byte 格式并加入词表
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    # 2. 预分词
    indices = multi_process_pre_token(input_path, 4, special_tokens)

    # 3. BPE 合并
    num_merges = vocab_size - 256 - len(special_tokens)
    for i in range(num_merges):
        # 找出出现次数最多的 pair
        pair = max_pair(indices)
        # merges 更新
        merges.append(pair)
        # 词表更新。
        vocab[next_token_id] = pair[0] + pair[1]
        next_token_id += 1
        # indices 序列更新
        indices = update_indices(indices, pair)

    return vocab, merges

if __name__ == '__main__':
    # run_train_bpe('../data/TinyStoriesV2-GPT4-valid.txt', 500, ['<|endoftext|>'])
    run_str = "run_train_bpe('../data/TinyStoriesV2-GPT4-valid.txt', 500, ['<|endoftext|>'])"
    cProfile.run(run_str, "valid-500-multi4.prof")