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

def modify_pair_to_indices(
    pair_to_indices: defaultdict[tuple[bytes], list[int]], 
    prev_pair, indices_value, i
    ) -> defaultdict[tuple[bytes], list[int]]:
    '''修改 pair_to_indices'''
    # 修改 pair 的个数
    if not pair_to_indices[prev_pair]:
        pair_to_indices[prev_pair].append(0)
    pair_to_indices[prev_pair][0] -= indices_value[i]
    # 修改 pair_to_indices 值中的索引
    if i in pair_to_indices[prev_pair][1:]:
        pair_to_indices[prev_pair].reverse()
        pair_to_indices[prev_pair].remove(i)
        pair_to_indices[prev_pair].reverse()
    return pair_to_indices

def add_pair_to_indices(
    pair_to_indices: defaultdict[tuple[bytes], list[int]], 
    new_pair, indices_value, i
    ) -> defaultdict[tuple[bytes], list[int]]:
    '''向 pair_to_indices 中添加新值'''
    if not pair_to_indices[new_pair]:
        pair_to_indices[new_pair] = [indices_value[i]]
    else:
        pair_to_indices[new_pair][0] += indices_value[i]
    pair_to_indices[new_pair].append(i)
    return pair_to_indices

def update_pair_to_indices(
    pair_to_indices: defaultdict[tuple[bytes], list[int]], 
    indices_value, index, pair, i, j, 
    loc: str = "right"
    )-> defaultdict[tuple[bytes], list[int]]:
    '''
    更新 pair_to_indices 的值，包括修改、添加 \n
    loc : {"left", "right"}
        用于指定要更新 pair 的位置：
        - "right": 当 pair 右侧存在相邻元素时。
        - "left":  当 pair 左侧存在相邻元素时。
    '''
    # pair 右侧有未匹配的元素时
    if loc == 'right':
        prev_pair = (pair[1], index[j + 2])
        new_pair = (pair[0] + pair[1], index[j + 2])
    # pair 左侧有未匹配的元素时
    elif loc == 'left':
        prev_pair = (index[j - 1], pair[0])
        new_pair = (index[j - 1], pair[0] + pair[1])
    # 修改
    pair_to_indices = modify_pair_to_indices(
        pair_to_indices, prev_pair, indices_value, i)
    # 添加
    pair_to_indices = add_pair_to_indices(
        pair_to_indices, new_pair, indices_value, i)
    return pair_to_indices

def generate_pair_to_indices(indices):
    # 构造 pair_to_indices
    indices_key = [list(k) for k in indices.keys()]
    indices_value = list(indices.values())
    pair_to_indices = defaultdict(list)
    for i in range(len(indices_key)):
        index = indices_key[i]
        for index1, index2 in zip(index, index[1:]):
            if pair_to_indices[(index1, index2)]:
                pair_to_indices[(index1, index2)][0] += indices_value[i]
            else:
                pair_to_indices[(index1, index2)].append(indices_value[i])
            pair_to_indices[(index1, index2)].append(i)
    return pair_to_indices, indices_key, indices_value

def merge(
    indices: defaultdict[tuple[bytes, ...], int],
    vocab_size, special_tokens, next_token_id,
    vocab, merges
    )-> tuple:
    '''合并的步骤，生成 pair 对'''
    # 构造 pair_to_indices
    pair_to_indices, indices_key, indices_value = generate_pair_to_indices(indices)
    # 开始合并
    num_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(num_merges):
        # 找到出现次数最多的 pair
        max_val = 0
        for i in pair_to_indices.values():
            if i[0] > max_val:
                max_val = i[0]
        pair = max([k for k, v in pair_to_indices.items() if v[0] == max_val])
        
        # 修改 pair_to_indices
        for i in pair_to_indices[pair][1:]:
            index = indices_key[i].copy()
            modify_position = []
            for j in range(len(index) - 1):
                if index[j] == pair[0] and index[j + 1] == pair[1]:
                    # index 中只存在 pair 而没有别的时
                    if len(index) == 2:
                        # pair_to_indices.pop(pair)
                        continue
                    # pair 在开头
                    elif j == 0:
                        pair_to_indices = update_pair_to_indices(
                            pair_to_indices, indices_value, index, pair, i, j, 'right')
                    # pair 在结尾 
                    elif j == len(index) - 2:
                        pair_to_indices = update_pair_to_indices(
                            pair_to_indices, indices_value, index, pair, i, j, 'left')
                    # pair 在中间
                    else:
                        pair_to_indices = update_pair_to_indices(
                            pair_to_indices, indices_value, index, pair, i, j, 'right')
                        pair_to_indices = update_pair_to_indices(
                            pair_to_indices, indices_value, index, pair, i, j, 'left')
                        
                    # 修改 indices_key
                    indices_key[i][j] = pair[0] + pair[1]
                    modify_position.append(j + 1)  # 记录下需要删除的元素的位置
            # 修改 indices_key
            for pos in sorted(modify_position, reverse=True):
                indices_key[i].pop(pos)
                
        # 删除 pair_to_indices 中已经合并的 pair
        pair_to_indices.pop(pair)
        
        # merges 更新
        merges.append(pair)
        
        # 词表更新
        vocab[next_token_id] = pair[0] + pair[1]
        next_token_id += 1
    
    return vocab, merges

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
    indices = multi_process_pre_token(input_path, 16, special_tokens)

    # 3. BPE 合并
    vocab, merges = merge(indices, vocab_size, special_tokens, 
                          next_token_id, vocab, merges)
    
    return vocab, merges

if __name__ == '__main__':
    run_str = "run_train_bpe('../data/TinyStoriesV2-GPT4-valid.txt', 500, ['<|endoftext|>'])"
    run_str = "run_train_bpe('../data/TinyStoriesV2-GPT4-train.txt', 500, ['<|endoftext|>'])"
    # run_str = "run_train_bpe('../data/text_example.txt', 270, ['<|endoftext|>'])"
    cProfile.run(run_str, "train-500-multi16-optimize.prof")