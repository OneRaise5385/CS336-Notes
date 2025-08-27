# Assignment 1: Building a Transformer LM
# 1 作业概述
从零开始构建训练一个标准 Transformer 语言模型（LM）所需的所有组件，并训练一些模型。  

## 1.1 你将实现
1. BPE 分词器
2. Transformer
3. 交叉熵损失函数和 AdamW 优化器
4. 支持模型与优化器状态序列化与加载的训练循环

## 1.2 你将运行
1. 在 TinyStories 数据集上训练一个 BPE 分词器。  
2. 用训练好的分词器将数据集转换为整数 ID 序列。  
3. 在 TinyStories 数据集上训练一个 Transformer 语言模型。  
4. 使用训练好的 Transformer LM 生成样本并评估困惑度（perplexity）。  
5. 在 OpenWebText 数据集上训练模型，并将你得到的困惑度提交到排行榜。  

## 1.3 工具
我们希望你从零实现这些组件。特别是，你不能使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义，除了以下内容：  
- `torch.nn.Parameter`  
- `torch.nn` 中的容器类（如 `Module`、`ModuleList`、`Sequential` 等）  
- `torch.optim.Optimizer` 基类  

你可以使用 PyTorch 的其他定义。如果不确定某个函数或类是否允许使用，可以在 Slack 上询问。遇到不确定时，请考虑使用它是否会破坏“从零开始”的作业理念。  

允许使用大型语言模型（如 ChatGPT）来回答低层次的编程问题或关于语言模型的高层次概念问题，但禁止直接用它来解决作业中的问题。  我们强烈建议你在完成作业时**禁用** IDE 中的 AI 自动补全（如 Cursor Tab、GitHub Copilot），但允许使用非 AI 自动补全（例如函数名自动补全）。我们发现 AI 自动补全会使你更难深入理解作业内容。  

## 1.4 代码与提交
所有作业代码和作业说明都在 GitHub 仓库中：  
[github.com/stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)  

请 `git clone` 该仓库，如果有更新，我们会通知你 `git pull` 获取最新版本。  

1. `cs336_basics/*`：你将编写代码的地方。注意，这里没有预先写好的代码，你可以完全从零开始。  
2. `adapters.py`：你的代码必须提供一组功能。对于每个功能（如缩放点积注意力），在 `adapters.py` 中的实现函数（如 `run_scaled_dot_product_attention`）里调用你写的代码即可。**注意**：你对 `adapters.py` 的修改不应包含实质性逻辑，它只是胶水代码。  
3. `test_*.py`：包含你必须通过的测试（如 `test_scaled_dot_product_attention`），这些测试会调用 `adapters.py` 中的钩子。**不要修改测试文件**。  

你需要向 Gradescope 提交以下文件：  
- `writeup.pdf`：回答所有书面问题，请用排版工具（如 LaTeX）编写。  
- `code.zip`：包含你编写的所有代码。  

要提交到排行榜，请向以下仓库提交 Pull Request：  
[github.com/stanford-cs336/assignment1-basics-leaderboard](https://github.com/stanford-cs336/assignment1-basics-leaderboard)  
排行榜提交的详细说明请参考仓库中的 `README.md`。  

## 1.5 数据集来源
本次作业将使用两个预处理好的数据集：  
- TinyStories（Eldan 和 Li，2023）  
- OpenWebText（Gokaslan 等人，2019）  

两个数据集都是单个的大型纯文本文件。  如果你是在课程中做作业，可以在任何非 head 节点机器的 `/data` 目录找到它们。如果你是在家跟做，可以用 `README.md` 中的命令下载它们。  

---

- 低资源/降规模提示（Init）
   在整个课程作业讲义中，我们会给出一些提示，帮助你在缺少 GPU 资源或没有 GPU 资源的情况下完成作业。例如，有时会建议缩小数据集或模型规模，或者解释如何在 MacOS 集成 GPU 或 CPU 上运行训练代码。  
   这些“低资源提示”会用蓝色方框标出。即使你是注册的斯坦福学生并有课程机器的访问权限，阅读这些提示也能帮助你更快迭代、节省时间。  

---

- 低资源/降规模提示：在 Apple Silicon 或 CPU 上运行作业 1  
   使用助教提供的参考代码，我们可以在一台配备 36 GB 内存的 Apple M3 Max 芯片上，在 **Metal GPU（MPS）** 模式下不到 5 分钟内训练出一个能够生成相对流畅文本的语言模型，用 CPU 训练则大约需要 30 分钟。  
   如果这些术语对你来说比较陌生，不必担心！只要你的笔记本电脑比较新、实现正确且高效，你就能训练出一个小型语言模型，生成简单儿童故事且流畅度不错。  
   作业后面会介绍如果你是在 CPU 或 MPS 上运行，需要做哪些调整。  

# 2 BPE Tokenizer

我们将训练并实现一个字节级的字节对编码（BPE）分词器 [Sennrich 等人，2016；Wang 等人，2019]。
我们会将任意（Unicode）字符串表示为一系列字节，并在这个字节序列上训练我们的 BPE 分词器。之后，我们会使用这个分词器将文本（字符串）编码成 tokens（整数序列）。

## 2.1 Unicode 标准
Unicode 是一种文本编码标准，用于将字符映射到整数码点。截至 Unicode 16.0（2024 年 9 月发布），该标准定义了 154,998 个字符，涵盖 168 种书写系统。
在 Python 中：
可以使用 ord() 函数将单个 Unicode 字符转换为它的整数表示；
可以使用 chr() 函数将整数 Unicode 码点转换为对应字符的字符串。
字符 “s” 的码点是 115（通常记作 U+0073，其中 U+ 是常规前缀，0073 是 115 的十六进制表示）。
字符 “牛” 的码点是 29275。
```python
print('Unicode: ', ord('s'), ord('牛'))
print('Character: ', chr(115), chr(29275))
```
> Unicode:  115 29275  
> Character:  s 牛

---

**Problem ：理解 Unicode（1 分）**

**(a)** chr(0) 返回的 Unicode 字符是什么？（一句话回答。）
```python
chr(0)
```
> '\x00'  
> '\x00'是十六进制的 `00` 的字节，表示的是一个空字符，'\x00' 是这个字符的 **字符串表示**（repr）

**(b)** 这个字符的字符串表示（`__repr__()`）与打印出来的表示有什么区别？（一句话回答。）
```python
chr(0).__repr__()
```
> "'\\x00'"

> 
**(c)** 当这个字符出现在文本中会发生什么？（一句话回答。）

```python
'this is a test' + chr(0) + 'string'
print('this is a test' + chr(0) + 'string')
```
> 'this is a test\x00string'  
> this is a teststring

## 2.2 Unicode 编码
虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器并不现实，因为**词表会非常大**（大约 15 万个条目）且**稀疏**（很多字符非常少见）。因此，我们使用 Unicode 编码，它可以将一个 Unicode 字符转换为一系列字节。Unicode 标准本身定义了三种编码方式：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主流编码（占网页总数的 98% 以上）。

> UTF-8 是一种 Unicode 字符编码方式，它的作用是把字符（比如 a、牛、🌍）转换成 字节序列，方便计算机存储和传输。特点：
> 1. 可变长度
>    - ASCII 字符（0–127）用 1 个字节表示。（**ASCII 字符**：包括数字0-9，大小写英文字母，标点符号，空格，控制字符（如：空字符0-NUL，删除字符127-DEL））
>    - 其他字符用 2–4 个字节表示。例如："a" → 0x61（1 字节）；"牛" → 0xe7 0x89 0x9b（3 字节）；"🌍" → 0xf0 0x9f 0x8c 0x8d（4 字节）
> 
> 2. 向后兼容 ASCII
>    - 所有 ASCII 字符在 UTF-8 下编码结果和 ASCII 本身完全一致。
> 
> 3. 自我同步
>    - 可以通过字节的前几位轻松判断一个字符的开始和长度，不会破坏原来的文本结构。
> 
> 4. 用途广泛
>    - 是互联网最常用的编码方式（几乎所有网页和现代编程语言都支持）。
> 
> **总结一句话**：UTF-8 就是把 Unicode 字符映射成 1–4 个字节的规则，使文本既节省空间又兼容 ASCII。

要将 Unicode 字符串编码为 UTF-8，可以在 Python 中使用 encode() 函数。要访问 Python bytes 对象的底层字节值，可以对它进行迭代（例如使用 list()）。最后，我们可以使用 decode() 函数将 UTF-8 字节串解码回 Unicode 字符串。

```python
test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
print(type(utf8_encoded))
print(utf8_encoded.decode("utf-8"))
```
>b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'  
><class 'bytes'>  
>hello! こんにちは! 

```python
print(list(utf8_encoded))
```
> [104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]

```python
print(len(test_string))
print(len(utf8_encoded))
```
>13  
>23

通过把 Unicode 码点（codepoints）转换成字节序列（比如使用 UTF-8 编码），我们把原本的整数序列（每个整数代表一个字符，范围大约是 0 到 154,997）变成了 字节值序列（每个字节的整数范围是 0 到 255）。字节只有 256 种可能，比直接用 Unicode 码点的 15 万多个字符要容易管理得多。任何输入文本都可以表示为 0–255 的整数序列，因此不会出现模型训练时没见过的 token。

---

**Problem ：Unicode 编码 (3 分)**

(**a**) 为什么我们更倾向于在 UTF-8 编码的字节上训练分词器，而不是 UTF-16 或 UTF-32？(一句到两句的回答。)
提示：可以对比不同编码方式下相同输入字符串的输出结果。

| 编码方式    | 每个字符大小   | 是否变长 | 优缺点 |
| ---------- | ------------- | -------- | ----- |
| **UTF-8**  | 1–4 字节   | ✅ 是  | 高效，兼容 ASCII（英文最省空间），互联网最常用 |
| **UTF-16** | 2 或 4 字节 | ✅ 是  | 对中文较省空间，但实现复杂              |
| **UTF-32** | 固定 4 字节  | ❌ 否  | 简单，但极度浪费内存                 |

`Hello 🌍 你好` 在不同编码方式下的字节长度
- UTF-8：17 字节
- UTF-16（带 BOM）：24 字节
- UTF-32（带 BOM）：44 字节

对于相同字符串大小的输入：UTF-8 编码的字节长度最短，UTF-16/UTF-32 处理的是 码点（几十万种），或者至少要处理 65,536（2 字节）的组合，词表规模巨大，不利于高效训练。

(**b**) 考虑下面这个（错误的）函数，本意是将 UTF-8 字节串解码为 Unicode 字符串。请说明为什么它是错误的，并提供一个会导致错误结果的输入字节串。
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
```
提交内容：给出一个会导致 decode_utf8_bytes_to_str_wrong 产生错误结果的输入字节串，并用一句话解释为什么这个函数是错误的。
> encode("utf-8")：str → bytes  
> decode("utf-8")：bytes → str

错误字符串："你好"。
错误原因：以上编码是按照一个字节一个字节编码的，但是有的字符比如：中文，表情符号是多个字节编码的。

(**c**) 给出一个不能解码为任何 Unicode 字符的两个字节序列。（一个示例，一句话解释原因）

举例：b'\xE4\xBD'。原因：UTF-8 编码中，两个字节的码点只有**部分**对应字符

> UTF-8 是一种 **可变长度编码**，每个字符使用 **1 到 4 个字节**表示。
> - 字节分配规则  
>    - 前缀位（如 `110`、`1110`）表示这是多字节的开头。
>    - 后续字节都以 `10` 开头，表示它们是延续字节。 
> | 字节数 | 码点范围         | 二进制格式                     | 说明 |
> |--------|----------------|--------------------------------|------|
> | 1      | U+0000 ~ U+007F | 0xxxxxxx                        | ASCII 字符，兼容旧系统 |
> | 2      | U+0080 ~ U+07FF | 110xxxxx 10xxxxxx               | 包含西欧及部分特殊符号 |
> | 3      | U+0800 ~ U+FFFF | 1110xxxx 10xxxxxx 10xxxxxx      | 常用汉字、日文、韩文 |
> | 4      | U+10000 ~ U+10FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx | 较少使用的辅助平面字符 |
> 
> - 编码示例，以汉字“**你**”（`U+4F60`）为例：
> 
>    1. 二进制码点：`0100 1111 0110 0000`（16 位）  
>    2. UTF-8 需要 3 个字节：  
>       - 按规则：`1110xxxx 10xxxxxx 10xxxxxx`  
>       - 填充二进制码点：
>      ```
>      11100100 10111101 10100000
>      ```  
>    - 转换为十六进制：`E4 BD A0`  

> - **ASCII 兼容**：0~127 的字符直接使用 1 字节，兼容老系统。  
> - **可变长度**：英文字符占 1 字节，汉字占 3 字节，节省存储空间。  
> - **自动同步**：UTF-8 的字节序列可以唯一识别字符边界，不易出错。

## 2.3 子词分词

虽然字节级分词可以缓解基于词的分词器遇到的“未登录词”（out-of-vocabulary, OOV）问题，但将文本分成字节会导致**输入序列非常长**。这会减慢模型训练速度：例如，一个包含 10 个单词的句子，在基于词的语言模型中可能只有 10 个 token，但在字符级模型中可能有 50 个或更多 token（取决于单词长度）。处理这些更长的序列会增加模型每一步的计算量。此外，对字节序列进行语言建模比较困难，因为更长的输入序列会在数据中产生长期依赖关系。

子词分词（subword tokenization）位于**词级分词器**和**字节级分词器**之间。注意，字节级分词器的词表只有 256 个条目（字节值为 0 到 255）。子词分词器通过**增加词表大小**来换取更好地压缩输入字节序列。例如，如果字节序列 b'the' 在训练文本中频繁出现，为其分配一个词表条目就可以把这个原本需要 **3** 个 token 的序列压缩为 **单个** token。

如何选择这些子词单元加入词表呢？Sennrich 等人 [2016] 提出使用 **字节对编码**（BPE, Byte-Pair Encoding; Gage, 1994），这是一种压缩算法，它通过迭代地将出现频率最高的 **字节对** 替换（“合并”）为一个新的、未使用的索引。需要注意的是，这个算法会把子词 token 加入词表，以 **最大化输入序列的压缩效果**——如果某个词在输入文本中出现足够多次，**它将被表示为单个子词单元**。

通过 BPE 构建词表的子词分词器通常称为 BPE 分词器。在本次作业中，我们将实现一个 **字节级** BPE 分词器，其词表条目为字节或字节序列的合并结果，这样既能处理未登录词，又能保持 **可管理的输入序列长度**。构建 BPE 分词器词表的过程被称为 **训练 BPE 分词器**。

## 2.4 BPE Tokenizer 步骤

BPE 分词器的训练过程主要包括三个步骤：

1. 词表初始化 (Vocabulary initialization)  
   分词器的词表是一一映射的结构：从字节串 token 到整数 ID。由于我们训练的是字节级 BPE 分词器，初始词表就是**所有可能的字节集合**。由于字节可能取值 0–255，因此初始词表大小为 256。

2. 预分词（Pre-tokenization）  
   一旦有了词表，就可以统计文本中相邻字节出现的频率，然后从最频繁的字节对开始合并。然而直接在整个语料上每次都统计相邻字节对非常耗费计算资源。此外，直接合并可能导致仅标点不同的 token 拥有不同的 ID（例如 dog! 和 dog.），而它们语义上可能非常相似。

   为避免这种情况，我们先**预分词**。可以把它理解为对语料的粗粒度分词，用于统计字符对出现的频率。例如，单词 text 作为一个预 token 出现 10 次，那么在统计字符 t 和 e 相邻出现次数时，可以直接增加 10 次，而不必遍历整个语料。由于是字节级 BPE，每个预 token 用 UTF-8 字节序列表示。

   Sennrich 等人 [2016] 的 BPE 原实现通过空格分词（s.split(" ")）进行预分词，而 GPT-2 使用的是基于正则的预分词器（Radford 等人, 2019）
   ```python
   import regex as re
   PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
   re.findall(PAT, "some text that i'll pre-tokenize")
   ```
   >['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
   
   > 这个正则表达式是 GPT-2 用于 **预分词（pre-tokenization）** 的规则，其作用是把文本拆成适合 BPE 训练的“预 token”。它的各部分含义如下：
   > ```regex
   > '(?:[sdmt]|ll|ve|re)       # 匹配英语缩写的结尾，例如 's, 'd, 'm, 't, 'll, 've, 're
   > | ?\p{L}+                  # 匹配一个或多个字母（Unicode 字母），前面可能有一个空格
   > | ?\p{N}+                  # 匹配一个或多个数字，前面可能有一个空格
   > | ?[^\s\p{L}\p{N}]+        # 匹配一个或多个非空白、非字母、非数字字符，前面可能有一个空格（通常是标点）
   > | \s+(?!\S)                # 匹配空格，但不包括非空白字符后面的空格
   > | \s+                      # 匹配一个或多个空白字符
   > ```
   在实际构建从预 token 到计数的映射时，应使用 re.finditer 避免保存所有预分词结果。这两个函数的区别是： 
   > | 特点       | `re.findall`          | `re.finditer`                                |
   > | -------- | --------------------- | -------------------------------------------- |
   > | **返回结果** | **列表**（一次性返回所有匹配的字符串） | **迭代器**（逐个返回 `Match` 对象）                  |
   > | **内存使用** | 占用大，所有结果一次性存入内存       | 占用小，只在需要时生成一个匹配                          |
   > | **适合场景** | 小文本，结果数量有限            | 大文本或海量匹配结果（流式处理）                            |
   > | **速度**   | 一次性取出，速度快             | 边遍历边取，速度略慢（但更节省内存）                           |
   > | **灵活性**  | 得到字符串，需要再做处理          | 得到 `Match` 对象，可以直接用 `.group()`、`.span()` 等信息 |
   > | **风险**   | 大语料可能 **OOM（内存溢出）**   | 几乎不会 OOM，可处理超大文件                             |


3. 计算 BPE 合并（BPE merges）
   1. 将输入文本转换为预 token 并表示为 UTF-8 字节序列后，就可以计算 BPE 合并（即训练 BPE 分词器）。
   2. BPE 算法会迭代统计每对字节出现的次数，找到频率最高的字节对 (A, B)。
   3. 每次出现的该字节对都被合并为新的 token AB，并加入词表。
   4. 最终词表大小 = 初始词表大小（256）+ 合并操作次数。
   5. 为了训练效率，不考虑跨越预分词边界的字节对。  
      原始的 BPE 公式 [Sennrich 等人, 2016] 指定需要包含一个“单词结束”token。而在训练字节级 BPE 模型时，我们不添加单词结束 token，因为所有字节（包括空格和标点）都包含在模型的词表中。由于我们明确表示了空格和标点，学习到的 BPE 合并规则自然会反映这些单词边界。
   6. 若出现频率相同的字节对，按 **字典序** 选取较大的对进行合并，例如：
      ```python
      max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
      ```
      > ('BA', 'A')

4. 特殊 token
   有些字符串（如 <|endoftext|>）用于编码元数据（如文档边界），通常希望这些字符串 **不被拆分**，保持为单个 token。因此需要将它们加入词表，并分配固定 ID。

举个例子： 
![BPEeg](..\images\BPE_eg.jpg)

## 2.5 BPE Tokenizer 训练

第 1 节中有下载该数据集的说明，建议先浏览 TinyStories 数据集。  
第一节给的下载数据的方法没成功，用的以下方法：
```python
# 下载数据并保存
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
ds.save_to_disk('../data/TinyStories')

# 然后再用下面的代码保存
text_train = "\n<|endoftext|>\n".join(p['text'].replace("\n\n", "\n") for p in dataset['train'])
text_valid = "\n<|endoftext|>\n".join(p['text'].replace("\n\n", "\n") for p in dataset['validation'])

# 写入到一个 txt 文件
with open("../data/TinyStoriesV2-GPT4-train.txt", "w", encoding="utf-8") as f:
    f.write(text_train)
with open("../data/TinyStoriesV2-GPT4-valid.txt", "w", encoding="utf-8") as f:
    f.write(text_valid)
```

1. 并行化预分词  
   你会发现预分词步骤是一个主要的瓶颈。可以通过使用内置库 `multiprocessing` 并行化代码来加速预分词。具体而言，我们建议在并行实现中，将语料**分块**（chunk），并确保分块边界出现在特殊 token 的开头。你可以直接使用以下链接提供的起始代码来获取分块边界，然后将工作分配到不同进程中。[pretokenization_example.py](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py) （这个代码也本项目的在 `assignment1\cs336_basics\pretokenization_example.py` 中）
   这种分块方式总是有效的，因为我们从不希望跨文档边界合并 token。在作业中，你可以始终采用这种方式分块。无需担心遇到非常大的语料且不包含 `<|endoftext|>` 的边缘情况。

2. 在预分词前移除特殊 token　　
   使用正则模式（`re.finditer`）进行预分词之前，你应该从**语料**（如果使用并行实现的话就是**分块**）中剔除所有特殊 token。确保在特殊 token 处进行分割，这样它们之间的文本不会被合并。例如，如果语料（或分块）是 `[Doc 1]<|endoftext|>[Doc 2]`，你应该在特殊 token `<|endoftext|>` 处分割，并分别对 `[Doc 1]` 和 `[Doc 2]` 进行预分词，这样文档边界之间就不会发生合并。可以使用 `re.split` 并将 `"|".join(special_tokens)` 作为分隔符（注意使用 `re.escape` 处理特殊 token 中可能出现的 `|`）。测试 `test_train_bpe_special_tokens` 会验证这一点。

3. 优化合并步骤
   上面示例中的 BPE 训练的朴素实现速度较慢，因为每次合并时都需要遍历所有字节对来找出最频繁的字节对。然而，合并后只有与已合并字节对重叠的字节对的计数会发生变化。因此，可以通过对所有字节对计数进行索引，并增量更新这些计数，而不是显式地遍历每个字节对来计算频率，从而提高 BPE 训练速度。使用这种缓存方法可以显著加速训练，但需要注意，BPE 训练中的合并步骤在 Python 中无法并行化。

低资源/降规模技巧：
1. 性能分析（Profiling）    
   你应该使用性能分析工具（如 **cProfile** 或 **scalene**）来识别实现中的性能瓶颈，并重点优化这些部分。  

2. “降规模（Downscaling）”    
   在直接用完整的 **TinyStories** 数据集训练分词器之前，我们建议你先在一个较小的数据子集（即“调试数据集”）上进行训练。例如，你可以先在 TinyStories 的**验证集** 上训练，它只有 **2.2 万个文档**，而不是完整数据集的 **212 万个文档**。  

这说明了一种通用的“降规模”开发策略：在可能的情况下，使用更小的数据集、更小的模型规模等来加快开发。  
但要注意合理选择调试数据集的大小或超参数配置：  
- 调试集要足够大，以保证暴露出与完整配置相同的瓶颈，这样你做的优化才能推广  
- 但又不能太大，否则运行起来太耗时。 

---
 
**Problem ：BPE 分词器训练（15 分）**
**任务交付内容**：  
编写一个函数，给定输入文本文件路径，训练一个**字节级 BPE 分词器**。  
你的 BPE 训练函数应当至少支持以下输入参数：  
- **input_path: str**  
  输入文本文件的路径，用于 BPE 分词器训练数据。  
- **vocab_size: int**  
  一个正整数，定义最终词表的最大大小（包括初始字节词表、合并生成的词表项，以及所有特殊 token）。  
- **special_tokens: list[str]**  
  一个字符串列表，指定要加入词表的特殊 token。这些特殊 token 不会影响 BPE 训练过程.

你的 BPE 训练函数应返回以下结果：  
- **vocab: dict[int, bytes]**  
  分词器词表，一个从 `int`（词表中的 token ID）映射到 `bytes`（token 字节序列）的字典。  
- **merges: list[tuple[bytes, bytes]]**  
  BPE 训练过程中产生的合并操作列表。每个列表元素是一个字节对 **`(<token1>, <token2>)`**，表示 `<token1>` 与 `<token2>` 被合并。这些合并操作必须**按照生成顺序排列**。  

**测试方法**：  
要用我们提供的测试来检验你的 BPE 训练函数，首先需要在 **[adapters.run_train_bpe]** 中实现测试适配器。  
然后运行：`uv run pytest tests/test_train_bpe.py` 。你的实现应能通过所有测试。

**可选**（这可能需要大量时间），可以使用某些系统语言（例如C++（可考虑使用cppyy）或Rust（使用PyO3）来实现训练方法的关键部分。如果这样做，请注意哪些操作需要复制，哪些可以直接从Python内存中读取，并确保提供构建说明，或确保仅使用pyproject.toml即可构建。  
另外请注意，GPT-2的正则表达式在大多数正则表达式引擎中支持不佳，在支持的引擎中也大多速度过慢。我们已经验证Oniguruma速度合理且支持负向先行断言，但Python中的regex包甚至更快

 






# 其他
## 1. python库 `abc`
**抽象类**：一种不能直接实例化的类，只能作为基类被继承。用来描述一类对象的共性，但不提供完整实现。特点：
1. 不能直接创建对象。
2. 可以包含普通方法和抽象方法。
   **抽象方法**：在抽象类中声明但没有具体实现的方法。强制子类必须实现，否则子类也不能实例化。  
3. 主要用来规定“子类**必须实现**什么功能”。  

**举个例子**：
```python
from abc import ABC, abstractmethod

class Car(ABC):  # 抽象类：定义所有汽车的共性
    def __init__(self, brand):
        self.brand = brand

    @abstractmethod
    def drive(self):   # 抽象方法：规定必须有“开车”的功能
        pass

    @abstractmethod  # @abstractmethod 的作用是把一个方法标记为抽象方法。
    def fuel_type(self):  # 抽象方法：规定必须说明“用什么燃料”
        pass

class Tesla(Car):  # 具体类：电动车
    def drive(self):
        return f"{self.brand} is driving silently."

    def fuel_type(self):
        return "Electric"

class Toyota(Car):  # 具体类：燃油车
    def drive(self):
        return f"{self.brand} is driving with engine sound."

    def fuel_type(self):
        return "Gasoline"


# 实例化（造车）
t1 = Tesla("Tesla Model 3")
t2 = Toyota("Toyota Corolla")

print(t1.drive())       # Tesla Model 3 is driving silently.
print(t1.fuel_type())   # Electric

print(t2.drive())       # Toyota Corolla is driving with engine sound.
print(t2.fuel_type())   # Gasoline
```

## 2. python 库 `collections`
**collections** 是 Python 的一个 内置标准库。它提供了一些 比内置数据类型（list、dict、tuple、set）更高效、更专用 的数据结构。用于优化性能、代码可读性，或者解决某些特定场景的问题。
`defaultdict` 是 dict 的子类：当用 d[key] 访问不存在的键时，不报 KeyError，而是用一个“工厂函数”自动创建默认值并写入字典。
