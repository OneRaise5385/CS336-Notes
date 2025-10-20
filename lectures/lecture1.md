# CS336: Language Modeling from Scratch
# 一、从零开始构建语言模型

## 1. 关于这门课程
### 1.1 课程简介: 
- 语言模型开启了一个全新的范式———通过一个通用系统来解决各种**下游任务**。本课程将介绍语言模型创建的各个环节———数据收集与清洗、Transformer 模型的构建、模型训练，以及部署前的评估。
- **开设这门课的原因**：科研人员正逐渐与底层技术脱节（自己实现模型$\rightarrow$对现有模型微调$\rightarrow$写提示词），但是基础研究必须深入了解底层技术才能完成。本课程的目标就是通过**搭建**语言模型来深入理其深层原理。这门课会培养工程能力，注重实现与优化。
  > **下游任务**：
  > 指利用已经训练好的语言模型在具体应用场景中执行的各种自然语言处理任务。例如文本分类、命名实体识别（NER，例如识别人名、地名、组织名等）、问答系统（如基于知识库或文档的问答）、文本生成（如对话生成、文章续写、代码生成、文本摘要（生成文章或文档的简要版本）、机器翻译（如英文翻译成中文）、语义相似度判断（判断两段文本是否语义相近）、文本纠错与语法改进等
- **先决条件**：python，深度学习与系统优化，微积分与线性代数，概率与统计，机器学习基础
- **课程内容**及**作业安排**。课程一共包含5个部分：basics、systems、scaling laws、data、alignment。每个部分对应一次作业，没有代码框架（scaffolding），但会提供单元测试与接口适配器帮助你检查功能是否正确。在本地实现和测试，通过后再提交至集群运行以进行准确率与速度评估。部分作业会设有排行榜。**尽量不使用AI编程工具**。
![design](..\images\design-decisions.png)

### 1.2 模型更大的时候会有所不同
前沿的模型对我们来说遥不可及，本课程中构建的小语言模型无法完全代表大预言模型的真实特征。
> **直觉**（intuitions）：有些设计决策目前**根本无法（或尚且无法）通过理论充分解释**，往往只能靠反复实验得出。
- 例1：随着模型规模扩大，注意力机制在计算中所占的比例迅速减少，而前馈层变成了真正的主力计算负担。
  ![](images\roller-flops.png)
  | 列名                | 含义                                  |
  | ----------------| ---------------------------------------- |
  | **description** | 模型的规模（参数数量）                     |
  | **FLOPs/update**| 每次参数更新所需的总浮点运算次数（FLOPs）    |
  | **FLOPS MHA**   | 多头注意力（Multi-Head Attention）部分占总 FLOPs 的百分比 |
  | **FLOPS FFN**   | 前馈神经网络（Feed-Forward Network）部分的 FLOPs 占比     |
  | **FLOPS attn**  | 计算注意力权重（attention scores）的 FLOPs 占比           |
  | **FLOPS logit** | 输出 logits（即分类器部分）的 FLOPs 占比                  |
- 例2：随着**模型规模**的增加，模型会在某些任务上突然出现**能力涌现（emergent abilities）**。
  ![](images\wei-emergence-plot.png)


### 1.3 这门课上能学到的能够迁移到前沿模型的知识
- 运行机制：Transformer原理、模型并行利用GPU
- 思维方式：如何最大限度地榨干硬件性能、认真对待规模问题（例如规模法则scaling laws）
- 直觉经验：在数据选择和建模决策中，哪些因素有助于提高准确率
机制和思维方式是可以迁移到大型模型上的，而直觉经验在不同规模下并不总是适用的。
> **Scaling laws**（规模法则）：通过大量实验观察，随着模型的规模、训练数据量和计算量的增长，模型性能系统性地提升的经验规律。[Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361)
>| 增加的因素| 对性能的影响（正面） | 影响机制说明 | 注意事项 |
>| ------------- | -------------------- | ---------- | ---------- |
>| **参数量** | 提升模型表达能力 | 更大的模型可以学习更复杂的模式，有能力拟合更大的语料和上下文信息| 如果数据量不足，可能导致过拟合。超大模型计算开销显著增加|
>| **数据量** | 提升模型泛化能力 | 学习更多样本，模型会拥有更全面的语言规律，降低训练偏差与过拟合风险|如果模型太小，无法充分利用大量数据（参数成为瓶颈）|
>| **计算量** | 提升整体训练质量 | 可以训练更久，达到更低损失。支持更大 batch size、更长训练周期、更多训练 steps。也代表更大的硬件资源投入 | 计算资源要与数据量/模型规模相匹配，不然会“浪费”预算（比如训练很小的模型却用超多计算） |


### 1.4 效率驱动设计
**错误**的理解：只有规模重要，算法不重要。
**正确**的理解：能**扩展**（scale）的**算法**才是关键。在大规模下效率更为重要，因为资源消耗巨大，不能浪费。
**$ \text{accuracy} = \text{efficiency} \times \text{resources} $**
思考方式的转变：在给定的计算资源和数据预算下，你能构建的最优模型是什么？核心目标是：**最大化效率**，即在固定资源条件下，如何训练出最优的模型？
**$ \text{resourse} = \text{data} + \text{hardware} $**
目前，我们受限于计算资源，因此必须最大限度地利用现有硬件性能：
- 数据处理（Data processing）：避免在低质量或无关数据上浪费宝贵的计算资源。
- 分词（Tokenization）：虽然直接处理原始字节非常优雅，但对当前模型架构而言，这样做计算效率极低。
- 模型架构（Model architecture）：很多架构上的变动，都是为了减少内存消耗或降低 FLOPs，例如：共享 KV 缓存、滑动窗口注意力等。
- 训练（Training）：在优化策略得当的情况下，**只进行一次完整训练**(single epoch)也可能足够。
- 扩展法则（Scaling laws）：使用较小模型进行超参数调优，从而节省计算资源。
- 对齐（Alignment）：如果模型能更好地对齐于具体用途，那么所需的基础模型可以更小。
在不远的未来，我们将会面临**数据受限**（data-constrained）的局面

## 2. 语言模型的发展状况
### 2.1 发展历程
1. 神经网络之前（2010年之前）
- 使用语言模型来衡量英文的熵[Shannon 1950]
- 大量基于 n-gram 的语言模型研究（应用于机器翻译、语音识别等）[Brants 等，2007]
 
2. 神经网络成分出现（2010 年代）
- 第一种神经语言模型[Bengio 等，2003]
- 序列到序列建模（用于机器翻译）[Sutskever 等，2014]
- Adam 优化器[Kingma 等，2014]
- 注意力机制（用于机器翻译）[Bahdanau 等，2014]
- Transformer 架构（用于机器翻译）[Vaswani 等，2017]
- Mixture of Experts（专家混合模型）[Shazeer 等，2017]
- 模型并行技术[Huang 等，2018]，[Rajbhandari 等，2019]，[Shoeybi 等，2019]
 
3. 早期基础模型（2010 年代末）
- ELMo：基于 LSTM 的预训练，微调能显著提升性能[Peters 等，2018]
- BERT：基于 Transformer 的预训练，广泛提升多任务性能[Devlin 等，2018]
- Google T5（110 亿参数）：将所有任务统一为文本到文本的格式[Raffel 等，2019]
 
4. 拥抱扩展性，走向闭源
- OpenAI GPT-2（15 亿参数）：生成流畅文本，首次展示 zero-shot 能力，采用分阶段发布策略[Radford 等，2019]
- Scaling laws（扩展法则）：提供规模扩展的可预测性与希望[Kaplan 等，2020]
- OpenAI GPT-3（175 亿参数）：引入上下文学习（in-context learning），完全闭源[Brown 等，2020]
- Google PaLM（5400 亿参数）：超大规模，但训练轮数不足[Chowdhery 等，2022]
- DeepMind Chinchilla（700 亿参数）：遵循计算最优的 scaling laws[Hoffmann 等，2022]
  > **闭源模型**（如 GPT-4o）：仅通过 API 提供访问[OpenAI，2023]
  > **开放权重模型**（如 DeepSeek）：开放模型权重与结构细节，但不公开数据[DeepSeek-AI，2024]
  > **开源模型**（如 OLMo）：开放模型权重与训练数据，并提供较完整论文（可能缺乏失败实验与设计动机）[Groeneveld，2024] 

5. 开源模型
- EleutherAI 提供开放数据集（The Pile）和模型（如 GPT-J）[Gao 等，2020] [Wang 等，2021]
- Meta OPT（175B）：尝试复现 GPT-3，过程中遇到大量硬件问题[Zhang 等，2022]
- Hugging Face / BigScience 的 BLOOM：注重数据来源与开放性[Workshop等，2022]
- Meta 的 Llama 系列[Touvron 等，2023] [Grattafiori 等，2024]
- 阿里巴巴的 Qwen 模型系列[Qwen 等，2024]
- DeepSeek 系列模型[DeepSeek-AI 等，2024]
- AI2 的 OLMo 2 模型[Groeneveld 等，2024] [OLMo 项目，2024]

6. 当今前沿模型（2025 年）

- [OpenAI o3](https://openai.com/index/openai-o3-mini/)，[Anthropic Claude Sonnet 3.7](https://www.anthropic.com/news/claude-3-7-sonnet)，[xAI Grok 3](https://x.ai/news/grok-3)，[Google Gemini 2.5](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/)，[Meta Llama 3.3](https://ai.meta.com/blog/meta-llama-3/)，[DeepSeek r1](https://www.deepseek.com/)，[Alibaba Qwen 2.5 Max](https://qwenlm.github.io/blog/qwen2.5-max/)，[Tencent Hunyuan-T1](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)


### 2.2 工业界的语言模型
GPT-4 拥有 1.8 万亿个参数，训练成本高达 1 亿美金；xAI 使用 20 万张 H100 GPU 组建集群，用于训练 Grok 模型；Stargate 项目计划在 4 年内投资 5000 亿美元。
> [星际之门计划（Stargate Project）](https://openai.com/index/announcing-the-stargate-project/)是美国一家人工智能合资企业，由OpenAI 、软银、甲骨文和投资公司 MGX 联合创立。该合资公司计划到2029年在美国的人工智能基础设施上投资高达**5000亿美元**。这一计划于2022年启动，并在2025年1月21日由美国总统唐纳德·川普正式宣布。该公司在美国特拉华州注册成立，名称为Stargate LLC。软银首席执行官孙正义被任命为公司董事长。

## 3. 章节简介
### 3.1 basics
![mindmap](..\images\mindmap-01.png)
1. Tokenization
   本课程使用的是**字节对编码分词器**（Byte-Pair Encoding, **BPE**）
   **分词器**（tokenizer）:负责将字符串和**整数序列**（strings and sequences of integers）(**token**)相互转换，如下图：
   ![tokenizer](..\images\tokenized-example.png)
   **无分词器（Tokenizer-free）方法**：直接使用字节（bytes）作为输入，思路有前景，但目前尚未在前沿大模型规模上验证。
2. 模型架构（Architecture）
   起点：原始 Transformer [Vaswani+ 2017]
   变体： 
   激活函数：ReLU，SwiGLU [Shazeer 2020]
   位置编码（Positional Encoding）：正弦位置编码（sinusoidal）、RoPE [Su+ 2021]
   归一化方法（Normalization）：LayerNorm, RMSNorm [Ba+ 2016][Zhang+ 2019]
   归一化位置：Pre-Norm vs Post-Norm [Xiong+ 2020]
   MLP 结构：全连接（dense）、专家混合（Mixture of Experts, MoE）[Shazeer+ 2017]
   注意力机制：全注意力（full）、滑动窗口（sliding window）、线性注意力（linear）[Jiang+ 2023][Katharopoulos+ 2020]
   降维注意力：组查询注意力（GQA）、多头潜在注意力（MLA）[Ainslie+ 2023][DeepSeek-AI+ 2024]
   状态空间模型（State-space models）：Hyena [Poli+ 2023]
3. 训练（Training）
   优化器（Optimizer）：AdamW、Muon、SOAP [Kingma+ 2014][Loshchilov+ 2017][Keller 2024][Vyas+ 2024]
   学习率调度（Learning Rate Schedule）：余弦退火（cosine）、WSD [Loshchilov+ 2016][Hu+ 2024]
   批大小（Batch Size）：如临界批大小（critical batch size）[McCandlish+ 2018]
   正则化（Regularization）：Dropout、权重衰减（weight decay）
   超参数（Hyperparameters）：如注意力头数、隐藏层维度，使用网格搜索（grid search）

作业 1  [Github](https://github.com/stanford-cs336/assignment1-basics)
任务：
1. 实现 BPE 分词器
2. 实现 Transformer、交叉熵损失、AdamW 优化器、训练循环
3. 在 TinyStories 和 OpenWebText 数据集上训练
4. 排行榜：在 H100 上运行 90 分钟，使 OpenWebText 困惑度（perplexity）最小化

### 3.2 systems
充分发挥硬件的性能
![mindmap](..\images\mindmap-02.png)
1. 核函数（Kernels）
本节通过最小化数据移动来组织计算，从而最大化 GPU 利用率，编写核函数的工具：CUDA / Triton / CUTLASS / ThunderKittens
计算都发生在 GPU 上面，GPU 长下面这样
![A100](..\images\A100.png)
DRAM，SRAM 是 GPU 的组成部分，DRAM 相当于库房，SRAM + compute 相当于工厂
![gpu](..\images\gpu_analogy.png)
2. 并行化（Parallelism）
多块 GPU 之间的数据传输效率很低，需要遵循“尽量减少数据移动”的原则
![parallelism](..\images\parallelism.png)
使用集合操作（collective operations），例如：gather（收集）、reduce（归约）、all-reduce（全归约）
在多块 GPU 之间分片（shard）参数、激活值、梯度和优化器状态
计算的拆分方式可以是：数据并行（data parallelism）、张量并行（tensor parallelism）、流水线并行（pipeline parallelism）、序列并行（sequence parallelism）
3. 推理（Inference）
**推理**：根据提示词生成 token，这是使用模型时必须要进行的。用于强化学习、测试时计算（test-time compute）、评估。全球范围看，推理每次使用的总计算量大于训练计算量
推理分两个阶段：
   1. Prefill（预填充）：类似训练，已知 token，可一次性批量处理（计算受限，compute-bound）
   2. Decode（解码）：需要一次生成一个 token（受限于内存，memory-bound）
加速解码的方法：
   1. 使用更便宜的模型（模型剪枝、量化、蒸馏）
   2. 推测解码（speculative decoding）：先用更便宜的“草稿”模型生成多个 token，然后用完整模型并行打分（得到精确解码结果）
   3. 系统优化：KV 缓存（KV caching）、批处理（batching）

作业 2
1. 在 Triton 中实现融合版 RMSNorm 核函数
2. 实现分布式数据并行训练
3. 实现优化器状态分片
4. 对实现结果进行基准测试和性能分析

### 3.3 scaling_laws
目标：在小规模上做实验，找到问题所在，预测大规模时的超参数和损失。
在给定 FLOPs 预算 $C$ 的情况下，模型规模 $N$，token 数量 $D$ 按照以下规律测算： [Kaplan+ 2020][Hoffmann+ 2022]
$$ D^* = 20 N^* $$
例如，一个 14 亿参数的模型应在 280 亿 token 上进行训练。但这个规律并没有考虑推理成本！

作业 3  [GitHub](https://github.com/stanford-cs336/spring2024-assignment3-scaling)

- 基于先前的运行结果，我们定义了一个训练 API（超参数 → 损失）
- 提交“训练任务”（在 FLOPs 预算范围内）并收集数据点
- 对这些数据点拟合一个缩放规律
- 提交放大规模后的超参数预测值
- 排行榜目标：在给定 FLOPs 预算下，将损失最小化

### 3.4 data
我们期望模型拥有的能力，通常需要使用相关数据来训练。
![mindmap](..\images\mindmap-03.png)
1. 评估
**困惑度**（**Perplexity**）：语言模型的经典评估指标。**标准化测试**（MMLU、HellaSwag、GSM8K），**指令遵循**（例AlpacaEval、IFEval、WildBench）。增加测试阶段的计算量：链式思维（chain-of-thought）、集成（ensembling）。可以以语言模型作为裁判来评估生成任务。**完整系统**：检索增强生成（RAG）、智能体（agents）。

1. 数据精选
数据不会凭空出现，一般从互联网爬取的网页、书籍、arXiv 论文、GitHub 代码等地方获取。网络数据十分混杂，必须对数据进行处理。在使用某些数据（尤其是版权受保护的数据）时，可能需要合法地从数据拥有者那里获得授权，才能合法使用这些数据。[Henderson+ 2023]。数据并非纯文本的，常见格式有HTML、PDF、目录。

1. 数据处理
   1. 转换：将 HTML/PDF 转为文本（保留内容、部分结构、重写）
   2. 过滤：保留高质量数据，剔除有害内容
   3. 去重：节省计算资源，避免记忆化；使用布隆过滤器（Bloom filters）或 MinHash

作业 4  [GitHub](https://github.com/stanford-cs336/spring2024-assignment4-data)
- 将 Common Crawl 的 HTML 转为文本
- 训练分类器以过滤高质量和有害内容
- 使用 MinHash 做去重
- 排行榜目标：在给定 token 预算下，最小化困惑度

### 3.5 alignment
到目前为止，基础模型只是具备原始潜力，非常擅长预测下一个词。对齐（alignment）会让模型真正变得有用。
![mindmap](..\images\mindmap-04.png)
对齐的目标：
- 让语言模型能够遵循指令
- 调整风格（格式、长度、语气等）
- 融入安全机制（例如拒绝回答有害问题）
1. 两个阶段：
   1. 监督微调（Supervised finetuning，SFT）
   基础模型已经具备了相应的技能，需要少量的示例来激发它们，通过微调模型，使其最大化条件概率$p   (response | prompt)$。SFT的指令数据由$(prompt, response)$对组成，这些数据通常涉及人工标注。
      ```python
      sft_data: list[ChatExample] = [
          ChatExample(
              turns=[
                  Turn(role="system", content="You are a helpful assistant."),
                  Turn(role="user", content="What is 1 + 1?"),
                  Turn(role="assistant", content="The answer is 2."),
              ],
          ),
      ]
      ```
   
   1. 基于反馈的学习（learning_from_feedback）
   现在我们已经有了一个初步的指令执行模型。在不依赖昂贵标注的情况下，针对给定提示，使用模型生成多个   回答 [A, B]，用户提供偏好判断（A < B 或 A > B），以此来让模型性能更好。
      ```python
      preference_data: list[PreferenceExample] = [
          PreferenceExample(
              history=[
                  Turn(role="system", content="You are a helpful assistant."),
                  Turn(role="user", content="What is the best way to train a language model?   "),
              ],
              response_a="You should use a large dataset and train for a long time.",
              response_b="You should use a small dataset and train for a short time.",
              chosen="a",
          )
      ]
      ```
2. 验证器(Verifiers)
形式化验证器（例如，用于代码、数学的验证器）
学习型验证器：通过训练让语言模型充当“裁判”来进行验证

3. 算法(Algorithms)
近端策略优化（PPO）：一种强化学习方法[Schulman+ 2017][Ouyang+ 2022]
直接策略优化（DPO）：针对偏好数据，方法更简单[Rafailov+ 2023]
群体相对偏好优化（GRPO）：去除价值函数[Shao+ 2024]


作业 5
- 实现监督微调
- 实现直接偏好优化（Direct Preference Optimization，DPO）
- 实现群体相对偏好优化（Group Relative Preference Optimization，GRPO）

# 二、分词器
## 1 Tokenization
![mindmap_tokenization](..\images\mindmap_tokenization.png){width="600px" height="px"}
### 1.1 什么是Tokenization？
原始文本通常用 Unicode 字符串表示。语言模型会对一系列**词元**（tokens）（通常用整数索引表示）放置一个概率分布。因此，我们需要一种将字符串**编码**（encode）成词元的过程，同时需要一种将词元**解码**（decode）回字符串的过程。Tokenizer（分词器）就是一个实现了编码和解码方法的类。**词表大小**（vocabulary size）就是可能的词元（整数）总数。为了直观了解分词器是如何工作的，可以试用这个[交互式网站](https://tiktokenizer.vercel.app/?encoder=gpt2)。观察到的现象：
1. 一个单词与其前面的空格会被视为同一个 token（例如 " world"）。
2. 处在句首的单词和处在句中间的同一单词会被不同地表示（例如 "hello hello"）。
3. 数字会被分成每几位一个 token。
4. 以上的三条是课程里给的，第一条和第二条有点没看懂或者有问题（或者是我翻译的有问题）。
5. 同一个单词加空格和不加空格对应不同的 token，这个跟第二条说的有不一样的地方。测试的结果是课程中给的"hello hello"的例子中，两个之所以不一样，是因为第二个hello之前有空格，而第一个之前没有，当在第一个 hello 前加上空格后，两个对应的 token 是一样的。
6. 不同模型对应的 tokens 结果不同（如 gpt2 和 gpt4o）
7. 同一个单词大小写不同对应的 token 也不同（如 gpt2模型中 Science 为 5800，science 为 3783）
8. 同一个单词大小写不同的时候可能被划分为多个tokens（university 被划分为了两个部分）
9.  不同的模型对数字的划分风格不同，gpt4o 貌似只有三个数字划分这一种情况，gpt2 有两个数字划分在一起的情况
10. 换行符也有对应的 token，测试的 gpt2 和 gpt4o 的 token 都是198
![tiktoken](..\images\Tiktokenizer.png)

下图是 OpenAI 的 GPT-2 分词器（tiktoken）的实际运行示例。字符串 `Hello, 🌍! 你好!` 的 UTF-8 编码占用 20 字节。tokens 占 12 字节。$ratio_{compression} = 12 / 20 = 1.6666666667$。
![gpt2tiktoken](..\images\gpt2_tiktoken.png){width="600px" height="px"}

### 1.2 基于字符的分词器
**Unicode 字符串**是由 **Unicode 字符**组成的**序列**。每个字符可以通过 `ord` 转换成对应的码点（整数）。Unicode 总共有大约 15 万 个字符。
> **Unicode** 是一个全球统一的字符集，它为几乎所有语言的文字、符号、表情符号等分配了唯一的数字编号（叫码点，code point），这样计算机就能在全球范围内一致地存储和处理文本。**Unicode 字符串**就是一串这样的 Unicode 字符，每个字符在背后对应一个或多个码点。
> - Unicode 字符串不是直接用字节存的，而是一个字符序列，字符本身用 Unicode 编号来标识；等需要存储或传输时，可以用 UTF-8、UTF-16 等具体编码方式转成字节。

在例子 `Hello, 🌍! 你好!` 中，编码结果如下图所示。在 indices 中最大值为 127757 可以看出这个字符集是非常大的。
![charactertokenizer](..\images\charactertokenizer.png){width="600px" height="px"}
问题 1：这个字符集非常大。
问题 2：许多字符（例如 🌍）非常少见，这会导致词表的使用效率很低。

### 1.3 基于字节的分词器
Unicode 字符串可以表示为一系列字节，而字节又可以用 0 到 255 之间的整数表示。最常用的 Unicode 编码是 UTF-8。有些 Unicode 字符只需要一个字节表示，有些则需要多个字节。
![bytetokenizer](..\images\bytetokenizer.png){width="600px" height="px"}
使用基于字节的分词器优点是词表很小：一个字节最多表示 256 个值。但压缩率（compression ratio）非常差，也就是说分词后的序列会很长。由于 Transformer 的上下文长度有限（注意力计算是平方级增长的），这会导致效率很低。

### 1.4 基于单词的分词器
另一种方法是把字符串按单词拆分，这种方法更接近传统自然语言处理的做法。

```python
string = "I'll say supercalifragilisticexpialidocious!"
segments = regex.findall(r"\w+|.", string) 
```
这个正则表达式会把所有字母数字字符连续地当作一个整体保留下来，而不会把它们拆开。就像下面这样：
![wordtokenizer](..\images\wordtokenization.png){width="600px" height="px"}

要把它变成一个 Tokenizer，需要把这些片段（segments）映射成整数 ID，然后建立一个片段 → 整数的映射表。但是这种方法有几个问题：
1. 单词的数量非常庞大（就像 Unicode 字符一样）。
2. 很多单词非常罕见，模型学不到太多关于它们的东西。
3. 不容易得到固定大小的词表（vocabulary size）。
$ Size_{vocabulary} = Number_{distinct ~ segments ~ of ~ training ~ data}$
1. 对于训练中没见过的新单词，要用特殊的 UNK token 来代替，这会让结果变得很糟糕，还可能影响困惑度（perplexity）的计算。

### 1.5 基于BPE的分词器
**字节对编码**（Byte Pair Encoding，**BPE**）最早由 Philip Gage 于 1994 年提出，用于数据压缩。它后来被改进并应用到自然语言处理（NLP）中的神经机器翻译任务中。 [Sennrich 等，2015] 在此之前，研究论文通常使用基于单词的分词方法。BPE 随后被 GPT-2 采用。 [Radford 等，2019] GPT-2 论文中先用基于单词的分词方法将文本分成初始片段，再对每个片段运行原始的 BPE 算法。
**基本思想**：在原始文本上训练分词器，让它自动确定词表。常见的字符序列用单个 token 表示，稀有的字符序列则用多个 token 表示。**流程**：先把每个字节当作一个 token，然后不断合并出现频率最高的一对相邻 token。
1. 把字符串转成 UTF-8 编码的字节序列
2. 统计所有相邻 token 对的出现次数，找到出现次数最多的一对 token
3. 创建新 token，把新 token 对应的字节序列放进词表
4. 重复步骤 2、步骤 3， 共 `num_merges` 次

![bpe_steps](..\images\bpe_steps.jpg)
