from reference import Reference, join
from arxiv_util import arxiv_reference

shannon1950 = Reference(
    title="Prediction and Entropy of Printed English", date="1950-09-15",
    authors=["Claude Shannon"],
    url="https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf",
)

brants2007 = Reference(
    title="Language Models in Machine Translation", date="2007",
    authors=["Thorsten Brants", "Ashok C. Popat", "Peng Xu", "Franz J. Och", "Jeffrey Dean"],
    organization="Google",
    url="https://aclanthology.org/D07-1090.pdf",
    notes=join(
        "Trained 5-gram model on 2T tokens"
    )
)

bengio2003 = Reference(
    title="A Neural Probabilistic Language Model", date="2003-02-01",
    authors=["Yoshua Bengio", "Rejean Ducharme", "Pascal Vincent", "Christian Jauvin"],
    url="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf",
    notes="Used a feedforward neural network over last n words to predict the next word in a sequence"
)

susketver2014 = arxiv_reference("https://arxiv.org/pdf/1409.3215.pdf", organization="Google", notes="Introduced seq2seq (encode entire sentence into one vector, decode translation)")

adam2014 = arxiv_reference("https://arxiv.org/pdf/1412.6980.pdf", notes="Introduced Adam optimizer based on RMSProp and momentum")

bahdanau2015_attention = arxiv_reference("https://arxiv.org/pdf/1409.0473.pdf", notes="Introduced attention mechanism (for machine translation)")

transformer_2017 = arxiv_reference("https://arxiv.org/pdf/1706.03762.pdf", organization="Google", notes="Introduced Transformer (for machine translation)")

layernorm_2016 = arxiv_reference("https://arxiv.org/pdf/1607.06450.pdf", notes="Introduced LayerNorm")

adamw2017 = arxiv_reference("https://arxiv.org/pdf/1711.05101.pdf", notes=join(
    "Improves Adam by decoupling weight decay",
))

ppo2017 = arxiv_reference("https://arxiv.org/pdf/1707.06347.pdf", notes="Introduced PPO (for RL)")

large_batch_training_2018 = arxiv_reference("https://arxiv.org/pdf/1812.06162.pdf", notes="Introduced critical batch size")

elmo = arxiv_reference("https://arxiv.org/abs/1802.05365")
bert = arxiv_reference("https://arxiv.org/abs/1810.04805")

gpt2 = Reference(
    title="Language Models are Unsupervised Multitask Learners",
    authors=["Alec Radford", "Jeffrey Wu", "Rewon Child", "David Luan", "Dario Amodei", "Ilya Sutskever"],
    organization="OpenAI", date="2019-02-14",
    url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    notes=join(
        "1.5B parameters",
        "Pioneered stage release",
    ),
)

t5 = arxiv_reference("https://arxiv.org/pdf/1910.10683.pdf", organization="Google", notes=join(
    "Encoder-decoder, frames tasks as text-to-text",
    "Introduced Colossal Cleaned Common Crawl (C4)",
    "Filtering (Section 2.2)",
    "11B parameters",
    "Remove bias from feedforward layers",
))

cosine_learning_rate_2017 = arxiv_reference("https://arxiv.org/pdf/1608.03983.pdf")

moe_2017 = arxiv_reference("https://arxiv.org/pdf/1701.06538.pdf", organization="Google")

gpipe_2018 = arxiv_reference("https://arxiv.org/pdf/1811.06965.pdf", organization="Google")
megatron_lm_2019 = arxiv_reference("https://arxiv.org/pdf/1909.08053.pdf", organization="NVIDIA")
zero_2019 = arxiv_reference("https://arxiv.org/abs/1910.02054", organization="Microsoft", notes="Introduced ZeRO optimizer, can train 100B parameter model over 400 GPUs")

rms_norm_2019 = arxiv_reference("https://arxiv.org/abs/1910.07467")

############################################################
# 2020

kaplan_scaling_laws_2020 = arxiv_reference("https://arxiv.org/pdf/2001.08361.pdf", organization="OpenAI", notes=join(
    "Vary model size, dataset size, compute; get power laws",
    "Larger models require fewer tokens",
))

shazeer_2020 = arxiv_reference("https://arxiv.org/pdf/2002.05202.pdf", organization="Google", notes=join(
    "Experiments with different activation functions",
    "Activation functions: ReLU, GeLU, Swish",
    "Apply idea of gated units (GLU): ReGLU, GeGLU, SwiGLU",
    "FFN-SwiGLU = Swish(x W1) * xV W2",
    "Have 3 matrices now, so make hidden dimension 2/3 of the 2 matrix version",
))

longformer = arxiv_reference("https://arxiv.org/pdf/2004.05150.pdf", organization="AllenAI", notes=join(
    "Sliding window (local) attention",
    "Global attention to capture task-specific information",
))

sparse_transformer = arxiv_reference("https://arxiv.org/pdf/1904.10509.pdf", organization="OpenAI", notes=join(
    "Local attention"
))

megatron_parallelism_2021 = arxiv_reference("https://arxiv.org/pdf/2104.04473.pdf", organization="NVIDIA", notes=join(
    "Compose tensor, pipeline, data parallelism",
    "Achieve 52% MFU on 1T parameter model on 3072 GPUs",
))

gpt_3 = arxiv_reference("https://arxiv.org/pdf/2005.14165.pdf", organization="OpenAI", notes=join(
    "Introduces GPT-3",
    "Same as GPT-2, but alternating sparse and dense attention layers",
    "175B parameters",
    "Data: 300B tokens",
))

mmlu = arxiv_reference("https://arxiv.org/pdf/2009.03300.pdf", organization="Berkeley", notes=join(
    "57 subjects, multiple-choice",
))

the_pile = arxiv_reference("https://arxiv.org/pdf/2101.00027.pdf", organization="EleutherAI", notes=join(
    "825GB text, 22 diverse subsets (CommonCrawl, PubMed, ArXiv, GitHub, StackExchange, USPTO, OpenWebText2, Books3, etc.)",
))

pre_post_norm_2020 = arxiv_reference("https://arxiv.org/pdf/2002.04745.pdf")

############################################################
# 2021

rope_2021 = arxiv_reference("https://arxiv.org/pdf/2104.09864.pdf", notes=join(
    "Encodes absolute position with rotation matrix, incorporate relative position dependency in self-attention",
    "Key: R W x, where R is a block-diagonal sequence of d/2 rotation matrices (equation 13)",
    "Extrapolates to longer sequences",
))

gpt_j = Reference(
    title="GPT-J", organization="EleutherAI", date="2021-06-04",
    authors=["Ben Wang", "Aran Komatsuzaki"],
    url="https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/",
    notes=join(
        "6.7B parameters",
        "Attention and feedforward layers put in parallel",
        "v3 256 TPUs (5.4 PFLOPs) for 5 weeks",
    ),
)

gopher = arxiv_reference("https://arxiv.org/pdf/2112.11446.pdf", organization="DeepMind", notes=join(
    "280B parameters",
    "Data: 300B tokens",
))

############################################################
# 2022

instruct_gpt = arxiv_reference("https://arxiv.org/pdf/2203.02155.pdf", organization="OpenAI", notes=join(
    "Training language models to follow instructions with human feedback",
))

chinchilla = arxiv_reference("https://arxiv.org/pdf/2203.15556.pdf", organization="DeepMind", notes=join(
    "Introduced the rigorous analysis scaling laws for language models",
    "Key improvement over Kaplan: tune learning rate for the compute budget",
    "Approach 1: for each model size, train with 4 learning rates, vary number of training tokens, fit lower envelope",
    "Approach 2 (IsoFLOP): for each model size, train with 9 training budgets, take last point",
    "Approach 3: fit parametric function L(N, D) = E + A/N^alpha + B/D^beta to data collected from approaches 1 and 2",
    "Conclusion: model and data should scale up at same rate",
    "Table 3: extrapolate to 10 trillion parameters",
    "MassiveText, different data distribution (1.5 trillion tokens)",
    "70B parameters",
))

palm = arxiv_reference("https://arxiv.org/pdf/2204.02311.pdf", organization="Google", notes=join(
    "Data: Social media conversations, webpages, books, GitHub, Wikipedia, news",
    "540B parameters",
    "SwiGLU, parallelize attention and feedforward layers, multi-query attention, RoPE, remove biases",
    "hardware: 6144 TPUv4, 46.2% MFU",
    "optimizer: Adafactor without factorization",
    "Introduced the term model FLOPs utilization (MFU) metric (observed tokens/sec / theoretical max tokens/sec)",
))

gpt_neox = arxiv_reference("https://arxiv.org/pdf/2204.06745.pdf", organization="EleutherAI", notes=join(
    "Data: The Pile",
    "20B parameters",
    "Use RoPE, parallel attention and feedforward layers (15% throughput increase)",
    "hardware: 12x8 A100s",
))

opt_175b = arxiv_reference("https://arxiv.org/pdf/2205.01068.pdf", organization="Meta", notes=join(
    "Data: The Pile, PushShift.io Reddit, deduplication",
    "175B parameters",
    "hardware: 992 A100 80GB for 2 months, lots of hardware failures",
    "FSDP with Megatron-LM, fp16 with loss scaling",
))

bloom = arxiv_reference("https://arxiv.org/abs/2211.05100", organization="BigScience", notes=join(
    "Model: BLOOM (176B parameters)",
    "Data: ROOTS",
    "Hardware: 48x8 A100s on Jean Zay supercomputer for 3.5 months",
    "ZeRO stage 1",
))

############################################################
# 2023

llama = arxiv_reference("https://arxiv.org/pdf/2302.13971.pdf", organization="Meta", notes=join(
    "Train only on open data (detailed recipe that is replicated by RedPajama)",
    "Optimize for fast inference at 7B",
    "Data: CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange",
    "Architecture: Pre-norm, SwiGLU, RoPE",
    "Training: 2048 A100 80GB for 21 days",
))

gpt4 = arxiv_reference("https://arxiv.org/pdf/2303.08774.pdf", organization="OpenAI", notes=join(
    "No details on the data or model architecture.",
))

gqa = arxiv_reference("https://arxiv.org/pdf/2305.13245.pdf", organization="Google", notes=join(
    "Multi-query attention (MQA) speeds up, but less expressive",
    "GQA: use an intermediate (more than one, less than number of heads) number of key-value heads",
    "Experiments on T5",
))

lima = arxiv_reference("https://arxiv.org/pdf/2305.11206.pdf")

dpo = arxiv_reference("https://arxiv.org/pdf/2305.18290.pdf")

llama2 = arxiv_reference("https://arxiv.org/pdf/2307.09288.pdf", organization="Meta", notes=join(
    "2T tokens"
    "70B parameters",
))

llama3 = arxiv_reference("https://arxiv.org/abs/2407.21783", organization="Meta", notes=join(
    "15T tokens",
    "405B parameters",
))

mistral_7b = arxiv_reference("https://arxiv.org/pdf/2310.06825.pdf", organization="Mistral", notes=join(
    "GQA, sliding window attention",
))

############################################################
# 2024

deepseek_67b = arxiv_reference("https://arxiv.org/pdf/2401.02954.pdf", organization="DeepSeek", notes=join(
    "Data: DeepSeek, The Stack, Reddit, etc. (2T tokens)",
    "Architecture: LLaMA, but: for GQA increased depth, 67B parameters",
    "Scaling laws: used non-embedding FLOPs with IsoFLOP",
))

mixtral = arxiv_reference("https://arxiv.org/pdf/2401.04088.pdf", organization="Mistral")

olmo_7b = arxiv_reference("https://arxiv.org/pdf/2402.00838.pdf", organization="AI2", notes=join(
    "Data: subset of Dolma (2.46T tokens, CommonCrawl, The Stack, Reddit, etc.)",
    "Architecture: no biases, non-parametric layer norm, SwiGLU (8/3 d increased to closest multiple of 128)",
    "Training: 256x4 AMD MI250X on LUMI supercomputer, 27x8 A100s, 800Gbps interconnect",
))

megascale = arxiv_reference("https://arxiv.org/pdf/2402.15627.pdf", organization="Bytedance", notes=join(
    "55.2% MFU for 175B parameter model over 12,288 GPUs",
    "Combine data, tensor, pipeline, sequence parallelism",
    "Parallelize attention and feedforward layers, sliding window attention, LAMB optimizer",
))

nemotron_15b = arxiv_reference("https://arxiv.org/pdf/2402.16819.pdf", organization="NVIDIA", notes=join(
    "Data: 8T tokens, 70% English, 15% multilingual, 15% code",
    "Architecture: RoPE, squared ReLU activations, no bias, no dropout, GQA (15B parameters)",
    "Training: 384x8 H100s, After 8T tokens, train on higher quality sources + benchmark tasks",
))

yi_34b = arxiv_reference("https://arxiv.org/pdf/2403.04652.pdf", organization="01.AI")

gemma = arxiv_reference("https://arxiv.org/pdf/2403.08295.pdf", organization="Google DeepMind", notes=join(
    "6T tokens",
    "MQA, RoPE, GeGLU, RMSNorm",
    "Training: 4096 v4 TPUs, Use ZeRO-3 like techniques",
))

overtrained_scaling_laws = arxiv_reference("https://arxiv.org/pdf/2403.08540.pdf", notes=join(
    "Chinchilla scaling laws focus on loss of the trained model, ignoring inference costs.",
    "Constant ratio of training tokens to parameters",
    "Extrapolate over 300x training compute to 1.4B model on 900B tokens",
    "Look at task performance rather than validation loss",
))

transformer_math = Reference(
    title="Transformer Math 101", organization="EleutherAI", date="2023-04-03",
    url="https://blog.eleuther.ai/transformer-math/",
)

bahdanau_training_costs = Reference(
    title="The FLOPs Calculus of Language Model Training", authors=["Dzmitry Bahdanau"], date="2022-01-09",
    url="https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4",
)

deepseek_v2 = mla = arxiv_reference("https://arxiv.org/abs/2405.04434")
deepseek_v3 = arxiv_reference("https://arxiv.org/pdf/2412.19437.pdf")
qwen_2_5 = arxiv_reference("https://arxiv.org/abs/2412.15115", organization="Alibaba")
minicpm = wsd_2024 = arxiv_reference("https://arxiv.org/pdf/2404.06395.pdf", organization="Tsinghua")
huanyuan_large = arxiv_reference("https://arxiv.org/abs/2411.02265", organization="Tencent")

muon = Reference(
    title="Muon: An optimizer for hidden layers in neural networks", date="2024-12-08",
    authors=["Jordan Keller"],
    url="https://kellerjordan.github.io/posts/muon/"
)
soap = arxiv_reference("https://arxiv.org/abs/2409.11321")

smollm2 = arxiv_reference("https://arxiv.org/pdf/2502.02737.pdf", notes=join(
    "1.7B parameter model"
    "Introduces FineMath, StackEdu",
))

############################################################
# 2025

olmo2 = arxiv_reference("https://arxiv.org/abs/2501.00656", organization="AI2")
dclm_2024 = arxiv_reference("https://arxiv.org/abs/2406.11794")
nemotron_cc_2024 = arxiv_reference("https://arxiv.org/abs/2412.02595")

deepseek_r1 = arxiv_reference("https://arxiv.org/pdf/2501.12948.pdf")

deepseek_math = grpo = arxiv_reference("https://arxiv.org/pdf/2402.03300.pdf")

kimi_1_5 = arxiv_reference("https://arxiv.org/pdf/2501.12599.pdf")

llama4 = Reference(title="Llama 4", organization="Meta", url="https://ai.meta.com/blog/llama-4-multimodal-intelligence/")
olmo2_32b = Reference(title="OLMo 2 (32B)", organization="AI2", url="https://allenai.org/blog/olmo2-32B")

byt5 = arxiv_reference("https://arxiv.org/abs/2105.13626")
megabyte = arxiv_reference("https://arxiv.org/pdf/2305.07185.pdf")
blt = arxiv_reference("https://arxiv.org/abs/2412.09871")
tfree = arxiv_reference("https://arxiv.org/abs/2406.19223")

sennrich_2016 = arxiv_reference("https://arxiv.org/abs/1508.07909")

openwebtext = Reference(title="OpenWebText", authors=["Aaron Gokaslan", "Vanya Cohen"], date="2019", url="https://skylion007.github.io/OpenWebTextCorpus/")
    
alpaca = Reference(title="Alpaca", authors=["Rohan Taori", "Ishaan Gulrajani", "Tianyi Zhang", "Yann Dubois", "Xuechen Li", "Carlos Guestrin", "Percy Liang", "Tatsunori B. Hashimoto"], date="2023-03-13", url="https://crfm.stanford.edu/2023/03/13/alpaca.html")

dolma = arxiv_reference("https://arxiv.org/abs/2402.00159")

qwen3 = arxiv_reference("https://arxiv.org/abs/2505.09388")