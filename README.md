<div align="center">

# Toward Scientific Reasoning in LLMs: Training from Expert Discussions via Reinforcement Learning

<div>
🧬 A New Benchmark Genome-Bench and RL Fine-Tuning for Scientific Reasoning 📊
</div>
</div>

<div>
<br>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2301.00001-red?style=for-the-badge&logo=arxiv&logoColor=auto)](https://arxiv.org/abs/your_arxiv_id)
[![GitHub](https://img.shields.io/badge/GitHub-Code-000000?style=for-the-badge&logo=github&logoColor=auto)](https://github.com/mingyin0312/RL4GenomeBench)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-ffcc00?style=for-the-badge&logo=huggingface&logoColor=auto)](https://huggingface.co/datasets/Mingyin0312/Genome-Bench)

</div>
</div>

## Overview

![](figure/overview.png)

We introduce **Genome-Bench**, a novel benchmark for evaluating and improving scientific reasoning in large language models. Genome-Bench consists of over 3,000 multiple-choice and QA items derived from CRISPR-related scientific discussions and forum threads, covering key topics in genome engineering, experimental design, and error analysis.

Our RL training pipeline (based on Group Relative Policy Optimization) improves model performance across expert-labeled evaluation sets. For example, our fine-tuned Qwen2.5-7B model exceeds GPT-4o in accuracy and consistency on multi-hop reasoning tasks.

![](figure/example_result.png)

---

## Getting Started 🎯

### Installation

```bash
git clone https://github.com/mingyin0312/RL4GenomeBench.git
cd RL4GenomeBench
pip install -r requirements.txt
```


### Dataset Preparation

We provide tools to parse .mbox email archives and convert them into standardized MCQ and QA formats.

```bash
cd dataset_pipeline
python 1_email_parse.py
python 2_convert_MCQ_full.py
python 3_dataset_prepare.py
python 4_convert_natural_question.py
```

## Training 

### Reinforcement Fine-tuning (GRPO)

```bash
python training/rl_training.py 
```

### Supervised Fine-Tuning (SFT)

```bash
python training/sft_training.py 
```

### Multi-Agent RL Routing

```bash
python training/rl_router_training.py 
```

## Evaluation 

To evaluate on the Genome-Bench test data: 

```bash
python evaluation/genome-bench_eval.py 
```

## Citation

```bibtex
@article{yin2025<<TBD>>,
  title={Toward Scientific Reasoning in LLMs: Training from Expert Discussions via Reinforcement Learning},
  author={Yin, Ming and Qu, Yuanhao and Ling, Yang and Cong, Le and Wang Mengdi},
  journal={arXiv preprint arXiv:<<TBD>>},
  year={2025}
}
```

## Acknowledgement

This project leverages the 🧠 [Transformers Reinforcement Learning (TRL)](https://github.com/huggingface/trl) library developed by Hugging Face, which provides powerful tools for fine-tuning large language models with reinforcement learning techniques such as PPO, DPO, and GRPO.

