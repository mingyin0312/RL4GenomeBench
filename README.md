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
git clone https://github.com/your_org/Genome-Bench.git
cd Genome-Bench
pip install -r requirements.txt
