<div align="center">
  <img src="images/dive-banner-v3.png" alt="DIVE" width="400">
  <h1>🤿 DIVE: Dynamic Iterative Visual Evidence Construction<br>for Efficient Vision-Language Models</h1>
  <a href="https://github.com/Zhong-Chenchen/DIVE"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" alt="GitHub Code"></a>
  <a href="https://arxiv.org/abs/2608.04496"><img src="https://img.shields.io/badge/arXiv-2608.04496-b31b1b?logo=arxiv&logoColor=white" alt="arXiv 2608.04496"></a>
</div>

<p align="center">Welcome to star🌟 this repo or cite✨ the paper if you find it interesting😊</p>

> **TL;DR:** We introduce DIVE, a token pruning method that iteratively construct visual evidence to identify and remove redundant visual tokens.

## ✨ Key Highlights

- ⚡ **Efficient Inference:** Achieves a **1.68× end-to-end speedup** while maintaining near-full performance.
- 🎯 **Extreme Compression:** Retains **98.2%** performance with only **64 visual tokens** (**88.9% reduction**).
- 🔄 **Dynamic Selection:** Selects complementary, prompt-relevant visual evidence through iterative residual updates.
- 🌐 **Broad Generalization:** Consistently improves image and video understanding across diverse VLM architectures.

## 🔥 News

- **`2026.08.01`** 🤗🤗 We have released DIVE, a training-free method for efficient visual-token reduction in vision-language models.
- **`2026.07.29`** 🤗🤗 The DIVE framework, LLaVA evaluation scripts, and Qwen2-VL and LLaVA-OneVision integrations are now available.

## 👀 Overview

DIVE provides a native LLaVA-1.5/1.6 language-model path using the DIVE
selector, together with Qwen2-VL and LLaVA-OneVision integrations.

<p align="center">
  <img src="images/framework.png" alt="DIVE framework" width="1000">
</p>

## 🛠 Preparation

### LLaVA

```bash
conda create -n DIVE python=3.10 -y
conda activate DIVE
pip install -e .
pip install flash-attn --no-build-isolation
```

Download the benchmark assets from their official sources and extract them
under `./playground/data/eval`.

### Qwen2-VL

```bash
conda create -n DIVE_Qwen2VL python=3.10 -y
conda activate DIVE_Qwen2VL
cd othermodels/qwen2_vl/transformers && pip install -e .
pip install accelerate qwen-vl-utils[decord]
pip install flash-attn --no-build-isolation
cd ../../../lmms-eval && pip install -e .
```

## 🎯 Usage

### LLaVA-1.5

```bash
bash scripts/v1_5/eval/[benchmark].sh [reduction_ratio] [max_num_trunction]
```

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh 0.778 128
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh 0.778 128
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh 0.778 128
```

### LLaVA-NeXT

```bash
bash scripts/v1_6/eval/[benchmark].sh [reduction_ratio] [max_num_trunction]
```

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_6/eval/pope.sh 0.778 128
```

### Qwen2-VL

```bash
cd othermodels/qwen2_vl
bash eval_scripts/lmms_eval.sh True [reduction_ratio]
```

## 📌 Citation

Please consider citing [our paper](https://arxiv.org/abs/2608.04496) if you find DIVE useful for your research.

```bibtex
@article{zhong2026dive,
  title={DIVE: Dynamic Iterative Visual Evidence Construction for Efficient Vision-Language Models},
  author={Zhong, Chen and An, Xiao and Wang, Zijie and Li, Jiepan and Yang, Guangyi and He, Wei},
  journal={arXiv preprint arXiv:2608.04496},
  year={2026},
  url={https://arxiv.org/abs/2608.04496}
}
```

## 🔑 License

This project is released under the [Apache License 2.0](LICENSE). Original
upstream copyright and license notices are retained in the source files.

## 👍 Acknowledgment

DIVE thanks the [LLaVA](https://github.com/haotian-liu/llava), [QwenVL](https://github.com/QwenLM/Qwen-VL), [DART](https://github.com/ZichenWen1/DART), and [lmms-eval](https://github.com/evolvinglmms-lab/lmms-eval) communities for their
open-source contributions.
