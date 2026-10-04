# SpecTr-GBV

Official implementation of **SpecTr-GBV: Multi-Draft Block Verification Accelerating Speculative Decoding**.

SpecTr-GBV samples multiple independent draft continuations and verifies them jointly with the target model. Setting `K=1` recovers the single-candidate GBV baseline.

## Repository structure

```text
src/                         SpecTr-GBV decoding implementation
benchmark/eval_humaneval.py  HumanEval throughput benchmark
data/humaneval.jsonl         HumanEval prompts
baselines/mcsd/              MCSD evaluation adapter
scripts/run_humaneval.sh     Three-seed SpecTr-GBV / GBV benchmark
scripts/summarize_results.py Aggregate FAIR_METRICS logs
```

## Installation

Python 3.10 and a CUDA-capable GPU are recommended.

```bash
pip install -r requirements.txt
```

If your CUDA setup requires a different PyTorch wheel, install the matching PyTorch build first and then install the remaining dependencies.

The release is pinned to `transformers==4.45.2`, whose `DynamicCache` API is used directly by the decoder.

## Models

`--draft_model` and `--target_model` accept either Hugging Face model IDs or local model paths. The two models must use the same token vocabulary.

For example, the 160M draft model is publicly available as `JackFram/llama-160m`. Llama-2 target models require the corresponding Hugging Face access permission.

## HumanEval

### SpecTr-GBV

```bash
python benchmark/eval_humaneval.py \
  --draft_model JackFram/llama-160m \
  --target_model meta-llama/Llama-2-7b-hf \
  --num_candidates 7 --gamma 6 \
  --max_tokens 64 --max_prompts 32 \
  --temp 0.7 --top_p 1.0 --dtype float16 \
  --seed 1001 -e spectrgbv
```

### GBV

Use the same command with one candidate:

```bash
python benchmark/eval_humaneval.py \
  --draft_model JackFram/llama-160m \
  --target_model meta-llama/Llama-2-7b-hf \
  --num_candidates 1 --gamma 6 \
  --max_tokens 64 --max_prompts 32 \
  --temp 0.7 --top_p 1.0 --dtype float16 \
  --seed 1001 -e gbv
```

The benchmark treats the first HumanEval prompt as a warm-up and excludes it from throughput statistics. Each run prints one machine-readable line beginning with `FAIR_METRICS`.

### Three-seed comparison

```bash
DRAFT_MODEL=JackFram/llama-160m \
TARGET_MODEL=meta-llama/Llama-2-7b-hf \
bash scripts/run_humaneval.sh
```

By default the script uses seeds `1001 1002 1003`, runs SpecTr-GBV (`K=7, L=6`) and GBV (`K=1, L=6`), and reports mean ± standard deviation for throughput, block efficiency, and acceptance rate.

## MCSD baseline

The MCSD baseline is not vendored in this repository. The adapter in `baselines/mcsd/eval_humaneval_fair.py` expects the external MCSD environment used in the experiments to expose `inference.generate.SpeculativeGenerator`. The original MCSD project is available at [NJUNLP/MCSD](https://github.com/NJUNLP/MCSD). Add a compatible MCSD implementation to `PYTHONPATH`, then run:

```bash
python baselines/mcsd/eval_humaneval_fair.py \
  --draft-model JackFram/llama-160m \
  --target-model meta-llama/Llama-2-7b-hf \
  --k-config 4,2,2 \
  --max-new-tokens 64 --max-prompts 32 \
  --temperature 0.7 --fp16 --seed 1001
```

## Reported HumanEval results

The following values are the reported results for the first 32 HumanEval prompts with `max_new_tokens=64`, temperature `0.7`, `top_p=1.0`, fp16, and a single H20 GPU. Throughput is hardware dependent.

| Method | Config | 68M → 7B | 68M → 13B | 160M → 7B | 160M → 13B |
|---|---|---:|---:|---:|---:|
| GBV | K=1, L=6 | 44.16 / 1.53 | 38.28 / 1.64 | 24.88 / 1.60 | 23.41 / 1.68 |
| MCSD | [4,2,2] | 42.76 / 1.62 | 26.27 / 1.61 | 29.97 / 1.64 | 21.75 / 1.71 |
| **SpecTr-GBV** | **K=7, L=6** | **68.77 / 3.17** | **53.05 / 3.28** | **42.68 / 3.54** | **34.93 / 3.36** |

Each entry is **tokens/s / block efficiency**, where block efficiency is generated tokens divided by target-model verification steps.

## Citation

SpecTr-GBV was accepted to the EMNLP 2026 Main Conference. The preprint is available on [arXiv](https://arxiv.org/abs/2604.25925).

```bibtex
@article{lin2026spectrgbv,
  title   = {SpecTr-GBV: Multi-Draft Block Verification Accelerating Speculative Decoding},
  author  = {Lin, Yijun and Sheng, Jinhao and Cai, Qingyue and Zhou, Feng},
  journal = {arXiv preprint arXiv:2604.25925},
  year    = {2026}
}
```
