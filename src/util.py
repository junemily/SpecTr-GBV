import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def positive_int(value: str) -> int:
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate SpecTr-GBV on HumanEval.")
    parser.add_argument("--draft_model", required=True, help="Hugging Face model ID or local path.")
    parser.add_argument("--target_model", required=True, help="Hugging Face model ID or local path.")
    parser.add_argument("--data_path", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--exp_name", "-e", default="spectrgbv")
    parser.add_argument("--num_samples_per_task", "-n", type=positive_int, default=1)
    parser.add_argument("--seed", "-s", type=int, default=1001)
    parser.add_argument("--max_tokens", type=positive_int, default=64)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--gamma", type=positive_int, default=6, help="Draft length L.")
    parser.add_argument(
        "--num_candidates",
        "-K",
        type=positive_int,
        default=7,
        help="Number of independent draft candidates. K=1 gives GBV.",
    )
    parser.add_argument("--max_prompts", type=int, default=32, help="Use -1 for all prompts.")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    if args.gamma < 2:
        parser.error("--gamma must be at least 2")
    if args.top_k < 0:
        parser.error("--top_k must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top_p must lie in (0, 1]")
    if args.temp < 0:
        parser.error("--temp must be non-negative")

    args.output_dir = args.output_dir / args.exp_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Apply top-k and nucleus filtering to a 2-D logits tensor."""
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = torch.zeros_like(sorted_mask).scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    return logits


def norm_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Convert logits of shape ``(batch, vocab)`` to sampling probabilities."""
    if logits.dim() != 2:
        raise ValueError(f"Expected a 2-D logits tensor, got shape {tuple(logits.shape)}")

    if temperature == 0:
        indices = logits.argmax(dim=-1, keepdim=True)
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(1, indices, 1.0)
        return probs

    logits = top_k_top_p_filter(logits / temperature, top_k=top_k, top_p=top_p)
    return F.softmax(logits, dim=-1)


def sample(probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=num_samples)
