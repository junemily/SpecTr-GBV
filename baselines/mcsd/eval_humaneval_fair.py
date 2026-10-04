"""HumanEval throughput adapter for an external MCSD implementation."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.generate import SpeculativeGenerator


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_humaneval(path: Path, max_prompts: int):
    prompts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"].strip())
    return prompts if max_prompts < 0 else prompts[:max_prompts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair HumanEval adapter for MCSD.")
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "humaneval.jsonl")
    parser.add_argument("--k-config", required=True, type=lambda x: tuple(map(int, x.split(","))))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--naive-sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = torch.float16 if args.fp16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.draft_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": args.device},
        "trust_remote_code": args.trust_remote_code,
    }
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, **model_kwargs).eval()
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model, **model_kwargs).eval()

    generator = SpeculativeGenerator(
        draft_model,
        target_model,
        eos_token_id=tokenizer.eos_token_id,
        k_config=args.k_config,
        max_new_tokens=args.max_new_tokens,
        draft_model_temp=args.temperature,
        target_model_temp=args.temperature,
        replacement=args.replacement,
        speculative_sampling=not args.naive_sampling,
        tree_attn=False,
    )

    prompts = load_humaneval(args.data_path, args.max_prompts)
    total_generated = 0
    total_acceptance = 0
    total_draft_tokens = 0
    total_invocations = 0

    # Warm up on the first prompt, then time generation only on the rest.
    runtime = 0.0
    with torch.no_grad():
        if prompts:
            inputs = tokenizer(prompts[0], return_tensors="pt").to(args.device)
            generator.generate(inputs.input_ids)

        for prompt in tqdm(prompts[1:], desc="MCSD"):
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            initial_len = inputs.input_ids.shape[-1]
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = generator.generate(inputs.input_ids)
            torch.cuda.synchronize()
            runtime += time.perf_counter() - start
            total_generated += min(output.sequences.shape[-1] - initial_len, args.max_new_tokens)
            total_acceptance += int(output.acceptance_count)
            total_draft_tokens += int(output.draft_token_count)
            total_invocations += int(output.invocation_count)

    print(
        "FAIR_METRICS "
        + json.dumps(
            {
                "method": "mcsd",
                "draft_model": args.draft_model,
                "target_model": args.target_model,
                "temperature": args.temperature,
                "k_config": list(args.k_config),
                "depth": len(args.k_config),
                "acceptance_rate": (
                    total_acceptance / total_draft_tokens if total_draft_tokens else 0.0
                ),
                "block_efficiency": (
                    total_generated / total_invocations if total_invocations else 0.0
                ),
                "tokens_per_second": total_generated / runtime if runtime else 0.0,
                "generated_tokens": total_generated,
                "target_steps": total_invocations,
                "runtime": runtime,
                "max_prompts": len(prompts),
                "seed": args.seed,
            }
        )
    )


if __name__ == "__main__":
    main()
