import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import Decoding
from src.util import parse_arguments, seed_everything


class EvalHumanEval(Decoding):
    def __init__(self, args):
        super().__init__(args)
        self.load_data()

    def load_data(self) -> None:
        data_file = self.args.data_path / "humaneval.jsonl"
        data = []
        with data_file.open(encoding="utf-8") as f:
            for line in f:
                datum = json.loads(line)
                datum["input_text"] = self.preprocess(datum["prompt"])
                input_ids = self.tokenizer.encode(
                    datum["input_text"],
                    add_special_tokens=True,
                )
                datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
                data.append(datum)

        if self.args.max_prompts > 0:
            data = data[: self.args.max_prompts]
        self.data = data

    def preprocess(self, input_text: str) -> str:
        return input_text.strip()

    def postprocess(self, input_text: str, output_text: str) -> str:
        if self.tokenizer.bos_token and output_text.startswith(self.tokenizer.bos_token):
            generation = output_text[len(input_text) + len(self.tokenizer.bos_token) + 1 :]
        else:
            generation = output_text[len(input_text) :]

        stop_words = [
            "\nclass",
            "\ndef",
            "\n#",
            "\n@",
            "\nprint",
            "\nif",
            "\n```",
            self.tokenizer.eos_token,
        ]
        for stop_word in filter(None, stop_words):
            if stop_word in generation:
                generation = generation[: generation.index(stop_word)].strip()
        return (input_text + "\n    " + generation).replace("\t", "    ")

    @torch.no_grad()
    def eval(self) -> None:
        output_file = self.args.output_dir / "humaneval.jsonl"
        aggregate = {
            "runtime": 0.0,
            "strict_generated_tokens": 0,
            "raw_generated_tokens": 0,
            "target_steps": 0,
            "accepted_tokens": 0,
            "candidate_draft_tokens": 0,
            "num_prompts": 0,
        }

        with output_file.open("w", encoding="utf-8") as out_f:
            for sample_idx in range(self.args.num_samples_per_task):
                sample_seed = self.args.seed + sample_idx
                seed_everything(sample_seed)

                for prompt_idx, datum in enumerate(tqdm(self.data, desc=f"seed={sample_seed}")):
                    input_ids = datum["input_ids"]
                    generated_ids, metrics = self.speculative_decoding(
                        input_ids,
                        self.args.num_candidates,
                    )

                    output = self.postprocess(
                        datum["input_text"],
                        self.tokenizer.decode(generated_ids[0]),
                    )
                    out_f.write(
                        json.dumps(
                            {
                                "task_id": datum["task_id"],
                                "seed": sample_seed,
                                "runtime": metrics["runtime"],
                                "new_tokens": metrics["strict_generated_tokens"],
                                "completion": output,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    # The first prompt is a warm-up and is excluded from throughput.
                    if prompt_idx == 0:
                        continue
                    aggregate["runtime"] += metrics["runtime"]
                    aggregate["strict_generated_tokens"] += metrics["strict_generated_tokens"]
                    aggregate["raw_generated_tokens"] += metrics["generated_tokens"]
                    aggregate["target_steps"] += metrics["target_steps"]
                    aggregate["accepted_tokens"] += metrics["accepted_tokens"]
                    aggregate["candidate_draft_tokens"] += metrics["candidate_draft_tokens"]
                    aggregate["num_prompts"] += 1

        runtime = aggregate["runtime"]
        generated = aggregate["strict_generated_tokens"]
        raw_generated = aggregate["raw_generated_tokens"]
        target_steps = aggregate["target_steps"]
        candidate_tokens = aggregate["candidate_draft_tokens"]
        result = {
            "method": "spectr-gbv" if self.args.num_candidates > 1 else "gbv",
            "draft_model": self.args.draft_model,
            "target_model": self.args.target_model,
            "temperature": self.args.temp,
            "depth": self.args.gamma,
            "num_candidates": self.args.num_candidates,
            "max_tokens": self.args.max_tokens,
            "max_prompts": len(self.data),
            "seed": self.args.seed,
            "num_samples_per_task": self.args.num_samples_per_task,
            "evaluated_prompts": aggregate["num_prompts"],
            "tokens_per_second": generated / runtime if runtime else 0.0,
            "block_efficiency": raw_generated / target_steps if target_steps else 0.0,
            "acceptance_rate": (
                aggregate["accepted_tokens"] / candidate_tokens if candidate_tokens else 0.0
            ),
            "generated_tokens": generated,
            "target_steps": target_steps,
            "runtime": runtime,
        }
        print("FAIR_METRICS " + json.dumps(result))


if __name__ == "__main__":
    EvalHumanEval(parse_arguments()).eval()
