import time
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .kvcache import KVCacheModel
from .util import seed_everything, sample


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        seed_everything(args.seed)
        self.load_tokenizer()
        self.load_model()

    def load_model(self) -> None:
        dtype = DTYPES[self.args.dtype]
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": {"": self.args.device},
            "trust_remote_code": self.args.trust_remote_code,
        }
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.args.draft_model, **model_kwargs
        ).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.args.target_model, **model_kwargs
        ).eval()

        draft_vocab = self.draft_model.get_output_embeddings().weight.shape[0]
        target_vocab = self.target_model.get_output_embeddings().weight.shape[0]
        if draft_vocab != target_vocab:
            raise ValueError(
                "Draft and target models must use the same output vocabulary size "
                f"({draft_vocab} != {target_vocab})."
            )
        self.vocab_size = draft_vocab

    def load_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.draft_model,
            trust_remote_code=self.args.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self, input_text):
        pass

    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass

    @torch.no_grad()
    def speculative_decoding(self, prefix: torch.Tensor, num_candidates: int):
        initial_token_count = prefix.shape[1]
        K = num_candidates
        gamma = self.args.gamma
        max_tokens = self.args.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device
        prefix = prefix.long().to(draft_device)

        draft_cache = KVCacheModel(
            self.draft_model,
            self.vocab_size,
            self.args.temp,
            self.args.top_k,
            self.args.top_p,
        )
        target_cache = KVCacheModel(
            self.target_model,
            self.vocab_size,
            self.args.temp,
            self.args.top_k,
            self.args.top_p,
        )

        eps = torch.tensor(1e-20, device=draft_device)
        one = torch.tensor(1.0, device=draft_device)
        k_tensor = torch.tensor(K, device=draft_device)

        def exp_term(x):
            return torch.pow(one - x, k_tensor)

        total_iterations = 0
        total_accepted_tokens = 0
        total_candidate_draft_tokens = 0

        if draft_device.type == "cuda":
            torch.cuda.synchronize(draft_device)
        wall_start = time.perf_counter()

        while prefix.shape[1] < initial_token_count + max_tokens:
            total_iterations += 1
            prefix_len = prefix.shape[1]
            input_ids = prefix.repeat(K, 1)

            x = draft_cache.generate(input_ids, gamma)
            target_cache.generate(x.to(target_device), 1)
            target_probs = target_cache.prob_history.to(draft_device)

            n = prefix_len - 1
            accepted_sequence = x.new_empty(0)
            selected_candidate = 0
            seen_prefixes = set()
            ms_final = torch.ones(1, device=draft_device)
            mb_final = torch.ones(1, device=draft_device)

            for j in range(K):
                probs_s_j = draft_cache.prob_history[j, prefix_len - 1 : prefix_len + gamma]
                probs_b_j = target_probs[j, prefix_len - 1 : prefix_len + gamma]

                m_s = torch.ones(1, device=draft_device)
                m_b = torch.ones(1, device=draft_device)

                for k in range(n - prefix_len + 1):
                    token = x[j, prefix_len + k]
                    m_s = m_s * draft_cache.prob_history[j, prefix_len + k - 1, token]
                    m_b = m_b * target_probs[j, prefix_len + k - 1, token]

                for i in range(n - prefix_len + 1, gamma - 1):
                    token = x[j, prefix_len + i]
                    prob_b = probs_b_j[i]
                    prob_s = probs_s_j[i]
                    next_prob_b = probs_b_j[i + 1]
                    next_prob_s = probs_s_j[i + 1]

                    m_s = m_s * prob_s[token]
                    m_b = m_b * prob_b[token]

                    ratio = (m_s * next_prob_s) / (m_b * next_prob_b + eps)
                    min_ratio = torch.minimum(ratio, one)
                    raw_ratio = torch.minimum(m_s / (m_b + eps), one)

                    numerator = (
                        torch.sum(m_b * next_prob_b * exp_term(min_ratio))
                        - m_b * exp_term(raw_ratio)
                    )
                    denominator = (
                        one
                        - exp_term(m_s)
                        - torch.sum(m_b * next_prob_b * (one - exp_term(min_ratio)))
                    )
                    acceptance_prob = numerator / (denominator + eps)

                    draw = torch.rand(1, device=draft_device)
                    candidate_prefix = x[j, prefix_len : prefix_len + i + 1]
                    candidate_key = tuple(candidate_prefix.tolist())
                    if candidate_key in seen_prefixes:
                        continue

                    if draw < acceptance_prob:
                        accepted_sequence = candidate_prefix
                        n = prefix_len + i
                        selected_candidate = j
                        ms_final = m_s
                        mb_final = m_b
                    else:
                        seen_prefixes.add(candidate_key)

                final_sequence = x[j, prefix_len : prefix_len + gamma]
                final_token = x[j, prefix_len + gamma - 1]
                m_s = m_s * probs_s_j[gamma - 1, final_token]
                m_b = m_b * probs_b_j[gamma - 1, final_token]

                ratio = torch.minimum(m_s / (m_b + eps), one)
                acceptance_prob = (m_b * (1 - exp_term(ratio))) / (1 - exp_term(m_s) + eps)
                if torch.rand(1, device=draft_device) < acceptance_prob:
                    accepted_sequence = final_sequence
                    n = prefix_len + gamma - 1
                    selected_candidate = j
                    ms_final = m_s
                    mb_final = m_b
                    break

            accepted_tokens = n - prefix_len + 1
            total_accepted_tokens += accepted_tokens
            total_candidate_draft_tokens += K * gamma

            if accepted_sequence.numel() > 0:
                prefix = torch.cat((prefix, accepted_sequence.unsqueeze(0)), dim=1)
                target_cache.select_candidate(selected_candidate)
                draft_cache.select_candidate(selected_candidate)

            if n < prefix_len + gamma - 1:
                prob_b = target_cache.prob_history.to(draft_device)[selected_candidate, n]
                prob_s = draft_cache.prob_history[selected_candidate, n]
                residual = mb_final * prob_b * (
                    one
                    - torch.minimum(
                        ms_final * prob_s / (mb_final * prob_b + eps),
                        one,
                    )
                    ** k_tensor
                )
                residual = residual / residual.sum().clamp_min(eps)
                next_token = sample(residual).to(draft_device).unsqueeze(1)
                prefix = torch.cat((prefix, next_token), dim=1).long()
                target_cache.rollback(n + 1)
                draft_cache.rollback(n + 1)
            else:
                next_token = sample(
                    target_cache.prob_history[selected_candidate, -1, : self.vocab_size]
                ).to(draft_device).unsqueeze(1)
                prefix = torch.cat((prefix, next_token), dim=1).long()
                target_cache.rollback(n + 2)
                draft_cache.rollback(n + 1)

        if draft_device.type == "cuda":
            torch.cuda.synchronize(draft_device)
        runtime = time.perf_counter() - wall_start

        generated_tokens = prefix.shape[1] - initial_token_count
        strict_generated_tokens = min(generated_tokens, max_tokens)
        metrics = {
            "method": "spectr-gbv" if K > 1 else "gbv",
            "depth": gamma,
            "num_candidates": K,
            "generated_tokens": generated_tokens,
            "strict_generated_tokens": strict_generated_tokens,
            "target_steps": total_iterations,
            "accepted_tokens": total_accepted_tokens,
            "candidate_draft_tokens": total_candidate_draft_tokens,
            "acceptance_rate": (
                total_accepted_tokens / total_candidate_draft_tokens
                if total_candidate_draft_tokens
                else 0.0
            ),
            "block_efficiency": (
                strict_generated_tokens / total_iterations if total_iterations else 0.0
            ),
            "tokens_per_second": strict_generated_tokens / runtime if runtime else 0.0,
            "runtime": runtime,
        }

        prefix = prefix[:, : initial_token_count + max_tokens]
        return prefix, metrics
