import torch

from .util import norm_logits, sample


class KVCacheModel:
    """Minimal Hugging Face DynamicCache wrapper used by SpecTr-GBV."""

    def __init__(
        self,
        model: torch.nn.Module,
        vocab_size: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> None:
        self.model = model
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.past_key_values = None
        self.prob_history = None

    def _normalize(self, logits: torch.Tensor) -> torch.Tensor:
        shape = logits.shape
        logits = logits[..., : self.vocab_size].reshape(-1, self.vocab_size)
        probs = norm_logits(logits, self.temperature, self.top_k, self.top_p)
        return probs.reshape(*shape[:-1], self.vocab_size)

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.past_key_values is None:
            outputs = self.model(input_ids, use_cache=True)
            self.prob_history = self._normalize(outputs.logits)
        else:
            cached_len = self.past_key_values.get_seq_length()
            new_input_ids = input_ids[:, cached_len:]
            outputs = self.model(
                new_input_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
            new_probs = self._normalize(outputs.logits)
            self.prob_history = torch.cat((self.prob_history, new_probs), dim=1)

        self.past_key_values = outputs.past_key_values
        return self.prob_history[:, -1, :]

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, gamma: int) -> torch.Tensor:
        batch_size, prefix_len = prefix.shape
        output = torch.empty(
            (batch_size, prefix_len + gamma),
            dtype=prefix.dtype,
            device=prefix.device,
        )
        output[:, :prefix_len] = prefix

        cur_len = prefix_len
        for _ in range(gamma):
            probs = self._forward_with_kvcache(output[:, :cur_len])
            output[:, cur_len : cur_len + 1] = sample(probs).to(output.device)
            cur_len += 1
        return output

    @torch.no_grad()
    def rollback(self, end_pos: int) -> None:
        self.past_key_values.crop(end_pos)
        self.prob_history = self.prob_history[:, :end_pos, :]

    @torch.no_grad()
    def select_candidate(self, index: int) -> None:
        """Keep one candidate branch and repeat it to restore the batch size."""
        batch_size = self.prob_history.shape[0]
        indices = torch.tensor([index], device=self.prob_history.device)

        self.prob_history = self.prob_history[index : index + 1].repeat(batch_size, 1, 1)
        self.past_key_values.batch_select_indices(indices)
        self.past_key_values.batch_repeat_interleave(batch_size)
