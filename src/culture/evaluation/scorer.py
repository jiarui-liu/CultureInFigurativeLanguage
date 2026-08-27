"""HuggingFace causal-LM wrapper for base-model evaluation.

Provides two primitives used by the tasks:

- :meth:`HFModel.loglikelihood` -- batched, tokenizer-robust log-probability of a
  continuation given a context. Used for multiple-choice scoring: score every
  option as a continuation and pick the arg-max. This needs no instruction
  following, so it works on the raw (non-instruct) CPT / base checkpoints.
- :meth:`HFModel.generate` -- greedy free-form completion, used for IdiomCE.

The log-likelihood implementation follows the EleutherAI lm-evaluation-harness
convention: it tokenizes ``context`` and ``context + continuation`` separately
and takes the token-count difference, which correctly handles tokenizer merges
at the context/continuation boundary (important for Devanagari).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class LLResult:
    """Log-likelihood of one (context, continuation) request."""

    logprob: float          # summed log-prob of the continuation tokens
    num_tokens: int         # number of continuation tokens scored

    @property
    def logprob_per_token(self) -> float:
        return self.logprob / max(self.num_tokens, 1)


class HFModel:
    """Thin wrapper around a local/HF causal LM for log-likelihood + generation."""

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_length: int = 4096,
        trust_remote_code: bool = True,
        attn_implementation: str = "sdpa",  # NOT fa2: Qwen3.5 s_aux crashes FA2 path
    ):
        self.model_path = model_path
        self.max_length = max_length

        logger.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        # Causal LM log-likelihood batching needs left padding so the scored
        # continuation tokens sit at fixed positions on the right.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model from %s (%s, %s)", model_path, dtype, attn_implementation)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------------ #
    # Log-likelihood (multiple-choice scoring)
    # ------------------------------------------------------------------ #
    def _encode_request(self, context: str, continuation: str) -> Tuple[List[int], int]:
        """Return (token_ids of context+continuation, #continuation tokens).

        Uses the tokenize-diff trick to be robust to boundary merges, then
        left-truncates to ``max_length + 1`` while preserving the (short)
        continuation at the tail.
        """
        ctx_ids = self.tokenizer(context, add_special_tokens=True).input_ids
        full_ids = self.tokenizer(context + continuation, add_special_tokens=True).input_ids
        cont_len = len(full_ids) - len(ctx_ids)
        if cont_len <= 0:
            # Degenerate boundary (continuation merged entirely into context);
            # fall back to scoring the last token.
            cont_len = 1
        full_ids = full_ids[-(self.max_length + 1):]
        # Guard: never let truncation eat into the continuation.
        cont_len = min(cont_len, len(full_ids) - 1) or 1
        return full_ids, cont_len

    @torch.no_grad()
    def loglikelihood(
        self,
        requests: List[Tuple[str, str]],
        batch_size: int = 8,
        progress: bool = True,
    ) -> List[LLResult]:
        """Score a list of (context, continuation) pairs.

        Returns one :class:`LLResult` per request, in the same order.
        """
        encoded = [self._encode_request(ctx, cont) for ctx, cont in requests]

        results: List[Optional[LLResult]] = [None] * len(requests)
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))  # length-bucket

        iterator = range(0, len(order), batch_size)
        if progress:
            iterator = tqdm(iterator, desc="loglikelihood", total=(len(order) + batch_size - 1) // batch_size)

        for start in iterator:
            idxs = order[start:start + batch_size]
            # Feed sequence[:-1]; targets are sequence[1:].
            inputs = [encoded[i][0][:-1] for i in idxs]
            maxlen = max(len(x) for x in inputs)
            pad_id = self.tokenizer.pad_token_id

            input_ids, attn = [], []
            for x in inputs:
                pad = maxlen - len(x)
                input_ids.append([pad_id] * pad + x)
                attn.append([0] * pad + [1] * len(x))

            input_ids = torch.tensor(input_ids, device=self.device)
            attn = torch.tensor(attn, device=self.device)

            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)  # [B, maxlen, V]

            for row, i in enumerate(idxs):
                full_ids, cont_len = encoded[i]
                # Targets = last cont_len tokens of the full sequence.
                target = torch.tensor(full_ids[-cont_len:], device=self.device)
                # Left padding puts the last cont_len prediction positions at [maxlen-cont_len : maxlen].
                sel = logprobs[row, maxlen - cont_len:maxlen, :]
                tok_lp = sel.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                results[i] = LLResult(logprob=float(tok_lp.sum().item()), num_tokens=cont_len)

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Rolling NLL (perplexity / bits-per-byte)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def rolling_nll(self, text: str, stride: Optional[int] = None) -> Tuple[float, int, int]:
        """Total negative log-likelihood (nats) of a document under the model.

        Uses the standard fixed-length sliding-window recipe (HF perplexity
        guide): the document is scored in windows of ``max_length`` with the
        given ``stride``; each window only contributes the log-prob of tokens not
        already scored by the previous window, so every token is scored once with
        up to ``max_length`` tokens of left context.

        Returns ``(nll_sum_nats, num_tokens, num_bytes)`` where ``num_bytes`` is
        the UTF-8 byte length of ``text`` (denominator for bits-per-byte).
        """
        stride = stride or self.max_length // 2
        num_bytes = len(text.encode("utf-8"))
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) == 0:
            return 0.0, 0, num_bytes
        bos = self.tokenizer.bos_token_id
        if bos is not None:
            ids = [bos] + ids  # give the first real token some context
        ids_t = torch.tensor(ids, device=self.device)

        nll_sum, n_tokens, prev_end = 0.0, 0, 0
        seq_len = ids_t.size(0)
        for begin in range(0, seq_len, stride):
            end = min(begin + self.max_length, seq_len)
            trg_len = end - prev_end  # tokens not yet scored
            input_ids = ids_t[begin:end].unsqueeze(0)
            target = input_ids.clone()
            target[:, :-trg_len] = -100  # only score the new tokens
            loss = self.model(input_ids, labels=target).loss  # mean NLL over scored tokens
            nll_sum += float(loss.item()) * trg_len
            n_tokens += trg_len
            prev_end = end
            if end == seq_len:
                break
        return nll_sum, n_tokens, num_bytes

    # ------------------------------------------------------------------ #
    # Generation (IdiomCE)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        batch_size: int = 8,
        stop: Optional[List[str]] = None,
        progress: bool = True,
        chat: bool = False,
    ) -> List[str]:
        """Greedy completion for a list of raw-text prompts.

        If ``chat=True``, each prompt is wrapped as a single user turn and the
        tokenizer's chat template is applied (with a generation prompt) before
        decoding -- the correct protocol for instruction-tuned checkpoints, which
        otherwise underperform when prompted as raw base models.
        """
        outputs: List[str] = []
        iterator = range(0, len(prompts), batch_size)
        if progress:
            iterator = tqdm(iterator, desc="generate", total=(len(prompts) + batch_size - 1) // batch_size)

        for start in iterator:
            batch = prompts[start:start + batch_size]
            if chat:
                batch = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False, add_generation_prompt=True,
                    )
                    for p in batch
                ]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=not chat,   # chat template already adds them
            ).to(self.device)

            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # Strip the prompt (left-padded, so new tokens are after input width).
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for text in decoded:
                if stop:
                    for s in stop:
                        if s in text:
                            text = text.split(s)[0]
                outputs.append(text.strip())

        return outputs
