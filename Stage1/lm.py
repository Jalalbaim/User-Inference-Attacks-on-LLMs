"""
Language-model helpers shared by Stage 1 training and scoring (and later stages).

Document encoding, the same for training and scoring:
    ids = [EOS] + first MAX_LEN tokens of the email text
The EOS token (GPT-Neo's <|endoftext|>) is a start marker, so every one of the MAX_LEN text tokens is predicted.
log p(x) = sum over those text tokens of log p(token | previous tokens). The EOS itself is not scored.
s(x) = log p_theta(x) - log p_ref(x)  (plan section 1: sum of token log-probs over the truncated document).
"""
from __future__ import annotations

import math

import numpy as np
import torch

TEXT_CHARS = 4000   # cut very long emails before tokenizing (128 tokens never need more than ~1K chars)


def encode(tokenizer, texts, max_len):
    eos = tokenizer.eos_token_id
    out = []
    for t in texts:
        ids = tokenizer(t[:TEXT_CHARS], add_special_tokens=False)["input_ids"][:max_len]
        out.append([eos] + ids)
    return out


def batches(seqs, bs, order):
    """Yield (input_ids, attention_mask) right-padded to the longest sequence in the batch."""
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        L = max(len(seqs[j]) for j in idx)
        ids = torch.zeros((len(idx), L), dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            s = seqs[j]
            ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        yield idx, ids, mask


def token_logprobs(model, ids, mask):
    """Per-sequence sum of log p(token_t | tokens_<t) over real (unpadded) tokens t >= 1, and their count."""
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    tgt = ids[:, 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:].float()
    return (lp * m).sum(1), m.sum(1)


def lm_loss(model, ids, mask):
    """Mean NLL per predicted token over the batch (the training objective)."""
    s, n = token_logprobs(model, ids, mask)
    return -s.sum() / n.sum()


@torch.no_grad()
def score(model, seqs, device, bs=32, amp=False):
    """log p(x) (sum over text tokens) and token count for every sequence, in input order.
    Scoring runs in fp32 unless amp=True; it is cheap (a few thousand docs)."""
    model.eval()
    order = sorted(range(len(seqs)), key=lambda j: len(seqs[j]))   # length-sorted: less padding
    lp = np.zeros(len(seqs))
    nt = np.zeros(len(seqs), dtype=int)
    for idx, ids, mask in batches(seqs, bs, order):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            s, n = token_logprobs(model, ids.to(device), mask.to(device))
        lp[idx] = s.double().cpu().numpy()
        nt[idx] = n.long().cpu().numpy()
    return lp, nt


@torch.no_grad()
def mean_nll(model, seqs, device, bs=32, amp=True):
    """Token-weighted mean NLL (validation loss)."""
    if not seqs:
        return float("nan")
    lp, nt = score(model, seqs, device, bs=bs, amp=amp)
    return float(-lp.sum() / nt.sum())


def n_steps(n_docs, bs):
    return math.ceil(n_docs / bs)
