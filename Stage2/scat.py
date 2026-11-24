"""
Stage 2 (plan Step D.2), GPU part: S_cat, the concatenated-context log-ratio.

For user u, a subset of m held-out docs and an order pi:
    ids = [EOS] x_pi(1) [EOS] x_pi(2) ... [EOS] x_pi(m)
Each x is the doc's first MAX_LEN tokens, encoded exactly as in Stage 1 (Stage1/lm.py). log p(concat) is the sum of
log p over the TEXT tokens only (the EOS separators are not scored), so at m = 1, S_cat equals Stage 1's s(x).
    S_cat(u) = (1/R) sum_r [ log p_theta(concat_r) - log p_ref(concat_r) ]   (R orders; 1 at m = 1)
Sequences longer than the context (GPT-Neo: 2048; 32 docs x 129 tokens can reach 4128) use a sliding window
(user's choice (c), 2026-11-24): windows of W tokens with stride W/2. Every token is scored exactly once,
with all earlier tokens as context inside the first window, and at least W/2 earlier tokens after that.
Subsets and orders come from common/draws.py: the same for every seed and model.

Outputs:
  <ckpt>/scat_ref_<tag>.csv.gz            reference log p per unit (computed once, reused)
  <ckpt>/<run>_s<seed>/scat.csv.gz        per-unit table for that model (skipped if present)
  <out>/scat_<run>_s<seed>.csv.gz         copy, with labels: run, seed, user, b, trained, half, m, draw, order,
                                          doc_ids, n_tokens, logp_theta, logp_ref, s_cat
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "Stage1")]
from common import draws as DR                          # noqa: E402
from common.enron_data import load_splits, members     # noqa: E402
import lm                                               # noqa: E402  (Stage1/lm.py)
import train_score as TS                                # noqa: E402  (Stage1/train_score.py: load_model, deadline)

EXIT_RESUME = 75
log = TS.log


def windows(n, W, stride):
    """(begin, end, first scored position) covering target positions 1..n-1 exactly once."""
    out, prev, begin = [], 1, 0
    while True:
        end = min(begin + W, n)
        out.append((begin, end, prev))
        prev = end
        if end == n:
            return out
        begin += stride


def build(units, enc):
    """Concatenate encoded docs; score mask = 1 on text tokens, 0 on each doc's leading EOS."""
    seqs = []
    for (_, _, _, _, docs) in units:
        ids, sm = [], []
        for x in docs:
            e = enc[x]
            ids += e
            sm += [0] + [1] * (len(e) - 1)
        seqs.append((ids, sm))
    return seqs


@torch.no_grad()
def score_long(model, seqs, device, W, stride, tok_budget=8192, amp=False):
    """Sum of log p over scored tokens, and their count, per sequence (sliding window if longer than W)."""
    model.eval()
    items = []
    for i, (ids, sm) in enumerate(seqs):
        for b, e, f in windows(len(ids), W, stride):
            items.append((i, ids[b:e], [1 if (b + j) >= f and sm[b + j] else 0 for j in range(e - b)]))
    order = sorted(range(len(items)), key=lambda k: len(items[k][1]))
    lp, nt = np.zeros(len(seqs)), np.zeros(len(seqs), dtype=int)
    k = 0
    while k < len(order):
        j = k + 1   # sorted by length: the last item of a batch is its longest
        while j < len(order) and (j - k + 1) * len(items[order[j]][1]) <= tok_budget:
            j += 1
        idx = order[k:j]
        L = len(items[idx[-1]][1])
        ids = torch.zeros((len(idx), L), dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        tm = torch.zeros((len(idx), L))
        for r, t in enumerate(idx):
            w, m = items[t][1], items[t][2]
            ids[r, :len(w)] = torch.tensor(w)
            att[r, :len(w)] = 1
            tm[r, :len(w)] = torch.tensor(m, dtype=torch.float)
        ids, att, tm = ids.to(device), att.to(device), tm.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
        l = torch.log_softmax(logits.float(), dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        msk = tm[:, 1:]
        s, n = (l * msk).sum(1).double().cpu().numpy(), msk.sum(1).long().cpu().numpy()
        for r, t in enumerate(idx):
            lp[items[t][0]] += s[r]
            nt[items[t][0]] += n[r]
        k = j
    return lp, nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-seeds", default="0")
    ap.add_argument("--null-seeds", default="")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="EleutherAI/gpt-neo-125m")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--draws", type=int, default=4, help="subset draws per (user, m) for m < 32")
    ap.add_argument("--R", type=int, default=4, help="orders per subset")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=0, help="0 = window/2")
    ap.add_argument("--tok-budget", type=int, default=8192)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (default fp32, as the Stage-C table)")
    ap.add_argument("--smoke", action="store_true", help="only the users of the smoke runs; prints an estimate")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.ckpt = os.path.join(a.ckpt, "smoke")
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    targets = [("main", int(s)) for s in a.main_seeds.split()] + [("null", int(s)) for s in a.null_seeds.split()]
    targets = [(r, s) for r, s in targets if os.path.exists(os.path.join(a.ckpt, f"{r}_s{s}", "DONE"))]
    if not targets:
        log("S_cat: no finished model to score")
        return

    docs, users, _ = load_splits()
    held = [d for d in docs if d["role"] == "heldout"]
    by_user = {}
    for d in held:
        by_user.setdefault(d["user"], []).append(d["doc_id"])
    if a.smoke:   # the smoke models only know a few users; score those
        keep = set(pd.read_csv(os.path.join(a.ckpt, f"{targets[0][0]}_s{targets[0][1]}", "scores.csv.gz")).user)
        by_user = {u: v for u, v in by_user.items() if u in keep}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    need = {x for v in by_user.values() for x in v}
    sel = [d for d in held if d["doc_id"] in need]
    enc = dict(zip([d["doc_id"] for d in sel], lm.encode(tok, [d["text"] for d in sel], a.max_len)))

    U = DR.units(by_user, DR.M_GRID, a.draws, a.R)
    seqs = build(U, enc)
    lens = np.array([len(s[0]) for s in seqs])
    log(f"S_cat: {len(by_user)} users, {len(U)} units (draws {a.draws}, R {a.R}), {lens.sum():,} tokens per model, "
        f"longest {lens.max()}; models: {targets}")
    base = pd.DataFrame({"user": [u[0] for u in U], "m": [u[1] for u in U], "draw": [u[2] for u in U],
                         "order": [u[3] for u in U], "doc_ids": [DR.SEP.join(u[4]) for u in U]})

    def load(path_or_name):
        m = TS.load_model(path_or_name, device, a.tiny and path_or_name == a.model, tok)
        return m.float().eval()   # fp32 always (best/ may be stored in fp16)

    def ctx(m):
        W = min(a.window, int(m.config.max_position_embeddings))
        return W, (a.stride or W // 2)

    tag = f"{a.max_len}_D{a.draws}_R{a.R}_W{a.window}" + ("_tiny" if a.tiny else "")
    ref_path = os.path.join(a.ckpt, f"scat_ref_{tag}.csv.gz")
    ref = pd.read_csv(ref_path) if os.path.exists(ref_path) else None
    if ref is not None and not set(base.doc_ids) <= set(ref.doc_ids):
        ref = None
    last_secs = None
    if ref is None:
        t0 = time.time()
        m = load(a.model)
        W, st = ctx(m)
        log(f"S_cat: scoring {len(U)} units with the reference model (window {W}, stride {st})")
        lp, nt = score_long(m, seqs, device, W, st, a.tok_budget, a.amp)
        del m
        ref = base.assign(n_tokens=nt, logp_ref=lp)
        ref.to_csv(ref_path, index=False)
        last_secs = time.time() - t0
        log(f"S_cat: reference done in {last_secs / 60:.1f} min")
    ref = ref.drop_duplicates(["user", "m", "draw", "order"])
    key = ["user", "m", "draw", "order", "doc_ids"]

    for run, seed in targets:
        cdir = os.path.join(a.ckpt, f"{run}_s{seed}")
        cpath = os.path.join(cdir, "scat.csv.gz")
        if not os.path.exists(cpath):
            need_s = 1.3 * (last_secs or 0) + 120
            if TS.deadline_left() < need_s:
                log(f"S_cat {run}_s{seed}: {TS.deadline_left():.0f}s left, needs about {need_s:.0f}s; stopping to resume later")
                sys.exit(EXIT_RESUME)
            t0 = time.time()
            m = load(os.path.join(cdir, "best"))
            W, st = ctx(m)
            lp, nt = score_long(m, seqs, device, W, st, a.tok_budget, a.amp)
            del m
            torch.cuda.empty_cache() if device.type == "cuda" else None
            base.assign(n_tokens=nt, logp_theta=lp).to_csv(cpath, index=False)
            last_secs = time.time() - t0
            log(f"S_cat {run}_s{seed}: done in {last_secs / 60:.1f} min")
        th = pd.read_csv(cpath)
        df = th.merge(ref[key + ["logp_ref", "n_tokens"]], on=key, suffixes=("", "_ref"))
        assert len(df) == len(th) and (df.n_tokens == df.n_tokens_ref).all(), "S_cat: theta/ref units differ"
        mem = members(users, seed, run)
        df = df.drop(columns="n_tokens_ref").assign(
            run=run, seed=seed, s_cat=lambda x: x.logp_theta - x.logp_ref,
            b=lambda x: x.user.map(lambda u: users[u][f"b_s{seed}"]),
            trained=lambda x: x.user.map(lambda u: int(u in mem)),
            half=lambda x: x.user.map(lambda u: users[u]["half"]))
        cols = ["run", "seed", "user", "b", "trained", "half", "m", "draw", "order", "doc_ids", "n_tokens",
                "logp_theta", "logp_ref", "s_cat"]
        df[cols].to_csv(os.path.join(a.out, f"scat_{run}_s{seed}.csv.gz"), index=False)

        # consistency check: at m = 1, S_cat must equal Stage C's s(x) for the same doc (same encoding, fp32)
        sc = pd.read_csv(os.path.join(cdir, "scores.csv.gz")).set_index("doc_id")
        one = df[(df.m == 1) & df.doc_ids.isin(sc.index)]   # smoke models only score their own few users
        diff = np.abs(one.s_cat.to_numpy() - sc.loc[one.doc_ids, "s"].to_numpy())
        if len(diff) == 0:
            diff = np.array([np.nan])
        log(f"S_cat {run}_s{seed}: m = 1 vs score table ({len(one)} docs), max |diff| {diff.max():.2e}; "
            f"mean S_cat at m = 32: trained {df[(df.m == 32) & (df.trained == 1)].s_cat.mean():.1f}, "
            f"untrained {df[(df.m == 32) & (df.trained == 0)].s_cat.mean():.1f}")

    if a.smoke and last_secs:
        n_all = sum(1 for _ in {d['user'] for d in held})
        log(f"S_cat SMOKE: {last_secs / 60:.2f} min per model for {len(by_user)} users -> full run about "
            f"{last_secs * n_all / len(by_user) / 60:.0f} min per model ({n_all} users), plus one reference pass")


if __name__ == "__main__":
    main()
