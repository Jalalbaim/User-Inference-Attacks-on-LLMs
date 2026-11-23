"""
Enron preparation, step 2: document roles per user, the FIT/EVAL partition, and member coin flips per seed.
Plan Step A.3-4 and Step B.1. Decisions: data/enron/README.md (agreed 2026-11-23).

Input:  data/processed/enron/docs.jsonl.gz (from prepare.py)
Output: data/processed/enron/splits/
  docs.jsonl.gz   every kept doc + doc_id (= tarball path, unique) + role in {heldout, val, train}
  users.csv       user, n_docs, n_heldout, n_val, n_train, half (FIT/EVAL), b_s0 .. b_s9 (1 = member)
  summary.json    counts per seed, alpha_u stats, cross-user leak check

Fixed across seeds: each user's held-out / val / train documents and the FIT/EVAL partition
(plan: "different seeds reuse the same users"). Varies with the seed: the coin flip b_u ~ Bern(1/2),
drawn independently for every user (plan Step B.1).
Members train on their 'train' docs. Non-members' 'train' docs are never used. 'heldout' docs are
never trained on, for anyone. Tokenization and 128-token truncation happen in Stage 1.

    .venv/Scripts/python data/enron/splits.py [--heldout 32] [--val-frac 0.05] [--val-min 4] [--seeds 10]
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import re
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "data", "processed", "enron", "docs.jsonl.gz")
OUT = os.path.join(ROOT, "data", "processed", "enron", "splits")


def user_rng(seed: int, user: str):
    """Per-user generator: independent of the order users are processed in."""
    return np.random.default_rng([seed, zlib.crc32(user.encode())])


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-min", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=10, help="coin-flip seeds 0..seeds-1")
    ap.add_argument("--doc-seed", type=int, default=0, help="seed of the per-user document split")
    ap.add_argument("--half-seed", type=int, default=0, help="seed of the FIT/EVAL partition")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    docs = collections.defaultdict(list)
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs[r["user"]].append(r)
    users = sorted(docs)
    paths = [r["path"] for u in users for r in docs[u]]
    assert len(paths) == len(set(paths)), "doc_id (tarball path) is not unique"

    # 1. per-user document roles: random held-out (main experiment; the temporal split is Stage 5)
    role = {}
    rows = {}
    for u in users:
        d = sorted(docs[u], key=lambda r: r["path"])       # fixed order before shuffling
        perm = user_rng(a.doc_seed, u).permutation(len(d))
        n_val = max(a.val_min, int(round(a.val_frac * (len(d) - a.heldout))))
        n_train = len(d) - a.heldout - n_val
        assert n_train > 0, u
        for j, i in enumerate(perm):
            role[d[i]["path"]] = "heldout" if j < a.heldout else ("val" if j < a.heldout + n_val else "train")
        rows[u] = {"user": u, "n_docs": len(d), "n_heldout": a.heldout, "n_val": n_val, "n_train": n_train}

    # 2. FIT/EVAL: a random half of the users (floor for FIT), fixed for all seeds
    perm = np.random.default_rng(a.half_seed).permutation(len(users))
    for rank, i in enumerate(perm):
        rows[users[i]]["half"] = "FIT" if rank < len(users) // 2 else "EVAL"

    # 3. coin flips, independent per user and per seed
    for s in range(a.seeds):
        b = np.random.default_rng(1000 + s).integers(0, 2, size=len(users))
        for u, bu in zip(users, b):
            rows[u][f"b_s{s}"] = int(bu)

    # 4. summary and checks
    text_of = {r["path"]: norm(r["text"]) for u in users for r in docs[u]}
    summ = collections.OrderedDict()
    summ["params"] = vars(a)
    summ["users"] = len(users)
    summ["docs"] = len(paths)
    summ["docs_by_role"] = dict(collections.Counter(role.values()))
    summ["users_by_half"] = dict(collections.Counter(r["half"] for r in rows.values()))
    summ["val_docs_per_user"] = [int(min(r["n_val"] for r in rows.values())),
                                 int(max(r["n_val"] for r in rows.values()))]
    summ["train_docs_per_user_min_med_max"] = [int(np.min([r["n_train"] for r in rows.values()])),
                                               int(np.median([r["n_train"] for r in rows.values()])),
                                               int(np.max([r["n_train"] for r in rows.values()]))]
    per_seed = []
    for s in range(a.seeds):
        mem = [u for u in users if rows[u][f"b_s{s}"] == 1]
        non = [u for u in users if rows[u][f"b_s{s}"] == 0]
        k = np.array([rows[u]["n_train"] for u in mem], float)
        alpha = k / k.sum()
        # cross-user leak: a non-member's held-out text appearing verbatim in some member's training docs
        member_train = {text_of[r["path"]] for u in mem for r in docs[u] if role[r["path"]] == "train"}
        leak = [u for u in non for r in docs[u] if role[r["path"]] == "heldout" and text_of[r["path"]] in member_train]
        per_seed.append({
            "seed": s, "members": len(mem), "nonmembers": len(non),
            "members_FIT": sum(rows[u]["half"] == "FIT" for u in mem),
            "nonmembers_FIT": sum(rows[u]["half"] == "FIT" for u in non),
            "members_EVAL": sum(rows[u]["half"] == "EVAL" for u in mem),
            "nonmembers_EVAL": sum(rows[u]["half"] == "EVAL" for u in non),
            "train_docs": int(k.sum()),
            "alpha_min_med_max": [float(alpha.min()), float(np.median(alpha)), float(alpha.max())],
            "nonmember_heldout_docs_verbatim_in_member_train": len(leak),
            "nonmember_users_affected": len(set(leak)),
        })
    summ["per_seed"] = per_seed

    # outputs
    with gzip.open(os.path.join(OUT, "docs.jsonl.gz"), "wt", encoding="utf-8") as f:
        for u in users:
            for r in sorted(docs[u], key=lambda r: r["path"]):
                f.write(json.dumps({"doc_id": r["path"], "user": u, "role": role[r["path"]],
                                    "date": r["date"], "address": r["address"], "text": r["text"]}) + "\n")
    cols = ["user", "n_docs", "n_heldout", "n_val", "n_train", "half"] + [f"b_s{s}" for s in range(a.seeds)]
    with open(os.path.join(OUT, "users.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows[u] for u in users)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_seed"}, indent=1))
    print("seed members/non  FIT m/n  EVAL m/n  train_docs  alpha max  leak docs (users)")
    for p in per_seed:
        print(f"{p['seed']:>4} {p['members']:>4}/{p['nonmembers']:<4} {p['members_FIT']:>3}/{p['nonmembers_FIT']:<3}"
              f" {p['members_EVAL']:>4}/{p['nonmembers_EVAL']:<4} {p['train_docs']:>9}  {p['alpha_min_med_max'][2]:.3f}"
              f"     {p['nonmember_heldout_docs_verbatim_in_member_train']} ({p['nonmember_users_affected']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
