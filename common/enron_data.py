"""
Load the committed Enron splits (data/processed/enron/splits, built by data/enron/splits.py).

Where the data is looked for, in order:
  $UIA_DATA_DIR/enron/splits                       (explicit override)
  $UIA_REPO_DIR/data/processed/enron/splits        (Kaggle: the repo clone made by run_kaggle.sh)
  <this repo>/data/processed/enron/splits          (local checkout)
The file's sha256 must match ENRON_SPLITS_SHA256 (also in data/enron/README.md).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os

ENRON_SPLITS_SHA256 = "115dabb94c400a9bcd9962b41a1212622de8fd0b94c704d4a59c7f4918f87ac1"
_REL = os.path.join("enron", "splits")


def find_splits_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = []
    if os.environ.get("UIA_DATA_DIR"):
        cands.append(os.path.join(os.environ["UIA_DATA_DIR"], _REL))
    if os.environ.get("UIA_REPO_DIR"):
        cands.append(os.path.join(os.environ["UIA_REPO_DIR"], "data", "processed", _REL))
    cands.append(os.path.join(here, "data", "processed", _REL))
    for c in cands:
        if os.path.isfile(os.path.join(c, "docs.jsonl.gz")):
            return c
    raise FileNotFoundError("Enron splits not found; looked in:\n  " + "\n  ".join(cands))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_splits(check_hash: bool = True):
    """Returns (docs, users, splits_dir).
    docs:  list of dicts {doc_id, user, role in heldout/val/train, date, address, text}
    users: dict user -> row of users.csv (ints converted; keys n_docs, n_heldout, n_val, n_train, half, b_s0..)"""
    d = find_splits_dir()
    p = os.path.join(d, "docs.jsonl.gz")
    if check_hash:
        got = sha256(p)
        if got != ENRON_SPLITS_SHA256:
            raise RuntimeError(f"{p}: sha256 {got} != expected {ENRON_SPLITS_SHA256}. "
                               "The splits changed: regenerate on purpose and update ENRON_SPLITS_SHA256.")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    users = {}
    with open(os.path.join(d, "users.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            users[r["user"]] = {k: (v if k in ("user", "half") else int(v)) for k, v in r.items()}
    return docs, users, d


def members(users: dict, seed: int, run: str = "main") -> set:
    """Users whose train docs are trained on.
    main: coin flip b_s<seed> = 1.  null: same coin flip, but FIT users only, so no EVAL user is trained on
    (plan Step B.4: the null model is fine-tuned without the evaluated users)."""
    key = f"b_s{seed}"
    if key not in next(iter(users.values())):
        raise KeyError(f"no coin flips for seed {seed} in users.csv (rebuild splits with more --seeds)")
    if run == "main":
        return {u for u, r in users.items() if r[key] == 1}
    if run == "null":
        return {u for u, r in users.items() if r[key] == 1 and r["half"] == "FIT"}
    raise ValueError(run)
