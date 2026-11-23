"""
Stage 1 analysis (local, no GPU): from the per-document score table to T-hat AUROC vs m and certificates.

T-hat_m(u) = mean of s(x) over m held-out docs of user u, drawn without replacement (plan Step D).
Each draw gives one score per user (plan D.5); metrics are averaged over R draws. At m = 32 all held-out
docs are used, so there is a single draw.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import certificates as C   # noqa: E402

M_GRID = (1, 2, 4, 8, 16, 32)
KANDPAL_AUROC = 0.873   # Kandpal et al., Enron, GPT-Neo 125M, mean aggregation (reported +/- 3.3)


def load(results_dir):
    """All scores*.csv.gz and history*.json under results_dir (recursive)."""
    fs = sorted(glob.glob(os.path.join(results_dir, "**", "scores*.csv.gz"), recursive=True))
    if not fs:
        raise FileNotFoundError(f"no scores*.csv.gz under {results_dir}")
    # keep_default_na=False: the run label "null" must not be read as a missing value
    df = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in fs])
    df = df.drop_duplicates(["run", "seed", "doc_id"]).reset_index(drop=True)
    hist = {}
    for f in glob.glob(os.path.join(results_dir, "**", "history_*.json"), recursive=True):
        hist[os.path.basename(f)[8:-5]] = json.load(open(f))
    return df, hist


def history_table(hist):
    rows = []
    for tag, h in sorted(hist.items()):
        for e in h["hist"]:
            rows.append({"run": tag, "epoch": e["epoch"], "train_loss": e["train_loss"], "val_in": e["val_in"],
                         "val_out": e["val_out"], "min": e["secs"] / 60,
                         "best": e["epoch"] == h["best"]["epoch"]})
    return pd.DataFrame(rows)


def per_user_draws(d, m, R, rng):
    """(labels, matrix users x R of mean s over m random held-out docs)."""
    g = {u: x.s.to_numpy() for u, x in d.groupby("user")}
    lab = d.groupby("user").b.first()
    users = lab.index.to_numpy()
    R_eff = 1 if all(len(g[u]) <= m for u in users) else R
    S = np.empty((len(users), R_eff))
    for i, u in enumerate(users):
        v = g[u]
        mm = min(m, len(v))
        for r in range(R_eff):
            S[i, r] = v[rng.choice(len(v), mm, replace=False)].mean() if mm < len(v) else v.mean()
    return lab.to_numpy(), S


def auroc_vs_m(d, m_grid=M_GRID, R=200, eta=0.05, seed=0):
    """One row per m: AUROC / J-hat / DKW certificate averaged over R draws (with 2.5-97.5% draw range),
    plus the AUROC -> J bound, TPR at 1% FPR (flag: below 1/n0), and users needed (Cor. 1)."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in m_grid:
        y, S = per_user_draws(d, m, R, rng)
        a, j, c, t1 = [], [], [], []
        for r in range(S.shape[1]):
            s1, s0 = S[y == 1, r], S[y == 0, r]
            a.append(C.auroc(s1, s0))
            j.append(C.youden_j(s1, s0)[0])
            c.append(C.dkw_certificate(s1, s0, eta))
            t1.append(C.tpr_at_fpr(s1, s0, 0.01)[0])
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        rows.append({"m": m, "n1": n1, "n0": n0, "draws": S.shape[1],
                     "auroc": np.mean(a), "auroc_lo": np.quantile(a, .025), "auroc_hi": np.quantile(a, .975),
                     "j_hat": np.mean(j), "dkw_cert": np.mean(c), "dkw_cert_positive_frac": np.mean(np.array(c) > 0),
                     "j_from_auroc": C.j_from_auroc(np.mean(a)), "tpr_at_1pct": np.mean(t1),
                     "tpr_1pct_below_1_over_n0": 0.01 < 1 / n0,
                     "dkw_threshold": C.dkw_eps(n1, eta) + C.dkw_eps(n0, eta),
                     "users_needed_per_class": C.users_needed(np.mean(j), eta)})
    return pd.DataFrame(rows)


def report_m32(d, eta=0.05):
    """Full Step-E row at m = 32 (all held-out docs): AUROC with user-bootstrap CI, J-hat, DKW certificate."""
    t = d.groupby("user").agg(b=("b", "first"), tau=("s", "mean"))
    return C.report(t.tau[t.b == 1].to_numpy(), t.tau[t.b == 0].to_numpy(), eta)


def gate(main_d, null_d=None, tol=0.05):
    """Plan gate for Stage 1, at m = 32:
    main: AUROC within tol of Kandpal's 0.873 (tol is our choice; Kandpal report +/- 0.033 over seeds),
    null (EVAL users, fake labels): user-bootstrap 95% CI of AUROC contains 0.5 and DKW certificate <= 0."""
    r = report_m32(main_d)
    out = {"auroc_m32": r["auroc"], "auroc_m32_ci": r["auroc_ci"], "kandpal": KANDPAL_AUROC, "tol": tol,
           "main_ok": abs(r["auroc"] - KANDPAL_AUROC) <= tol}
    if null_d is not None:
        n = report_m32(null_d)
        out.update({"null_auroc_m32": n["auroc"], "null_auroc_ci": n["auroc_ci"], "null_dkw_cert": n["dkw_cert"],
                    "null_ok": n["auroc_ci"][0] <= 0.5 <= n["auroc_ci"][1] and n["dkw_cert"] <= 0})
    return out
