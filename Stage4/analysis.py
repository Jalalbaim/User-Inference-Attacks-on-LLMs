"""
Stage 4a (local, no GPU): light users, plan Step G items 1-4, WITHOUT cap retraining (that is Stage 4b, not decided).

Post-processing of the Stage 2 per-user statistics (Stage2/analysis.cached_user_values).
Per (seed, m, draw, statistic): threshold t_gamma = (1 - gamma)-quantile of the statistic among NON-member users of the
same group; a member is detected if its statistic > t_gamma (common.certificates.tpr_at_fpr rule). Members are binned
by k_u (training docs of the user) and by alpha_u (the user's share of the training set, depends on the seed).
TPR per bin is averaged over draws, then seeds.

Cor. 2 prediction: as m grows, a member's detection tends to 1{tau_u > t_gamma}; T-hat stays near chance (TPR ~ gamma)
for light users at any m, while user-level statistics are not bound by that ceiling.
Caveat: Enron after the 150+32 filter has NO truly light users: k_u >= 114 for every user. This stage can only compare
lighter with heavier users; real light users need caps (Stage 4b).
T and S_cat: all users; the aggregator: EVAL users only (threshold from EVAL non-members).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
from common.stage2_link import stage2  # noqa: E402
S2 = stage2()

STATS = {"T": "all", "S_cat": "all", "Agg": "EVAL"}
GAMMA = 0.05


def k_bins(uv, q=4):
    """Quantile bin edges of k_u over all users (k_u does not depend on the seed)."""
    k = uv.groupby("user").k_u.first()
    edges = np.unique(np.quantile(k, np.linspace(0, 1, q + 1)))
    return edges


def member_detection(uv, run="main", gamma=GAMMA):
    """One row per (seed, m, draw, stat, member user): k_u, alpha_u, value, threshold, detected."""
    u = uv[uv.run == run]
    out = []
    for (sd, m, dr), g in u.groupby(["seed", "m", "draw"]):
        for st, grp in STATS.items():
            x = g if grp == "all" else g[g.half == "EVAL"]
            if st not in x or x[st].isna().any():
                continue
            thr = np.quantile(x.loc[x.b == 0, st].to_numpy(float), 1 - gamma)
            mem = x[x.b == 1]
            out.append(pd.DataFrame({"seed": sd, "m": m, "draw": dr, "stat": st, "user": mem.user.to_numpy(),
                                     "k_u": mem.k_u.to_numpy(), "alpha_u": mem.alpha_u.to_numpy(),
                                     "value": mem[st].to_numpy(float), "thr": thr,
                                     "detected": (mem[st].to_numpy(float) > thr).astype(float)}))
    return pd.concat(out, ignore_index=True)


def tpr_by_bin(det, edges, by="k_u"):
    """TPR at FPR gamma per (stat, m, bin): mean over members in the bin, then draws, then seeds."""
    d = det.assign(bin=pd.cut(det[by], edges, include_lowest=True))
    per = d.groupby(["stat", "m", "bin", "seed", "draw"], observed=True).agg(tpr=("detected", "mean"),
                                                                           members=("user", "size")).reset_index()
    per = per.groupby(["stat", "m", "bin", "seed"], observed=True).mean(numeric_only=True).reset_index()
    return per.groupby(["stat", "m", "bin"], observed=True).agg(
        tpr=("tpr", "mean"), tpr_min=("tpr", "min"), tpr_max=("tpr", "max"),
        members_per_seed=("members", "mean")).reset_index()


def rank_corr(det):
    """Spearman correlation of k_u and alpha_u with the member's statistic value, per (stat, m), mean over seeds/draws."""
    rows = []
    for (st, m), g in det.groupby(["stat", "m"]):
        r = [(spearmanr(x.k_u, x.value)[0], spearmanr(x.alpha_u, x.value)[0]) for _, x in g.groupby(["seed", "draw"])]
        r = np.array(r)
        rows.append({"stat": st, "m": m, "rho_k_u": r[:, 0].mean(), "rho_alpha_u": r[:, 1].mean(),
                     "rho_k_u_min": r[:, 0].min(), "rho_k_u_max": r[:, 0].max()})
    return pd.DataFrame(rows)
