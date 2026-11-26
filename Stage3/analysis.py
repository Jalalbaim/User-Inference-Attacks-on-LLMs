"""
Stage 3 (local, no GPU): DKW certificate vs number of users per class (plan section 4 row 3, Cor. 1).

Post-processing of the Stage 2 per-user statistics (Stage2/analysis.cached_user_values).
For each (seed, m, draw, statistic): draw n users per class WITHOUT replacement (R times), compute J-hat_n and the DKW
certificate J-hat_n - 2 eps(n). Average over subsamples, then draws, then seeds.

Cor. 1 prediction: the certificate is positive only if n > 8.76 / J^2 (eta = 0.05). With the full-sample J-hat:
  n_pred = 8.76 / J-hat_full^2, and predicted mean certificate(n) = J-hat_full - 2 eps(n).
Empirical: n_50 = smallest n at which P(certificate > 0) >= 0.5 (linear interpolation on the n grid).
Caveats: J-hat_n is a max over thresholds, so it is biased upwards at small n (the predicted curve ignores this);
near the full sample size the subsamples overlap heavily, so they are not independent replications.
T and S_cat use all 175 users; the aggregator only EVAL users (its FIT users are its training set).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
from common import certificates as C   # noqa: E402
from common.stage2_link import stage2  # noqa: E402
S2 = stage2()                           # Stage2/analysis.py

N_GRID = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80)
STATS = {"T": "all", "S_cat": "all", "Agg": "EVAL"}
ETA = 0.05


def cert_vs_n(uv, run="main", n_grid=N_GRID, R=200, eta=ETA, m_list=None, seed=0):
    """Long table: seed, m, draw, stat, users, n, n_max, mean_jhat, mean_cert, p_pos, jhat_full, n_pred."""
    rng = np.random.default_rng(seed)
    u = uv[uv.run == run]
    if m_list is not None:
        u = u[u.m.isin(m_list)]
    rows = []
    for (sd, m, dr), g in u.groupby(["seed", "m", "draw"]):
        for st, grp in STATS.items():
            x = g if grp == "all" else g[g.half == "EVAL"]
            if st not in x or x[st].isna().any():
                continue
            s1, s0 = x.loc[x.b == 1, st].to_numpy(float), x.loc[x.b == 0, st].to_numpy(float)
            nmax = min(len(s1), len(s0))
            jfull = C.youden_j(s1, s0)[0]
            base = {"seed": sd, "m": m, "draw": dr, "stat": st, "users": grp, "n_max": nmax, "jhat_full": jfull,
                    "n_pred": C.users_needed(jfull, eta)}
            for n in [k for k in n_grid if k <= nmax]:
                j = np.array([C.youden_j(rng.choice(s1, n, replace=False), rng.choice(s0, n, replace=False))[0]
                              for _ in range(R)])
                c = j - 2 * C.dkw_eps(n, eta)
                rows.append({**base, "n": n, "mean_jhat": j.mean(), "mean_cert": c.mean(), "p_pos": (c > 0).mean(),
                             "pred_cert": jfull - 2 * C.dkw_eps(n, eta)})
    return pd.DataFrame(rows)


def summarize(cv):
    """Mean over draws within seed, then mean / min / max over seeds, per (stat, m, n)."""
    per_seed = cv.groupby(["stat", "users", "m", "n", "seed"]).mean(numeric_only=True).drop(columns="draw").reset_index()
    agg = per_seed.groupby(["stat", "users", "m", "n"]).agg(
        seeds=("seed", "nunique"), mean_jhat=("mean_jhat", "mean"), mean_cert=("mean_cert", "mean"),
        cert_min=("mean_cert", "min"), cert_max=("mean_cert", "max"), p_pos=("p_pos", "mean"),
        pred_cert=("pred_cert", "mean"), jhat_full=("jhat_full", "mean")).reset_index()
    return per_seed, agg


def n50(per_seed):
    """Per (stat, m, seed): empirical n at which P(cert > 0) reaches 0.5, vs Cor. 1's n_pred = 8.76 / J-hat_full^2."""
    rows = []
    for (st, m, sd), g in per_seed.groupby(["stat", "m", "seed"]):
        g = g.sort_values("n")
        n, p = g.n.to_numpy(float), g.p_pos.to_numpy()
        if p[0] >= 0.5:
            emp = n[0]
        elif (p >= 0.5).any():
            i = int(np.argmax(p >= 0.5))
            emp = n[i - 1] + (0.5 - p[i - 1]) * (n[i] - n[i - 1]) / (p[i] - p[i - 1])
        else:
            emp = np.nan   # not reached within the available users
        rows.append({"stat": st, "m": m, "seed": sd, "n50_empirical": emp, "n_pred": g.n_pred.iloc[0],
                     "jhat_full": g.jhat_full.iloc[0], "n_max": g.n_max.iloc[0]})
    return pd.DataFrame(rows)
