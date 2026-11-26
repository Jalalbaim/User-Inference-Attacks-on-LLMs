"""
Stage 6 (local, no GPU): pool identity, Prop. 3 / plan Step I. SANITY CHECK ONLY, not a headline result.

Post-processing of the Stage 2 per-user statistics (Stage2/analysis.cached_user_values).
Setting: a data owner holds a POOL of n_p users, a fraction pi of which were trained on, and a disjoint REFERENCE set of
known non-members. Prop. 3: KS(pool, non) = pi * J(S), with the one-sided KS sup_t [F_non(t) - F_pool(t)] (pool
scores higher); so KS-hat - eps(n_p) - eps(n_ref) lower-bounds both J(S) and pi (DKW, eta/2 per sample).

Per seed and replicate: non-members are split at random into a reference half and a supply for the pool; the pool
takes round(pi * n_p) random members and the rest from that supply. One-sided KS-hat = youden_j(pool, reference).
Expectation (plan): <= 0 at pi = 0; at Enron's user counts probably <= 0 even at pi = 1.
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
S2 = stage2()

PI_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
NP_GRID = (15, 25, 35)
ETA = 0.05


def pool_bounds(uv, stat="T", m=32, run="main", pi_grid=PI_GRID, np_grid=NP_GRID, R=500, eta=ETA, seed=0):
    """Long table per (seed, n_p, pi): n_ref, mean KS-hat, pi * J-hat_full (Prop. 3 identity), mean bound, P(bound > 0)."""
    rng = np.random.default_rng(seed)
    u = uv[(uv.run == run) & (uv.m == m) & (uv.draw == 0)]
    rows = []
    for sd, g in u.groupby("seed"):
        s1, s0 = g.loc[g.b == 1, stat].to_numpy(float), g.loc[g.b == 0, stat].to_numpy(float)
        jfull = C.youden_j(s1, s0)[0]
        n_ref = len(s0) // 2
        for n_p in np_grid:
            if n_p > len(s0) - n_ref or n_p > len(s1):
                continue
            for pi in pi_grid:
                k = int(round(pi * n_p))
                ks = np.empty(R)
                for r in range(R):
                    p0 = rng.permutation(s0)
                    ref, supply = p0[:n_ref], p0[n_ref:]
                    pool = np.concatenate([rng.choice(s1, k, replace=False), rng.choice(supply, n_p - k, replace=False)])
                    ks[r] = C.youden_j(pool, ref)[0]
                b = ks - C.dkw_eps(n_p, eta) - C.dkw_eps(n_ref, eta)
                rows.append({"stat": stat, "m": m, "seed": sd, "n_p": n_p, "n_ref": n_ref, "pi": pi, "ks": ks.mean(),
                             "pi_times_J": pi * jfull, "bound": b.mean(), "p_bound_pos": (b > 0).mean(),
                             "dkw_terms": C.dkw_eps(n_p, eta) + C.dkw_eps(n_ref, eta)})
    return pd.DataFrame(rows)


def summarize(pb):
    return pb.groupby(["stat", "m", "n_p", "pi"]).agg(
        seeds=("seed", "nunique"), n_ref=("n_ref", "mean"), ks=("ks", "mean"), pi_times_J=("pi_times_J", "mean"),
        bound=("bound", "mean"), bound_max_seed=("bound", "max"), p_bound_pos=("p_bound_pos", "mean"),
        dkw_terms=("dkw_terms", "mean")).reset_index()
