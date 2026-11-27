"""
Stage 7a (local, no GPU): certified LOWER bounds on user-level epsilon from the existing audits, and the ceiling that
the number of users puts on any such bound. Post-processing of Stage 2 (Stage2/analysis.cached_user_values).
The Stage 2 models are NOT DP (true epsilon = infinity), so every bound here is a demonstration of audit power: it is
the largest epsilon this audit could refute if the trainer claimed it.

Two routes, both at confidence 1 - 0.05:
  (1) TV inversion (Lemma 1). (eps, delta)-user-level DP => Adv <= (e^eps - 1 + 2 delta) / (e^eps + 1). The DKW
      certificate c lower-bounds J(S) <= Adv, so eps >= log((1 + c - 2 delta) / (1 - c)). Adjacency as in Lemma 1.
  (2) One-run audit (Steinke, Nasr, Jagielski 2023), users as the audited units. Valid here because each user's
      membership is an independent fair coin (data/enron/splits.py: rng(1000+s).integers(0, 2)). The guesser ranks
      users by the statistic and guesses on k of them (rule "sym": k/2 top = member, k/2 bottom = non-member;
      rule "top": k top = member). With v correct guesses, eps-DP implies W <= Bin(k, e^eps/(1+e^eps)) stochastically,
      so eps is refuted when P[Bin(k, p_eps) >= v] < beta. PURE-eps version (delta = 0): the delta correction of the
      paper must be added before applying this to the DP runs of 7b.
      The (rule, k) is not chosen on the data: every pair of the fixed grid is tested, Bonferroni beta / |grid|.
      "oracle" = best pair chosen after seeing the labels, NOT valid, shown only as a reference.
Null run: EVAL users only for every statistic (its FIT members are trained on, as in Stage 2).
Aggregator: EVAL users only; its FIT users (and their coins) are treated as fixed training data, so the EVAL coins are
still independent fair coins given everything the guesser uses.
One test per model: draw 0 of the m-doc subsets (pre-specified), not a max over draws.
Pooled over seeds: S independent runs, same (rule, k) in each, so W_total <= Bin(S k, p_eps); assumes every run
satisfies the same eps (meaningful for the DP runs of 7b; here only shows the ceiling with S runs).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
from common import certificates as C   # noqa: E402
from common.stage2_link import stage2  # noqa: E402
S2 = stage2()

ETA = 0.05                      # confidence 1 - ETA for every bound
BETA = 0.05
DELTAS = (0.0, 1e-5)
STATS = {"T": "all", "S_cat": "all", "Agg": "EVAL"}
K_GRID = (10, 20, 30, 50, 80, 120, 175)
RULES = ("sym", "top")
GRID = [(r, k) for r in RULES for k in K_GRID]   # pre-registered; |GRID| = 14 for the Bonferroni correction
EPS_MAX = 30.0


# ------------------------------------------------------------------ route 1: TV inversion
def eps_from_adv(adv, delta=0.0):
    """Smallest eps with (e^eps - 1 + 2 delta)/(e^eps + 1) >= adv; 0 if adv <= 2 delta / 2 (nothing certified)."""
    if adv <= 0 or 1 + adv - 2 * delta <= 1 - adv:
        return 0.0
    return float(math.log((1 + adv - 2 * delta) / (1 - adv))) if adv < 1 else math.inf


def adv_from_eps(eps, delta=0.0):
    return (math.exp(eps) - 1 + 2 * delta) / (math.exp(eps) + 1)


# ------------------------------------------------------------------ route 2: one-run audit
def eps_one_run(k, v, beta=BETA):
    """Largest eps refuted: sup{eps : P[Bin(k, p_eps) >= v] < beta}; 0 if not even eps = 0 is refuted."""
    if k == 0 or binom.sf(v - 1, k, 0.5) >= beta:
        return 0.0
    lo, hi = 0.0, EPS_MAX
    for _ in range(60):
        e = (lo + hi) / 2
        if binom.sf(v - 1, k, 1 / (1 + math.exp(-e))) < beta:
            lo = e
        else:
            hi = e
    return lo


def guesses(s, b, rule, k):
    """(guesses made, correct guesses) for ranking by s (larger = member). None if k does not fit."""
    n = len(s)
    if k > n:
        return None
    o = np.argsort(s, kind="stable")
    if rule == "top":
        return k, int(b[o[n - k:]].sum())
    h = k // 2
    if h == 0:
        return None
    return 2 * h, int(b[o[n - h:]].sum() + (1 - b[o[:h]]).sum())


def _users(uv, run):
    u = uv[(uv.run == run) & (uv.draw == 0)]
    for (sd, m), g in u.groupby(["seed", "m"]):
        for st, grp in STATS.items():
            grp = "EVAL" if run == "null" else grp      # null: FIT members ARE trained on; only EVAL users are placebos
            x = g if grp == "all" else g[g.half == "EVAL"]
            if st not in x or x[st].isna().any():
                continue
            yield sd, m, st, grp, x[st].to_numpy(float), x.b.to_numpy(int)


def bounds(uv, run="main", eta=ETA, beta=BETA):
    """Per (seed, m, stat): DKW certificate and eps_TV; one-run eps (Bonferroni over GRID) and the invalid oracle."""
    rows, gr = [], []
    bonf = beta / len(GRID)
    for sd, m, st, grp, s, b in _users(uv, run):
        s1, s0 = s[b == 1], s[b == 0]
        cert = C.dkw_certificate(s1, s0, eta)
        r = {"run": run, "seed": sd, "m": m, "stat": st, "users": grp, "n1": len(s1), "n0": len(s0),
             "auroc": C.auroc(s1, s0), "cert": cert}
        for d in DELTAS:
            r[f"eps_tv_d{d:g}"] = eps_from_adv(cert, d)
        best, orc = (0.0, None), (0.0, None)
        for rule, k in GRID:
            g = guesses(s, b, rule, k)
            if g is None:
                continue
            kk, v = g
            e, eo = eps_one_run(kk, v, bonf), eps_one_run(kk, v, beta)
            gr.append({"run": run, "seed": sd, "m": m, "stat": st, "rule": rule, "k": kk, "v": v})
            if e > best[0]:
                best = (e, f"{rule}{kk}:{v}")
            if eo > orc[0]:
                orc = (eo, f"{rule}{kk}:{v}")
        r.update(eps_1run=best[0], pick_1run=best[1], eps_1run_oracle=orc[0], pick_oracle=orc[1])
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(gr)


def pooled(gr, beta=BETA):
    """Pooled over seeds (same (rule, k) in each run): eps from (sum k, sum v), Bonferroni over GRID."""
    bonf = beta / len(GRID)
    p = gr.groupby(["run", "m", "stat", "rule", "k"]).agg(seeds=("seed", "nunique"), K=("k", "sum"), V=("v", "sum"))
    p = p.reset_index()
    p["eps"] = [eps_one_run(K, V, bonf) for K, V in zip(p.K, p.V)]
    i = p.groupby(["run", "m", "stat"]).eps.idxmax()
    return p.loc[i].reset_index(drop=True)


# ------------------------------------------------------------------ ceilings (perfect attack)
def ceilings(n_grid=(20, 50, 100, 175, 350, 1000, 3000, 10000), runs=(1, 10), eta=ETA, beta=BETA):
    """Largest certifiable eps vs total users n (balanced classes), for a PERFECT attack (all guesses right, J-hat = 1)."""
    rows = []
    for n in n_grid:
        cmax = 1 - 2 * C.dkw_eps(n / 2, eta)
        r = {"n": n, "cert_max": cmax, "eps_tv_max": eps_from_adv(cmax)}
        for R in runs:
            r[f"eps_1run_max_{R}runs"] = eps_one_run(R * n, R * n, beta)                 # one fixed guess rule
            r[f"eps_1run_max_{R}runs_bonf"] = eps_one_run(R * n, R * n, beta / len(GRID))
        rows.append(r)
    return pd.DataFrame(rows)


def users_needed(eps_list=(0.5, 1, 2, 4, 8), eta=ETA, beta=BETA):
    """Users needed (total, balanced) for a PERFECT attack to refute eps: TV route and one-run route (one run)."""
    rows = []
    for e in eps_list:
        gap = (1 - adv_from_eps(e)) / 2                 # need 2 * dkw_eps(n/2) <= 1 - tanh(eps/2)
        n_tv = 2 * math.log(4 / eta) / (2 * gap**2)
        n_1r = math.log(1 / beta) / math.log(1 + math.exp(-e))   # p_eps^n < beta
        rows.append({"eps": e, "adv_at_eps": adv_from_eps(e), "users_tv": math.ceil(n_tv), "users_1run": math.ceil(n_1r)})
    return pd.DataFrame(rows)


def summarize(bd):
    cols = ["cert", "eps_tv_d0", "eps_tv_d1e-05", "eps_1run", "eps_1run_oracle"]
    return bd.groupby(["run", "stat", "users", "m"])[cols].agg(["mean", "min", "max"]).round(3)
