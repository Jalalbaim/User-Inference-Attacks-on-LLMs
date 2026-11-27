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
from common import certificates as C   # noqa: E402
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


# ================================================================== 4b: cap retraining (Kaggle results)
# Members of seed s are the SAME users as Stage 2's main run of seed s (checked: labels identical), each trained on only
# c docs (nested per-user permutation). Stage 2's uncapped main run is the c = inf point, so every comparison is paired
# by seed and by user. All 175 users are scored on the same 32 held-out docs, with the same m-doc subsets (common.draws).
CAPS_DIR = os.path.join(HERE, "Stage4b")
CAPS = (10, 25, 50, 100)
INF = 10**6   # c = inf (Stage 2, uncapped) in numeric columns


def load_caps(caps_dir=CAPS_DIR):
    """Per-document score tables of every capped model (ckpt/S4/cap<c>_s<seed>/scores.csv.gz; smoke excluded)."""
    import glob
    fs = sorted(glob.glob(os.path.join(caps_dir, "ckpt", "S4", "cap*_s*", "scores.csv.gz")))
    if not fs:
        raise FileNotFoundError(f"no cap score tables under {caps_dir}")
    sc = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in fs], ignore_index=True)
    return sc.drop_duplicates(["run", "seed", "doc_id"]).reset_index(drop=True)


def load_cap_histories(caps_dir=CAPS_DIR):
    """One row per model: cap, seed, best epoch, member/non-member validation loss at the best epoch."""
    import glob
    import json
    rows = []
    for f in sorted(glob.glob(os.path.join(caps_dir, "ckpt", "S4", "cap*_s*", "history.json"))):
        h = json.load(open(f))
        best = h["best"]["epoch"] if isinstance(h["best"], dict) else h["best"]
        e = {r["epoch"]: r for r in h["hist"]}[best]
        tag = os.path.basename(os.path.dirname(f))
        c, s = tag[3:].split("_s")
        rows.append({"cap": int(c), "seed": int(s), "best_epoch": best, "n_train_docs": h.get("n_train_docs"),
                     "val_in": e["val_in"], "val_out": e["val_out"],
                     "val_in_ep0": h["hist"][0]["val_in"], "val_out_ep0": h["hist"][0]["val_out"]})
    return pd.DataFrame(rows)


def _subset_frame(sc):
    """Stage 2's 'cat' layout (order 0 only, s_cat = NaN) built from common.draws, so Stage 2's user_tables can be reused."""
    from common import draws as DR
    held = sc.groupby("user").doc_id.apply(lambda x: sorted(set(x))).to_dict()
    sub = [(u, m, d, DR.SEP.join(sub_ids)) for u, ids in held.items() for m in DR.M_GRID
           for d, sub_ids in enumerate(DR.subsets(ids, u, m, 4))]
    base = pd.DataFrame(sub, columns=["user", "m", "draw", "doc_ids"]).assign(order=0, s_cat=np.nan)
    rs = sc[["run", "seed"]].drop_duplicates()
    return pd.concat([base.assign(run=r, seed=s) for r, s in rs.itertuples(index=False)], ignore_index=True)


def cap_user_values(caps_dir=CAPS_DIR):
    """Per-user T and Agg for every capped model (S_cat not computed in 4b), cached in stage4b/_user_values_caps.csv.gz.
    Columns as Stage 2's user_values plus cap."""
    path = os.path.join(caps_dir, "_user_values_caps.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    sc = load_caps(caps_dir)
    uv = S2.user_values(sc, _subset_frame(sc), blind=False).drop(columns="S_cat")
    uv["cap"] = uv.run.str[3:].astype(int)
    uv.to_csv(path, index=False)
    return uv


def with_uncapped(uvc, uv2):
    """Capped user values + Stage 2's uncapped main run as cap = INF (T and Agg only)."""
    u2 = uv2[uv2.run == "main"].drop(columns=[c for c in ("S_cat", "Blind") if c in uv2]).assign(cap=INF, run="uncapped")
    return pd.concat([uvc, u2[uvc.columns]], ignore_index=True)


CAP_STATS = {"T": ["all", "EVAL"], "Agg": ["EVAL"]}


def cap_metrics(uva, eta=0.05, fprs=(0.05, 0.01)):
    """Per (cap, seed, m, stat, users), mean over draws: AUROC, J-hat, DKW certificate, member TPR at each FPR."""
    rows = []
    for (c, sd, m, dr), g in uva.groupby(["cap", "seed", "m", "draw"]):
        for st, groups in CAP_STATS.items():
            for grp in groups:
                x = g if grp == "all" else g[g.half == "EVAL"]
                s1, s0 = x.loc[x.b == 1, st].to_numpy(float), x.loc[x.b == 0, st].to_numpy(float)
                j = C.youden_j(s1, s0)[0]
                r = {"cap": c, "seed": sd, "m": m, "draw": dr, "stat": st, "users": grp, "n1": len(s1), "n0": len(s0),
                     "auroc": C.auroc(s1, s0), "j_hat": j, "dkw_cert": j - C.dkw_eps(len(s1), eta) - C.dkw_eps(len(s0), eta)}
                for f in fprs:
                    r[f"tpr@{f:g}"] = C.tpr_at_fpr(s1, s0, f)[0]
                rows.append(r)
    per = pd.DataFrame(rows).groupby(["cap", "seed", "m", "stat", "users"]).mean(numeric_only=True)
    return per.drop(columns="draw").reset_index()


def cap_summary(per):
    """Mean / min / max over seeds per (stat, users, m, cap)."""
    cols = ["auroc", "j_hat", "dkw_cert", "tpr@0.05", "tpr@0.01"]
    s = per.groupby(["stat", "users", "m", "cap"])[cols].agg(["mean", "min", "max"])
    s.columns = [f"{a}_{b}" if b != "mean" else a for a, b in s.columns]
    return s.reset_index().assign(seeds=per.groupby(["stat", "users", "m", "cap"]).seed.nunique().to_numpy())


def paired_cap_bootstrap(uva, B=1000, seed=0, base=INF):
    """95% user-bootstrap CIs (percentile), one user resample per replicate shared by every cap, seed, m and draw:
    AUROC per (stat, users, m, cap) averaged over draws then seeds; its difference with the same statistic at
    cap = base (uncapped); and Agg - T-hat (EVAL users) at the same cap. Agg held fixed (conditional on the FIT fit)."""
    users = np.array(sorted(uva.user.unique()))
    pos = {x: i for i, x in enumerate(users)}
    W = np.random.default_rng(seed).multinomial(len(users), np.full(len(users), 1 / len(users)), size=B).astype(float)
    acc = {}
    for (c, sd, m, dr), g in uva.groupby(["cap", "seed", "m", "draw"]):
        for st, groups in CAP_STATS.items():
            for grp in groups:
                x = g if grp == "all" else g[g.half == "EVAL"]
                s_, b_ = x[st].to_numpy(float), x.b.to_numpy()
                v = (C.auroc(s_[b_ == 1], s_[b_ == 0]), S2._boot_auroc(s_, b_, W[:, x.user.map(pos).to_numpy()]))
                acc.setdefault((st, grp, m, c), {}).setdefault(sd, []).append(v)
    est = {k: (np.mean([np.mean([p for p, _ in v]) for v in d.values()]),
               np.mean([np.mean([r for _, r in v], axis=0) for v in d.values()], axis=0)) for k, d in acc.items()}
    q = lambda r: (np.quantile(r, .025), np.quantile(r, .975))
    rows = []
    for (st, grp, m, c), (p, r) in sorted(est.items()):
        row = {"stat": st, "users": grp, "m": m, "cap": c, "auroc": p}
        row["auroc_lo"], row["auroc_hi"] = q(r)
        pb, rb = est[(st, grp, m, base)]
        row["d_uncapped"] = p - pb
        row["d_uncapped_lo"], row["d_uncapped_hi"] = q(r - rb)
        if st == "Agg":
            pt, rt = est[("T", "EVAL", m, c)]
            row["agg_minus_T"] = p - pt
            row["agg_minus_T_lo"], row["agg_minus_T_hi"] = q(r - rt)
        rows.append(row)
    return pd.DataFrame(rows)
