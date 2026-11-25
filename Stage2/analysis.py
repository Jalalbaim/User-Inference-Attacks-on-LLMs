"""
Stage 2 analysis (local, no GPU): plan Steps D-F on the downloaded Kaggle outputs.

Statistics, per (run, seed, m, draw), on the SAME m-document subsets (common/draws.py, shared with S_cat):
  T      T-hat = mean s(x) over the subset                                       all users certified
  S_cat  mean over R orders of log p_theta(concat) - log p_ref(concat)          all users certified
  Agg    logistic regression on 11 per-user features of the subset, fit on FIT, scored on EVAL
  Blind  TF-IDF + logistic regression on the subset's text (no model), fit on FIT, scored on EVAL
Metrics: AUROC, J-hat, DKW certificate (Prop. 2 ii); on EVAL also the fixed-test certificate (Prop. 2 i) with the
threshold chosen on FIT (Youden argmax). Metrics are computed per draw, then averaged over draws (plan D.5);
draw 0 is the one whose certificate is a valid single-sample statement.
Headline (pre-registered 2026-11-24): T-hat, m = 32, eta = 0.05, all 175 users, main run, seed 1.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
from common import certificates as C   # noqa: E402
from common import draws as DR         # noqa: E402

HEADLINE = {"stat": "T", "m": 32, "eta": 0.05, "run": "main", "seed": 1, "users": "all"}
FEATURES = ["mean", "sd", "min", "max", "q10", "q25", "q50", "q75", "q90", "mean_logp_theta", "mean_logp_ref"]
STATS = ["T", "S_cat", "Agg", "Blind"]


# ---------------------------------------------------------------- loading
def _read(f):
    return pd.read_csv(f, keep_default_na=False, na_values=[""])   # "null" is a run label, not NaN


def load(results_dir):
    """scores (Stage-C tables), scat (per-unit S_cat tables), hist {run_s<seed>: history}."""
    def grab(pat):
        fs = sorted(glob.glob(os.path.join(results_dir, "**", pat), recursive=True))
        return pd.concat([_read(f) for f in fs]) if fs else None
    sc = grab("scores*.csv.gz")
    if sc is None:
        raise FileNotFoundError(f"no scores*.csv.gz under {results_dir}")
    sc = sc.drop_duplicates(["run", "seed", "doc_id"]).reset_index(drop=True)
    cat = grab("scat_*_s*.csv.gz")
    if cat is not None:
        cat = cat.drop_duplicates(["run", "seed", "user", "m", "draw", "order"]).reset_index(drop=True)
    hist = {}
    for f in glob.glob(os.path.join(results_dir, "**", "history_*.json"), recursive=True):
        hist[os.path.basename(f)[8:-5]] = json.load(open(f))
    return sc, cat, hist


def load_texts():
    from common.enron_data import load_splits
    docs, users, _ = load_splits()
    return {d["doc_id"]: d["text"] for d in docs if d["role"] == "heldout"}, users


# ---------------------------------------------------------------- per-user tables
def user_tables(sc, cat, run, seed, texts=None):
    """{(m, draw): DataFrame indexed by user with b, half, T, S_cat, 11 features, text}."""
    d = sc[(sc.run == run) & (sc.seed == seed)].set_index("doc_id")
    c = cat[(cat.run == run) & (cat.seed == seed)]
    lab = d.groupby("user")[["b", "half", "trained", "k_u"]].first()
    out = {}
    for (m, dr), g in c.groupby(["m", "draw"]):
        scat = g.groupby("user").s_cat.mean()
        subs = g[g.order == 0].set_index("user").doc_ids.str.split(DR.SEP)
        rows = {}
        for u, ids in subs.items():
            x = d.loc[ids]
            s = x.s.to_numpy()
            q = np.quantile(s, [.1, .25, .5, .75, .9])
            rows[u] = {"T": s.mean(), "mean": s.mean(), "sd": s.std(ddof=1) if len(s) > 1 else 0.0,
                       "min": s.min(), "max": s.max(), "q10": q[0], "q25": q[1], "q50": q[2], "q75": q[3],
                       "q90": q[4], "mean_logp_theta": x.logp_theta.mean(), "mean_logp_ref": x.logp_ref.mean()}
            if texts is not None:
                rows[u]["text"] = "\n".join(texts[i] for i in ids)
        t = pd.DataFrame.from_dict(rows, orient="index").join(lab).assign(S_cat=scat)
        out[(int(m), int(dr))] = t
    return out


# ---------------------------------------------------------------- learned statistics (fit on FIT, score EVAL)
def _fit_score(Xf, yf, Xe, kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(set(yf)) < 2:
        return np.full(len(Xe), np.nan), np.full(len(Xf), np.nan)
    if kind == "Agg":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    else:
        clf = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2), LogisticRegression(C=1.0, max_iter=5000))
    clf.fit(Xf, yf)
    return clf.decision_function(Xe), clf.decision_function(Xf)


def add_learned(t, blind=True):
    """Adds columns Agg and Blind (EVAL: out-of-sample; FIT: in-sample, used only to choose a threshold)."""
    t = t.copy()
    f, e = t.half == "FIT", t.half == "EVAL"
    se, sf = _fit_score(t.loc[f, FEATURES].to_numpy(), t.loc[f, "b"].to_numpy(), t.loc[e, FEATURES].to_numpy(), "Agg")
    t.loc[e, "Agg"], t.loc[f, "Agg"] = se, sf
    if blind and "text" in t:
        se, sf = _fit_score(t.loc[f, "text"].tolist(), t.loc[f, "b"].to_numpy(), t.loc[e, "text"].tolist(), "Blind")
        t.loc[e, "Blind"], t.loc[f, "Blind"] = se, sf
    return t


# ---------------------------------------------------------------- metrics
def metrics(t, stat, users, eta=0.05):
    """users = 'all' or 'EVAL'. Returns a dict (AUROC, J-hat, DKW certificate, fixed-test certificate)."""
    e = t if users == "all" else t[t.half == "EVAL"]
    s1, s0 = e.loc[e.b == 1, stat].to_numpy(float), e.loc[e.b == 0, stat].to_numpy(float)
    if len(s1) == 0 or len(s0) == 0 or np.isnan(s1).any() or np.isnan(s0).any():
        return None
    j, _ = C.youden_j(s1, s0)
    r = {"n1": len(s1), "n0": len(s0), "auroc": C.auroc(s1, s0), "j_hat": j,
         "dkw_cert": j - C.dkw_eps(len(s1), eta) - C.dkw_eps(len(s0), eta)}
    if users == "EVAL":   # Prop. 2(i): threshold from FIT users only
        f = t[t.half == "FIT"]
        f1, f0 = f.loc[f.b == 1, stat].to_numpy(float), f.loc[f.b == 0, stat].to_numpy(float)
        if len(f1) and len(f0) and not (np.isnan(f1).any() or np.isnan(f0).any()):
            r["fixed_cert"] = C.fixed_test_certificate(s1, s0, C.youden_j(f1, f0)[1], eta)
    return r


def evaluate(sc, cat, runs_seeds=None, texts=None, eta=0.05, blind=True):
    """Long table: run, seed, m, draw, stat, users, n1, n0, auroc, j_hat, dkw_cert, fixed_cert."""
    if runs_seeds is None:
        runs_seeds = sorted(set(map(tuple, cat[["run", "seed"]].drop_duplicates().to_numpy())))
    rows = []
    for run, seed in runs_seeds:
        for (m, dr), t in user_tables(sc, cat, run, seed, texts).items():
            t = add_learned(t, blind=blind and texts is not None)
            for stat in STATS:
                if stat not in t:
                    continue
                groups = ["EVAL"] if run == "null" or stat in ("Agg", "Blind") else ["all", "EVAL"]
                for g in groups:
                    r = metrics(t, stat, g, eta)
                    if r:
                        rows.append({"run": run, "seed": int(seed), "m": m, "draw": dr, "stat": stat, "users": g, **r})
    return pd.DataFrame(rows)


def summarize(ev):
    """Mean over draws per (run, seed, m, stat, users), then mean / sd / min / max over seeds."""
    per_seed = ev.groupby(["run", "seed", "m", "stat", "users"]).mean(numeric_only=True).drop(columns="draw").reset_index()
    agg = per_seed.groupby(["run", "m", "stat", "users"]).agg(
        seeds=("seed", "nunique"), auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
        j_hat=("j_hat", "mean"), dkw_cert=("dkw_cert", "mean"), dkw_cert_min=("dkw_cert", "min"),
        fixed_cert=("fixed_cert", "mean") if "fixed_cert" in per_seed else ("dkw_cert", "size"),
        n1=("n1", "mean"), n0=("n0", "mean")).reset_index()
    return per_seed, agg


def paired_vs_T(per_seed, users="EVAL"):
    """For each m and statistic: AUROC(stat) - AUROC(T) per seed, mean, and in how many seeds it is > 0.
    Seeds reuse the same users, so the spread over seeds understates the uncertainty over users."""
    p = per_seed[(per_seed.run == "main") & (per_seed.users == users)].pivot_table(
        index=["seed", "m"], columns="stat", values="auroc")
    rows = []
    for st in [s for s in p.columns if s != "T"]:
        d = (p[st] - p["T"]).dropna().groupby(level="m")
        for m, x in d:
            rows.append({"stat": st, "m": m, "mean_diff": x.mean(), "sd": x.std(), "seeds_better": int((x > 0).sum()),
                         "seeds": len(x)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- user bootstrap (plan row 2: user-bootstrap CIs)
def user_values(sc, cat, runs_seeds=None, texts=None, blind=True):
    """Per-user statistic values per (run, seed, m, draw): columns user, b, half, T, S_cat, Agg, Blind."""
    if runs_seeds is None:
        runs_seeds = sorted(set(map(tuple, cat[["run", "seed"]].drop_duplicates().to_numpy())))
    out = []
    for run, seed in runs_seeds:
        for (m, dr), t in user_tables(sc, cat, run, seed, texts).items():
            t = add_learned(t, blind=blind and texts is not None)
            cols = ["b", "half"] + [s for s in STATS if s in t]
            out.append(t[cols].rename_axis("user").reset_index().assign(run=run, seed=int(seed), m=m, draw=dr))
    return pd.concat(out, ignore_index=True)


def _boot_auroc(s, b, W):
    """AUROC for every bootstrap weight row of W (B x n users); ties count 1/2."""
    i1, i0 = np.flatnonzero(b == 1), np.flatnonzero(b == 0)
    P = (s[i1][:, None] > s[i0][None, :]) + 0.5 * (s[i1][:, None] == s[i0][None, :])
    w1, w0 = W[:, i1], W[:, i0]
    return np.einsum("bi,ij,bj->b", w1, P, w0) / (w1.sum(1) * w0.sum(1))


def bootstrap_vs_T(uv, run="main", B=1000, seed=0):
    """95% user-bootstrap CIs (percentile) for the seed- and draw-averaged AUROC of each statistic and for its
    difference with T-hat, per m and user group. One user resample per replicate, shared by every seed, m, draw and
    statistic (users are what all seeds share). Learned statistics are held fixed (conditional on the FIT fit)."""
    u = uv[uv.run == run]
    users = np.array(sorted(u.user.unique()))
    W = np.random.default_rng(seed).multinomial(len(users), np.full(len(users), 1 / len(users)), size=B).astype(float)
    pos = {x: i for i, x in enumerate(users)}
    acc = {}   # (users, m, stat) -> list of (point, B-vector)
    for (sd, m, dr), g in u.groupby(["seed", "m", "draw"]):
        for grp in ("all", "EVAL"):
            x = g if grp == "all" else g[g.half == "EVAL"]
            idx = x.user.map(pos).to_numpy()
            b = x.b.to_numpy()
            for st in STATS:
                if st not in x or x[st].isna().any() or (grp == "all" and st in ("Agg", "Blind")):
                    continue
                s = x[st].to_numpy(float)
                acc.setdefault((grp, m, st), {}).setdefault(sd, []).append(
                    (C.auroc(s[b == 1], s[b == 0]), _boot_auroc(s, b, W[:, idx])))
    # mean over draws within seed, then over seeds
    est = {k: (np.mean([np.mean([p for p, _ in v]) for v in d.values()]),
               np.mean([np.mean([r for _, r in v], axis=0) for v in d.values()], axis=0)) for k, d in acc.items()}
    rows = []
    for (grp, m, st), (p, r) in sorted(est.items()):
        row = {"users": grp, "m": m, "stat": st, "auroc": p, "auroc_lo": np.quantile(r, .025), "auroc_hi": np.quantile(r, .975)}
        if st != "T" and (grp, m, "T") in est:
            pt, rt = est[(grp, m, "T")]
            dlt = r - rt
            row.update(diff=p - pt, diff_lo=np.quantile(dlt, .025), diff_hi=np.quantile(dlt, .975), p_gt0=(dlt > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- headline (pre-registered)
def headline(sc, h=HEADLINE):
    d = sc[(sc.run == h["run"]) & (sc.seed == h["seed"])]
    if d.empty:
        return None
    t = d.groupby("user").agg(b=("b", "first"), tau=("s", "mean"), n=("s", "size"))
    assert (t.n == h["m"]).all(), "headline: every user must contribute all 32 held-out docs"
    r = C.report(t.tau[t.b == 1].to_numpy(), t.tau[t.b == 0].to_numpy(), h["eta"])
    return {**h, **r}


# ---------------------------------------------------------------- Step F: variance decomposition (Prop. 6)
TRANSFORMS = (None, "slog", "int")


def _scores(sc, run, seed, transform=None):
    """Score rows of one model (EVAL users only for the null), with s optionally transformed per document:
    'slog' = sign(s) log(1 + |s|); 'int' = rank-inverse-normal over all documents of the model.
    The transformed variants are EXPLORATORY (added 2026-11-25 after eq. (3) failed on raw s; not pre-registered)."""
    d = sc[(sc.run == run) & (sc.seed == seed)]
    if run == "null":
        d = d[d.half == "EVAL"]
    if transform == "slog":
        d = d.assign(s=np.sign(d.s) * np.log1p(np.abs(d.s)))
    elif transform == "int":
        d = d.assign(s=norm.ppf((d.s.rank(method="average") - 0.5) / len(d)))
    elif transform is not None:
        raise ValueError(transform)
    return d


def variance_decomposition(sc, run="main", seed=0, m_grid=DR.M_GRID, transform=None):
    """Per class: v_W (mean within-user variance of s), v_B (between-user variance of tau minus sampling noise,
    clipped at 0), Delta; predicted AUROC(m) = Phi(Delta / sqrt(vB1 + vB0 + (vW1 + vW0)/m)) and the ceiling."""
    d = _scores(sc, run, seed, transform)
    u = d.groupby("user").agg(b=("b", "first"), tau=("s", "mean"), vw=("s", lambda x: x.var(ddof=1)), k=("s", "size"))
    par = {}
    for b in (0, 1):
        x = u[u.b == b]
        vw = x.vw.mean()
        vb = max(0.0, x.tau.var(ddof=1) - (x.vw / x.k).mean())
        par[b] = {"tau_mean": x.tau.mean(), "v_W": vw, "v_B": vb, "n": len(x)}
    delta = par[1]["tau_mean"] - par[0]["tau_mean"]
    vb, vw = par[1]["v_B"] + par[0]["v_B"], par[1]["v_W"] + par[0]["v_W"]
    pred = {m: float(norm.cdf(delta / np.sqrt(vb + vw / m))) for m in m_grid}
    ceiling = float(norm.cdf(delta / np.sqrt(vb))) if vb > 0 else 1.0
    return {"run": run, "seed": seed, "transform": transform or "raw", "delta": delta, "v_B1": par[1]["v_B"], "v_B0": par[0]["v_B"],
            "v_W1": par[1]["v_W"], "v_W0": par[0]["v_W"], "pred": pred, "ceiling": ceiling,
            "share_between_at_m32": vb / (vb + vw / 32) if vb + vw > 0 else float("nan")}


def measured_T_auroc(sc, run="main", seed=0, R=200, m_grid=DR.M_GRID, rng_seed=0, transform=None):
    """AUROC of the mean (transformed) score per m, averaged over R random subsets (as in Stage 1), for comparison
    with the prediction made on the SAME transformed scores."""
    d = _scores(sc, run, seed, transform)
    rng = np.random.default_rng(rng_seed)
    g = {u: x.s.to_numpy() for u, x in d.groupby("user")}
    lab = d.groupby("user").b.first()
    out = {}
    for m in m_grid:
        reps = 1 if all(len(v) <= m for v in g.values()) else R
        a = []
        for _ in range(reps):
            S = np.array([v[rng.choice(len(v), m, replace=False)].mean() if m < len(v) else v.mean()
                          for v in (g[u] for u in lab.index)])
            a.append(C.auroc(S[lab.to_numpy() == 1], S[lab.to_numpy() == 0]))
        out[m] = float(np.mean(a))
    return out


def history_table(hist):
    rows = []
    for tag, h in sorted(hist.items()):
        for e in h["hist"]:
            rows.append({"run": tag, "epoch": e["epoch"], "train_loss": e["train_loss"], "val_in": e["val_in"],
                         "val_out": e["val_out"], "min": e["secs"] / 60, "best": e["epoch"] == h["best"]["epoch"]})
    return pd.DataFrame(rows)
