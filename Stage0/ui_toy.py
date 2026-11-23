#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui_toy.py -- Gaussian-mixture testbed for user inference.

Companion code for the note "Optimal advantage for user inference".
Setting (after Kandpal et al., "User Inference Attacks on LLMs", EMNLP 2024):

    population   mu_u ~ N(0, sigma_mu^2 I_d)            (a "user")
    user data    x | u ~ D_u = N(mu_u, sigma^2 I_d)      (a "document")
    fine-tuning  n member users, k_u documents each, alpha_u = k_u / sum_v k_v
    target model p_theta(x) = sum_u alpha_u N(x; muhat_u, sigma^2 I)
                 muhat_u = mean of u's k_u training documents (k_u = inf: exact)
    reference    p_ref(x)  = N(x; 0, (sigma_mu^2 + sigma^2) I)   (population predictive)

Population game: a member target is a uniformly drawn member user, a non-member
target is a fresh user from the population; the attacker gets m FRESH documents
from the target (never the training documents) and query access to p_theta.

Scores
    kandpal : (1/m) sum_i log p_theta(x_i)/p_ref(x_i)           (the paper's statistic)
    generic : same, with a broader reference N(0, c (sigma_mu^2+sigma^2) I)
    noref   : (1/m) sum_i log p_theta(x_i)                        (LOSS attack)
    opt     : exact log-likelihood ratio of the attacker's view (Bayes-optimal)

Run   python ui_toy.py all        (all experiments, figures and numbers.tex; about 25 minutes)
      python ui_toy.py selftest   (numerical checks of the closed forms)
      python ui_toy.py E1 figs    (any subset: E1 E2a E2b E2c E3b E5 E6 illus figs numbers)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import ncx2, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("UIA_OUT_DIR", HERE)          # set by kaggle/header.sh; defaults to Stage0/
RES = os.path.join(_OUT, "results")
FIG = os.path.join(_OUT, "figs")


# =============================================================================
# 1. World, densities, models
# =============================================================================
class World:
    """Hierarchical Gaussian population of users."""

    def __init__(self, d=2, sigma=1.0, sigma_mu=3.0):
        self.d, self.sigma, self.sigma_mu = d, float(sigma), float(sigma_mu)
        self.s2 = sigma_mu**2 + sigma**2          # variance of the population predictive
        self.snr = sigma_mu**2 / sigma**2         # between-user / within-user variance


def log_iso(sq, var, d):
    """log N(x; c, var I_d) given the squared distance sq = |x - c|^2."""
    return -0.5 * sq / var - 0.5 * d * np.log(2 * np.pi * var)


def log_mixture(X, means, logw, var, chunk=3_000_000):
    """log sum_j w_j N(x; means_j, var_j I) for each row x of X. var: scalar or (n,)."""
    N, d = X.shape
    n = means.shape[0]
    var = np.broadcast_to(np.asarray(var, dtype=float), (n,))
    const = logw - 0.5 * d * np.log(2 * np.pi * var)
    m2 = (means**2).sum(1)
    out = np.empty(N)
    step = max(1, chunk // n)
    for s in range(0, N, step):
        x = X[s:s + step]
        d2 = (x**2).sum(1)[:, None] - 2.0 * (x @ means.T) + m2[None, :]
        a = -0.5 * d2 / var[None, :] + const[None, :]
        amax = a.max(1)
        out[s:s + step] = amax + np.log(np.exp(a - amax[:, None]).sum(1))
    return out


class Model:
    """Fitted mixture p_theta = sum_u alpha_u N(muhat_u, sigma^2 I)."""

    def __init__(self, W, mu, mu_hat, alpha, kvec):
        self.W, self.mu, self.mu_hat, self.alpha, self.kvec = W, mu, mu_hat, alpha, kvec
        with np.errstate(divide="ignore"):
            noise = W.sigma**2 / kvec                       # var of muhat_u - mu_u (0 if k=inf)
        self.rho = W.sigma_mu**2 / (W.sigma_mu**2 + noise)  # E[mu_u | muhat_u] = rho * muhat_u
        self.s2post = W.sigma_mu**2 * (1.0 - self.rho)      # Var[mu_u | muhat_u]

    @property
    def n(self):
        return len(self.alpha)


def draw_model(rng, W, n, k=np.inf, kvec=None, exact=False):
    """Draw n member users and fit the mixture. k: documents per user (inf = exact)."""
    mu = W.sigma_mu * rng.standard_normal((n, W.d))
    if kvec is None:
        kvec = np.full(n, float(k))
        alpha = np.full(n, 1.0 / n)
    else:
        kvec = np.asarray(kvec, dtype=float)
        alpha = kvec / kvec.sum()
    if exact:                                    # keep the weights, store exact components
        kfit = np.full(n, np.inf)
    else:
        kfit = kvec
    with np.errstate(divide="ignore"):
        sd = W.sigma / np.sqrt(kfit)
    mu_hat = mu + sd[:, None] * rng.standard_normal((n, W.d))
    return Model(W, mu, mu_hat, alpha, kfit)


def draw_targets(rng, model, R, m_max, member):
    """R targets with m_max fresh documents each. Returns X (R, m_max, d), mu_t, J."""
    W = model.W
    if member:
        J = rng.integers(model.n, size=R)        # uniform over member users
        mu_t = model.mu[J]
    else:
        J = np.full(R, -1)
        mu_t = W.sigma_mu * rng.standard_normal((R, W.d))
    X = mu_t[:, None, :] + W.sigma * rng.standard_normal((R, m_max, W.d))
    return X, mu_t, J


def per_doc_logp(model, X):
    """log p_theta(x_i) for every document; X (R, m, d) -> (R, m)."""
    R, m, d = X.shape
    lp = log_mixture(X.reshape(-1, d), model.mu_hat, np.log(model.alpha), model.W.sigma**2)
    return lp.reshape(R, m)


def lam_opt(model, xbar, m):
    """Exact LLR of the view (theta, x_1..x_m) in the population game, uniform targets.

    Lambda = log (1/n) sum_J N(xbar; rho_J muhat_J, (sigma^2/m + s_J^2) I)
             - log N(xbar; 0, (sigma_mu^2 + sigma^2/m) I).
    Terms in the within-sample scatter cancel: xbar is sufficient.
    """
    W = model.W
    logw = np.full(model.n, -math.log(model.n))
    num = log_mixture(xbar, model.rho[:, None] * model.mu_hat, logw,
                      W.sigma**2 / m + model.s2post)
    den = log_iso((xbar**2).sum(1), W.sigma_mu**2 + W.sigma**2 / m, W.d)
    return num - den


# =============================================================================
# 2. Metrics: AUROC, best-threshold advantage (KS), TV between score laws
# =============================================================================
def auroc(s1, s0):
    """P(S1 > S0) + 0.5 P(S1 = S0)  (Mann-Whitney)."""
    r = rankdata(np.concatenate([s1, s0]))
    n1, n0 = len(s1), len(s0)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ks_adv(s1, s0):
    """sup_tau [TPR(tau) - FPR(tau)] for the rule 'member iff S >= tau', and argmax tau."""
    s = np.concatenate([s1, s0])
    y = np.concatenate([np.ones(len(s1)), np.zeros(len(s0))])
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    tp, fp = np.cumsum(y) / len(s1), np.cumsum(1 - y) / len(s0)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]     # end of each tie group
    j = last[np.argmax((tp - fp)[last])]
    return float(tp[j] - fp[j]), float(s[j])


def _inner_edges(s, bins):
    return np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)[1:-1]))


def _hist(s, edges):
    return np.bincount(np.searchsorted(edges, s, side="right"), minlength=len(edges) + 1)


def tv_plugin(s1, s0, bins=40):
    """Histogram plug-in TV between two score samples (biased UPWARD)."""
    e = _inner_edges(np.concatenate([s1, s0]), bins)
    return 0.5 * np.abs(_hist(s1, e) / len(s1) - _hist(s0, e) / len(s0)).sum()


def crossfit(s1, s0, rng, bins=40):
    """2-fold cross-fitted advantages over the available targets (unbiased for a valid
    test given this sample): (a) best single threshold, (b) best union of bins.
    NOTE: if both folds contain documents of the same member users, (b) can pick up
    user-specific score values; use RocAcc for rules that must transfer to unseen users."""
    p1, p0 = rng.permutation(len(s1)), rng.permutation(len(s0))
    h1, h0 = len(s1) // 2, len(s0) // 2
    A = (s1[p1[:h1]], s0[p0[:h0]])
    Bf = (s1[p1[h1:]], s0[p0[h0:]])
    ks = tv = 0.0
    for (a1, a0), (b1, b0) in ((A, Bf), (Bf, A)):
        _, thr = ks_adv(a1, a0)
        ks += (b1 >= thr).mean() - (b0 >= thr).mean()
        e = _inner_edges(np.concatenate([a1, a0]), bins)
        region = _hist(a1, e) / len(a1) > _hist(a0, e) / len(a0)
        tv += (_hist(b1, e) / len(b1) - _hist(b0, e) / len(b0))[region].sum()
    return ks / 2, tv / 2


class RocAcc:
    """Model-averaged ROC of a score whose threshold is calibrated, model by model, to a
    common FPR. Members are binned by their percentile under that model's own non-member
    score law, so the pooled non-member law is uniform. Rules are fitted on one half of
    the MODELS and evaluated on the other half (2-fold), so nothing can be fitted to the
    member users it is evaluated on.

      adv : best common-FPR threshold rule              max_f [mean TPR(f) - f]
      tv  : best union of percentile bins (any rule)    sum_b (h_b - 1/B)_+
    If the member histogram h is non-decreasing (monotone likelihood ratio) they coincide."""

    def __init__(self, bins=50):
        self.bins, self.h = bins, []

    def add(self, s1, s0):
        u = np.searchsorted(np.sort(s0), s1, side="left") / len(s0)
        b = np.minimum((u * self.bins).astype(int), self.bins - 1)
        self.h.append(np.bincount(b, minlength=self.bins) / len(s1))

    def _fold(self, fit, ev):
        B = self.bins
        gain_f = np.cumsum(fit[::-1]) - np.arange(1, B + 1) / B
        k = int(np.argmax(gain_f))
        adv = (np.cumsum(ev[::-1]) - np.arange(1, B + 1) / B)[k]
        tv = (ev - 1.0 / B)[fit > 1.0 / B].sum()
        return adv, tv

    def summary(self, seed=0):
        H = np.asarray(self.h)
        if len(H) < 2:
            adv, tv = self._fold(H[0], H[0])
            return {"adv": float(adv), "tv": float(tv)}
        vals = []
        for r in range(20):                         # average over 20 random half-splits of the models
            perm = np.random.default_rng(seed + r).permutation(len(H))
            A, Bf = perm[:len(H) // 2], perm[len(H) // 2:]
            f1, f2 = self._fold(H[A].mean(0), H[Bf].mean(0)), self._fold(H[Bf].mean(0), H[A].mean(0))
            vals.append([(f1[0] + f2[0]) / 2, (f1[1] + f2[1]) / 2])
        adv, tv = np.mean(vals, 0)
        return {"adv": float(adv), "tv": float(tv)}


def tpr_at_fpr(s1, s0, f):
    thr = np.quantile(s0, 1 - f)
    return float((s1 > thr).mean()), float(thr)


# =============================================================================
# 3. Closed forms, bounds, approximations
# =============================================================================
def chi2_information(m, k, snr, d):
    """chi^2-information between m fresh documents of a user and the model's
    k-document summary of the same user (hierarchical Gaussian): exp(2 I) - 1."""
    if np.isinf(k):
        return (1 + m * snr) ** d - 1
    return ((1 + m * snr) * (1 + k * snr) / (1 + (m + k) * snr)) ** d - 1


def crowd_bound(n, m, k, snr, d):
    """Theorem 4:  Adv* <= (1/2) sqrt( I_chi2 / n )  (uniform targets, equal channels)."""
    return np.minimum(1.0, 0.5 * np.sqrt(chi2_information(m, k, snr, d) / np.asarray(n, float)))


def dense_constant(W, m, k=np.inf, nmc=2_000_000, seed=0):
    """Dense-crowd asymptote Adv* ~ C / sqrt(n),  C = E_{P0} sqrt(v(y)) / sqrt(2 pi),
    v(y) = Var_u[g_u(y)] / p0(y)^2 (CLT over the n i.i.d. member components)."""
    rng = np.random.default_rng(seed)
    d = W.d
    a = W.sigma**2 / m
    s2 = 0.0 if np.isinf(k) else W.sigma_mu**2 * (W.sigma**2 / k) / (W.sigma_mu**2 + W.sigma**2 / k)
    at, smt, c = a + s2, W.sigma_mu**2 - s2, W.sigma_mu**2 + a
    y2 = c * rng.chisquare(d, nmc)
    logv1 = (-0.5 * d * np.log(4 * np.pi * at) + log_iso(y2, smt + at / 2, d) - 2 * log_iso(y2, c, d))
    return float(np.sqrt(np.expm1(logv1)).mean() / math.sqrt(2 * math.pi))


def auroc_envelope(tv):
    """For the LR score: (1+TV)/2 <= AUROC <= 1 - (1-TV)^2/2; binormal reference."""
    tv = np.asarray(tv, float)
    lo, hi = 0.5 * (1 + tv), 1 - 0.5 * (1 - tv) ** 2
    binormal = ndtr(math.sqrt(2) * ndtri(np.clip(0.5 * (1 + tv), 1e-12, 1 - 1e-12)))
    return lo, hi, binormal


def gh_nodes(q, d):
    """Tensor Gauss-Hermite rule for E f(Z), Z ~ N(0, I_d)."""
    t, w = np.polynomial.hermite_e.hermegauss(q)
    w = w / w.sum()
    grids = np.meshgrid(*([t] * d), indexing="ij")
    Z = np.stack([g.ravel() for g in grids], 1)
    wt = np.ones(len(Z))
    for g in np.meshgrid(*([w] * d), indexing="ij"):
        wt = wt * g.ravel()
    return Z, wt


# =============================================================================
# 4. Experiments (population game)
# =============================================================================
SCORES = ("kandpal", "generic", "noref", "opt")


def _score_table(model, X, m_grid, c_generic):
    """All scores for all m in m_grid. Returns dict name -> (R, len(m_grid))."""
    W = model.W
    lp = per_doc_logp(model, X)
    x2 = (X**2).sum(-1)
    per_doc = {
        "kandpal": lp - log_iso(x2, W.s2, W.d),
        "generic": lp - log_iso(x2, c_generic * W.s2, W.d),
        "noref": lp,
    }
    idx = np.asarray(m_grid) - 1
    out = {k: (np.cumsum(v, 1)[:, idx] / np.asarray(m_grid)) for k, v in per_doc.items()}
    cx = np.cumsum(X, 1)
    out["opt"] = np.stack([lam_opt(model, cx[:, m - 1, :] / m, m) for m in m_grid], 1)
    return out, per_doc


def exp_m_sweep(W, n=50, k=np.inf, B=40, R=5000, m_grid=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                c_generic=4.0, seed=0, m_strat=8, plateau=True, tag="base"):
    """E1: effect of the number m of attacker documents (also: distance strata, plateau)."""
    rng = np.random.default_rng(seed)
    M = len(m_grid)
    mets = ("auroc", "ks", "ks_same", "tv_same", "tv_pl", "tpr01", "tpr05")
    res = {s: {mm: np.zeros((B, M)) for mm in mets} for s in SCORES}
    roc = {s: [RocAcc() for _ in range(M)] for s in SCORES}
    adv0 = np.zeros((B, M))                                  # optimal test: Lambda > 0
    edges = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf])      # |mu| / sigma_mu
    strat = {s: {"tp": np.zeros(len(edges) - 1), "fp": np.zeros(len(edges) - 1)} for s in SCORES}
    strat["opt0"] = {"tp": np.zeros(len(edges) - 1), "fp": np.zeros(len(edges) - 1)}
    cnt = {"in": np.zeros(len(edges) - 1), "out": np.zeros(len(edges) - 1)}
    plat = {s: np.zeros(B) for s in ("kandpal", "generic", "noref")}
    varc = {s: np.zeros((B, 6)) for s in ("kandpal", "generic", "noref")}
    Z, wz = gh_nodes(20, W.d)
    js = list(m_grid).index(m_strat)
    max_gap_m1 = 0.0
    t0 = time.time()
    for b in range(B):
        model = draw_model(rng, W, n, k=k)
        Xi, mui, _ = draw_targets(rng, model, R, max(m_grid), True)
        Xo, muo, _ = draw_targets(rng, model, R, max(m_grid), False)
        Si, pdi = _score_table(model, Xi, m_grid, c_generic)
        So, pdo = _score_table(model, Xo, m_grid, c_generic)
        if np.isinf(k):   # exact identity: at m = 1 the paper's statistic IS the optimal LLR
            max_gap_m1 = max(max_gap_m1, float(np.abs(Si["kandpal"][:, 0] - Si["opt"][:, 0]).max()))
        for s in SCORES:
            for j in range(M):
                a, c = Si[s][:, j], So[s][:, j]
                res[s]["auroc"][b, j] = auroc(a, c)
                res[s]["ks"][b, j] = ks_adv(a, c)[0]
                roc[s][j].add(a, c)       # rules calibrated per model to a common FPR
                # rule fitted on (other documents of) the SAME member users: per-model quantities
                res[s]["ks_same"][b, j], res[s]["tv_same"][b, j] = crossfit(a, c, rng)
                res[s]["tv_pl"][b, j] = tv_plugin(a, c)
                res[s]["tpr01"][b, j] = tpr_at_fpr(a, c, 0.01)[0]
                res[s]["tpr05"][b, j] = tpr_at_fpr(a, c, 0.05)[0]
        adv0[b] = (Si["opt"] > 0).mean(0) - (So["opt"] > 0).mean(0)
        # ---- strata by distance from the population centre (rule calibrated at 5% FPR) ----
        bi = np.searchsorted(edges, np.linalg.norm(mui, axis=1) / W.sigma_mu, side="right") - 1
        bo = np.searchsorted(edges, np.linalg.norm(muo, axis=1) / W.sigma_mu, side="right") - 1
        cnt["in"] += np.bincount(bi, minlength=len(edges) - 1)
        cnt["out"] += np.bincount(bo, minlength=len(edges) - 1)
        for s in SCORES:
            thr = np.quantile(So[s][:, js], 0.95)
            strat[s]["tp"] += np.bincount(bi, weights=Si[s][:, js] > thr, minlength=len(edges) - 1)
            strat[s]["fp"] += np.bincount(bo, weights=So[s][:, js] > thr, minlength=len(edges) - 1)
        strat["opt0"]["tp"] += np.bincount(bi, weights=Si["opt"][:, js] > 0, minlength=len(edges) - 1)
        strat["opt0"]["fp"] += np.bincount(bo, weights=So["opt"][:, js] > 0, minlength=len(edges) - 1)
        # ---- m -> infinity plateau of the mean-aggregated scores + variance components ----
        if plateau:
            Rq = min(R, 2000)
            fq, x2q = [], []
            for mu_t in (mui[:Rq], muo[:Rq]):        # E over one document by Gauss-Hermite
                Xq = mu_t[:, None, :] + W.sigma * Z[None, :, :]
                fq.append(per_doc_logp(model, Xq))
                x2q.append((Xq**2).sum(-1))
            for s, cref in (("kandpal", 1.0), ("generic", c_generic), ("noref", None)):
                tau = [(f if cref is None else f - log_iso(x2, cref * W.s2, W.d)) @ wz
                       for f, x2 in zip(fq, x2q)]
                plat[s][b] = auroc(tau[0], tau[1])
                vw_i, vw_o = pdi[s].var(1, ddof=1).mean(), pdo[s].var(1, ddof=1).mean()
                varc[s][b] = (tau[0].mean(), tau[1].mean(), tau[0].var(), tau[1].var(), vw_i, vw_o)
        if b == 0:
            print(f"  [{tag}] first model {time.time() - t0:.1f}s", flush=True)
    out = {"n": n, "k": None if np.isinf(k) else k, "B": B, "R": R, "m_grid": list(m_grid),
           "d": W.d, "sigma": W.sigma, "sigma_mu": W.sigma_mu, "c_generic": c_generic,
           "max_abs_gap_kandpal_vs_opt_m1": max_gap_m1,
           "adv0_mean": adv0.mean(0).tolist(), "adv0_se": (adv0.std(0, ddof=1) / math.sqrt(B)).tolist(),
           "edges": [float(e) if np.isfinite(e) else None for e in edges], "m_strat": m_strat,
           "cnt_in": cnt["in"].tolist(), "cnt_out": cnt["out"].tolist()}
    for s in SCORES:
        for mm in mets:
            out[f"{s}_{mm}_mean"] = res[s][mm].mean(0).tolist()
            out[f"{s}_{mm}_se"] = (res[s][mm].std(0, ddof=1) / math.sqrt(B)).tolist()
        sm = [r_.summary() for r_ in roc[s]]
        for key in ("adv", "tv"):
            out[f"{s}_roc_{key}"] = [x[key] for x in sm]
    for s in strat:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"strat_{s}_tpr"] = (strat[s]["tp"] / cnt["in"]).tolist()
            out[f"strat_{s}_fpr"] = (strat[s]["fp"] / cnt["out"]).tolist()
    if plateau:
        for s in plat:
            out[f"{s}_plateau_mean"] = float(plat[s].mean())
            out[f"{s}_plateau_se"] = float(plat[s].std(ddof=1) / math.sqrt(B))
            v = varc[s].mean(0)
            out[f"{s}_varcomp"] = dict(tau_in=v[0], tau_out=v[1], vB_in=v[2], vB_out=v[3],
                                       vW_in=v[4], vW_out=v[5])
    print(f"  [{tag}] done in {time.time() - t0:.0f}s", flush=True)
    return out


def exp_crowd(W, n_grid=(5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000), m_k=(1, 8),
              m_opt=(1, 8, 64), R=5000, R_opt=20000, seed=1):
    """E2a: crowd size n (equal weights alpha = 1/n)."""
    rng = np.random.default_rng(seed)
    out = {"n_grid": list(n_grid), "m_k": list(m_k), "m_opt": list(m_opt)}
    acc = {f"k_auroc_m{m}": [] for m in m_k}
    acc.update({f"k_ks_m{m}": [] for m in m_k})
    acc.update({f"o_adv_m{m}": [] for m in m_opt})
    acc.update({f"o_auroc_m{m}": [] for m in m_opt})
    se = {key: [] for key in acc}
    t0 = time.time()
    for n in n_grid:
        B = int(np.clip(4000 // n, 16, 200))
        Rk = R if n < 500 else 2 * R          # small advantages need more targets
        tmp = {key: [] for key in acc}
        roc = {m: RocAcc() for m in m_k}
        for b in range(B):
            model = draw_model(rng, W, n)
            Xi, _, _ = draw_targets(rng, model, Rk, max(m_k), True)
            Xo, _, _ = draw_targets(rng, model, Rk, max(m_k), False)
            pdi = per_doc_logp(model, Xi) - log_iso((Xi**2).sum(-1), W.s2, W.d)
            pdo = per_doc_logp(model, Xo) - log_iso((Xo**2).sum(-1), W.s2, W.d)
            for m in m_k:
                a, c = pdi[:, :m].mean(1), pdo[:, :m].mean(1)
                tmp[f"k_auroc_m{m}"].append(auroc(a, c))
                roc[m].add(a, c)
            # optimal test: xbar ~ N(mu_t, sigma^2/m I) is sufficient, so draw it directly
            mui = model.mu[rng.integers(n, size=R_opt)]
            muo = W.sigma_mu * rng.standard_normal((R_opt, W.d))
            for m in m_opt:
                xi = mui + W.sigma / math.sqrt(m) * rng.standard_normal(mui.shape)
                xo = muo + W.sigma / math.sqrt(m) * rng.standard_normal(muo.shape)
                li, lo = lam_opt(model, xi, m), lam_opt(model, xo, m)
                tmp[f"o_adv_m{m}"].append((li > 0).mean() - (lo > 0).mean())
                tmp[f"o_auroc_m{m}"].append(auroc(li, lo))
        for m in m_k:
            acc[f"k_ks_m{m}"].append(roc[m].summary()["adv"])
            se[f"k_ks_m{m}"].append(float("nan"))
        for key in acc:
            if key.startswith("k_ks"):
                continue
            acc[key].append(float(np.mean(tmp[key])))
            se[key].append(float(np.std(tmp[key], ddof=1) / math.sqrt(B)))
        print(f"  [crowd] n={n} B={B} {time.time() - t0:.0f}s", flush=True)
    out.update(acc)
    out.update({key + "_se": v for key, v in se.items()})
    out["bound"] = {str(m): crowd_bound(np.array(n_grid), m, np.inf, W.snr, W.d).tolist() for m in m_opt}
    out["dense_C"] = {str(m): dense_constant(W, m) for m in m_opt}
    return out


def exp_k(W, n=50, k_list=(2, 8, 32, np.inf), m_grid=(1, 2, 4, 8, 16, 32, 64, 128, 256),
          B=24, R=4000, seed=2):
    """E2b: documents per user in the fine-tuning set (what the model stored about u)."""
    rng = np.random.default_rng(seed)
    out = {"k_list": [None if np.isinf(k) else k for k in k_list], "m_grid": list(m_grid), "n": n}
    for k in k_list:
        adv = np.zeros((B, len(m_grid)))
        roc = [RocAcc() for _ in m_grid]
        for b in range(B):
            model = draw_model(rng, W, n, k=k)
            Xi, _, _ = draw_targets(rng, model, R, max(m_grid), True)
            Xo, _, _ = draw_targets(rng, model, R, max(m_grid), False)
            Si, _ = _score_table(model, Xi, m_grid, 4.0)
            So, _ = _score_table(model, Xo, m_grid, 4.0)
            adv[b] = (Si["opt"] > 0).mean(0) - (So["opt"] > 0).mean(0)
            for j in range(len(m_grid)):
                roc[j].add(Si["kandpal"][:, j], So["kandpal"][:, j])
        key = "inf" if np.isinf(k) else str(k)
        out[f"adv_{key}"], out[f"adv_{key}_se"] = adv.mean(0).tolist(), (adv.std(0, ddof=1) / math.sqrt(B)).tolist()
        out[f"kks_{key}"] = [r_.summary()["adv"] for r_ in roc]
        out[f"chi2info_{key}"] = [float(chi2_information(m, k, W.snr, W.d)) for m in m_grid]
        print(f"  [k] k={key} done", flush=True)
    return out


def exp_hetero(W, n=50, kmin=2, kmax=512, m=8, B=80, R=5000, seed=3):
    """E2c: heterogeneous user weights inside one population (stratify TPR by weight)."""
    rng = np.random.default_rng(seed)
    kedges = 2.0 ** np.arange(math.log2(kmin), math.log2(kmax) + 1)   # [2,4),[4,8),...
    nb = len(kedges) - 1
    names = ("kandpal_noisy", "opt_noisy", "kandpal_exact", "opt_exact")
    tp = {s: np.zeros(nb) for s in names}
    cnt = np.zeros(nb)
    glob = {s: [] for s in names}
    for b in range(B):
        kvec = np.floor(2.0 ** rng.uniform(math.log2(kmin), math.log2(kmax), n))
        st = rng.bit_generator.state
        for exact in (False, True):
            rng.bit_generator.state = st            # same users in both variants
            model = draw_model(rng, W, n, kvec=kvec, exact=exact)
            if not exact:
                st2 = rng.bit_generator.state
            rng.bit_generator.state = st2           # same targets and documents
            Xi, _, J = draw_targets(rng, model, R, m, True)
            Xo, _, _ = draw_targets(rng, model, R, m, False)
            bins = np.clip(np.searchsorted(kedges, kvec[J], side="right") - 1, 0, nb - 1)
            if not exact:
                cnt += np.bincount(bins, minlength=nb)
            suf = "exact" if exact else "noisy"
            for nm, fn in (("kandpal", lambda X: (per_doc_logp(model, X)
                                                 - log_iso((X**2).sum(-1), W.s2, W.d)).mean(1)),
                           ("opt", lambda X: lam_opt(model, X.mean(1), m))):
                a, c = fn(Xi), fn(Xo)
                thr = np.quantile(c, 0.95)
                tp[f"{nm}_{suf}"] += np.bincount(bins, weights=a > thr, minlength=nb)
                glob[f"{nm}_{suf}"].append(auroc(a, c))
    out = {"k_edges": kedges.tolist(), "m": m, "n": n, "B": B, "count": cnt.tolist()}
    for s in names:
        p = tp[s] / cnt
        out[f"tpr_{s}"] = p.tolist()
        out[f"tpr_{s}_se"] = np.sqrt(p * (1 - p) / cnt).tolist()
        out[f"auroc_{s}"] = float(np.mean(glob[s]))
    return out


def exp_snr(d=2, sigma_mu_list=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), n=50, m_list=(1, 8), B=40,
            R=5000, seed=4):
    """E3b: population-level distinctiveness of users (mean KL(D_u || P_pop) = I(user; doc))."""
    rng = np.random.default_rng(seed)
    out = {"sigma_mu": list(sigma_mu_list), "m_list": list(m_list), "n": n,
           "mean_kl": [0.5 * d * math.log(1 + s**2) for s in sigma_mu_list]}
    for m in m_list:
        for key in ("k_auroc", "k_ks", "o_adv", "o_auroc"):
            out[f"{key}_m{m}"] = []
    for sm in sigma_mu_list:
        W = World(d, 1.0, sm)
        tmp = {f"{key}_m{m}": [] for m in m_list for key in ("k_auroc", "o_adv", "o_auroc")}
        roc = {m: RocAcc() for m in m_list}
        for b in range(B):
            model = draw_model(rng, W, n)
            Xi, _, _ = draw_targets(rng, model, R, max(m_list), True)
            Xo, _, _ = draw_targets(rng, model, R, max(m_list), False)
            pdi = per_doc_logp(model, Xi) - log_iso((Xi**2).sum(-1), W.s2, W.d)
            pdo = per_doc_logp(model, Xo) - log_iso((Xo**2).sum(-1), W.s2, W.d)
            for m in m_list:
                a, c = pdi[:, :m].mean(1), pdo[:, :m].mean(1)
                li, lo = lam_opt(model, Xi[:, :m].mean(1), m), lam_opt(model, Xo[:, :m].mean(1), m)
                tmp[f"k_auroc_m{m}"].append(auroc(a, c))
                roc[m].add(a, c)
                tmp[f"o_adv_m{m}"].append((li > 0).mean() - (lo > 0).mean())
                tmp[f"o_auroc_m{m}"].append(auroc(li, lo))
        for key, v in tmp.items():
            out[key].append(float(np.mean(v)))
        for m in m_list:
            out[f"k_ks_m{m}"].append(roc[m].summary()["adv"])
        print(f"  [snr] sigma_mu={sm} done", flush=True)
    return out


# =============================================================================
# 5. Two-world (counterfactual) identities in the paper's idealisation
# =============================================================================
def exp_two_world(sigma_mu=3.0, deltas=(3.0, 9.0), alphas=None, alpha_q=0.02, delta_q=3.0,
                  q_grid=(1, 3, 10, 30, 100, 300, 1000, 3000, 10000), reps=20000, seed=5):
    """E5: P = D_u = N(delta e1, I_2), Q = D_{-u} = N(0, s^2 I_2), M_a = a P + (1-a) Q.

    (a) Tbar(a) = E_P log(1 + a (r - 1)) against the four closed-form bounds.
    (b) model-sample channel: TV(M_a^{(x)q}, Q^{(x)q}) against two-sided bounds.
    All one-sample functionals by quadrature on a fine 2-D grid."""
    rng = np.random.default_rng(seed)
    d, sig, s2 = 2, 1.0, sigma_mu**2 + 1.0
    if alphas is None:
        alphas = np.logspace(-4, 0, 41)
    h = 0.02
    g = np.arange(-30, 40 + h, h)
    Y1, Y2 = np.meshgrid(g, g, indexing="ij")
    out = {"alphas": list(map(float, alphas)), "deltas": list(deltas), "s2": s2}

    def one_sample(delta):
        lq = log_iso(Y1**2 + Y2**2, s2, d)
        lpp = log_iso((Y1 - delta) ** 2 + Y2**2, sig**2, d)
        p, q, r = np.exp(lpp), np.exp(lq), np.exp(lpp - lq)
        kl = float((p * (lpp - lq)).sum() * h * h)
        chi2 = float((p * r).sum() * h * h - 1)
        tv = float(0.5 * np.abs(p - q).sum() * h * h)
        return p, q, r, kl, chi2, tv

    for delta in deltas:
        p, q, r, kl, chi2, tv = one_sample(delta)
        lq = log_iso(Y1**2 + Y2**2, s2, d)
        lpp = log_iso((Y1 - delta) ** 2 + Y2**2, sig**2, d)
        kl_cf = 0.5 * (d * sig**2 / s2 + delta**2 / s2 - d + d * math.log(s2 / sig**2))
        chi2_cf = (s2 / (sig * math.sqrt(2 * s2 - sig**2))) ** d * math.exp(delta**2 / (2 * s2 - sig**2)) - 1
        # exact TV between two isotropic Gaussians: {p > q} is a ball
        A = 1 / sig**2 - 1 / s2
        c0 = delta * s2 / (s2 - sig**2)
        rad2 = (d * math.log(s2 / sig**2) + delta**2 / (s2 - sig**2)) / A
        tv_cf = ncx2.cdf(rad2 / sig**2, d, (c0 - delta) ** 2 / sig**2) - ncx2.cdf(rad2 / s2, d, c0**2 / s2)
        with np.errstate(divide="ignore"):       # log(1 + a (r - 1)) in the log domain (a = 1 included)
            tbar = [float((p * np.logaddexp(np.log1p(-a), math.log(a) + lpp - lq)).sum() * h * h)
                    for a in alphas]
        tv_mix = [float(0.5 * np.abs(a * (p - q)).sum() * h * h) for a in alphas]
        key = f"d{delta:g}"
        out[key] = {"kl": kl, "kl_closed": kl_cf, "chi2": chi2, "chi2_closed": chi2_cf, "tv": tv,
                    "tv_closed": float(tv_cf), "tbar": tbar, "tv_mix": tv_mix}
    # ---- (b) sample channel for one (alpha, delta) ----
    p, q, r, kl, chi2, tv = one_sample(delta_q)
    a = alpha_q
    bc = float((q * np.sqrt(1 + a * (r - 1))).sum() * h * h)
    kl_mix = float(((a * p + (1 - a) * q) * np.log1p(a * (r - 1))).sum() * h * h)
    qmax = max(q_grid)
    chunk = 1000
    hits = np.zeros((2, len(q_grid)))
    ratio = []                                  # (L - 1)_+ under H0, L = likelihood ratio

    def llr_cum(member):
        y = math.sqrt(s2) * rng.standard_normal((chunk, qmax, d))
        if member:
            from_p = rng.random((chunk, qmax)) < a
            yp = sig * rng.standard_normal((chunk, qmax, d))
            yp[..., 0] += delta_q
            y = np.where(from_p[..., None], yp, y)
        lr = log_iso((y[..., 0] - delta_q) ** 2 + y[..., 1] ** 2, sig**2, d) - log_iso((y**2).sum(-1), s2, d)
        return np.cumsum(np.log1p(a * np.expm1(lr)), 1)[:, np.asarray(q_grid) - 1]

    for _ in range(reps // chunk):
        hits[1] += (llr_cum(True) > 0).sum(0)
        l0 = llr_cum(False)
        hits[0] += (l0 > 0).sum(0)
        ratio.append(np.maximum(np.expm1(l0), 0.0))
    ratio = np.concatenate(ratio)
    tv_sign = (hits[1] - hits[0]) / reps        # TV = P1(L > 1) - P0(L > 1)
    se_sign = np.sqrt((hits[1] / reps * (1 - hits[1] / reps) + hits[0] / reps * (1 - hits[0] / reps)) / reps)
    tv_ratio, se_ratio = ratio.mean(0), ratio.std(0, ddof=1) / math.sqrt(reps)   # TV = E0 (L - 1)_+
    use_ratio = se_ratio < se_sign
    tvq, tvse = np.where(use_ratio, tv_ratio, tv_sign), np.where(use_ratio, se_ratio, se_sign)
    qg = np.asarray(q_grid, float)
    x = a * a * chi2
    out["channel"] = {
        "alpha": a, "delta": delta_q, "q_grid": list(q_grid), "tv_mc": tvq.tolist(),
        "tv_mc_se": tvse.tolist(), "tv_sign": tv_sign.tolist(), "tv_ratio": tv_ratio.tolist(),
        "exact_q1": a * tv, "chi2_PQ": chi2, "tv_PQ": tv, "bc": bc, "kl_mix": kl_mix,
        "ub_chi2": (0.5 * np.sqrt(np.expm1(qg * math.log1p(x)))).tolist(),
        "ub_bh": np.sqrt(-np.expm1(-qg * math.log1p(x))).tolist(),
        "lb_tv": (-np.expm1(0.5 * qg * math.log1p(-(a * tv) ** 2))).tolist(),
        "lb_bc": (-np.expm1(qg * math.log(bc))).tolist(),
        "ub_bc": np.sqrt(-np.expm1(2 * qg * math.log(bc))).tolist(),
    }
    return out


# =============================================================================
# 6. What an auditor can estimate from finite labelled users
# =============================================================================
def exp_estimation(W, n=50, m=8, big=1_000_000, sizes=(25, 50, 100, 200, 500, 1000, 2000, 5000),
                   reps=400, eta=0.05, seed=6):
    """E6: plug-in TV, KS, cross-fitted lower bounds and the DKW-certified bound as a
    function of the number of labelled users per class; with leakage and under the null."""
    rng = np.random.default_rng(seed)
    model = draw_model(rng, W, n)

    def scores(member, R):
        out = np.empty(R)
        for s in range(0, R, 100_000):
            X, _, _ = draw_targets(rng, model, min(100_000, R - s), m, member)
            out[s:s + X.shape[0]] = (per_doc_logp(model, X) - log_iso((X**2).sum(-1), W.s2, W.d)).mean(1)
        return out

    s1, s0, s0b = scores(True, big), scores(False, big), scores(False, big)
    truth = {"ks": ks_adv(s1, s0)[0], "tv_plugin_400bins": tv_plugin(s1, s0, 400),
             "tv_crossfit_400bins": crossfit(s1, s0, rng, 400)[1], "auroc": auroc(s1, s0)}
    out = {"sizes": list(sizes), "truth": truth, "eta": eta, "m": m, "n": n}
    for label, mem in (("leak", s1), ("null", s0b)):
        rows = {key: [] for key in ("tv_pl", "ks", "ks_cf", "tv_cf", "ks_dkw")}
        cover, false_alarm = [], []
        for na in sizes:
            bins = max(5, int(round(math.sqrt(na))))
            eps = math.sqrt(math.log(4 / eta) / (2 * na))
            tmp = {key: [] for key in rows}
            for _ in range(reps):
                a = mem[rng.integers(big, size=na)]
                c = s0[rng.integers(big, size=na)]
                tmp["tv_pl"].append(tv_plugin(a, c, bins))
                ks = ks_adv(a, c)[0]
                tmp["ks"].append(ks)
                kcf, tcf = crossfit(a, c, rng, max(3, bins // 2))
                tmp["ks_cf"].append(kcf)
                tmp["tv_cf"].append(tcf)
                tmp["ks_dkw"].append(ks - 2 * eps)
            for key in rows:
                v = np.asarray(tmp[key])
                rows[key].append([float(v.mean()), float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))])
            target = truth["ks"] if label == "leak" else 0.0
            cover.append(float((np.asarray(tmp["ks_dkw"]) <= target).mean()))
            false_alarm.append(float((np.asarray(tmp["ks_dkw"]) > 0).mean()))
        out[label] = rows
        out[label + "_dkw_coverage"] = cover
        out[label + "_dkw_positive"] = false_alarm
    return out


def exp_illustration(W, n=50, m=8, R=40000, seed=7):
    """One model draw: geometry and member / non-member score histograms."""
    rng = np.random.default_rng(seed)
    model = draw_model(rng, W, n)
    Xi, mui, J = draw_targets(rng, model, R, m, True)
    Xo, muo, _ = draw_targets(rng, model, R, m, False)
    ki = (per_doc_logp(model, Xi) - log_iso((Xi**2).sum(-1), W.s2, W.d)).mean(1)
    ko = (per_doc_logp(model, Xo) - log_iso((Xo**2).sum(-1), W.s2, W.d)).mean(1)
    li, lo = lam_opt(model, Xi.mean(1), m), lam_opt(model, Xo.mean(1), m)
    g = np.linspace(-11, 11, 221)
    G1, G2 = np.meshgrid(g, g, indexing="ij")
    P = np.stack([G1.ravel(), G2.ravel()], 1)
    ratio = (log_mixture(P, model.mu_hat, np.log(model.alpha), W.sigma**2)
             - log_iso((P**2).sum(1), W.s2, W.d)).reshape(G1.shape)
    return {"mu": model.mu.tolist(), "grid": g.tolist(), "logratio": ratio.tolist(),
            "ex_in": Xi[0].tolist(), "ex_out": Xo[0].tolist(),
            "k_in": ki[:20000].tolist(), "k_out": ko[:20000].tolist(),
            "l_in": li[:20000].tolist(), "l_out": lo[:20000].tolist(),
            "stats": {"k_auroc": auroc(ki, ko), "k_ks": ks_adv(ki, ko)[0],
                      "l_auroc": auroc(li, lo), "l_adv0": float((li > 0).mean() - (lo > 0).mean()),
                      "l_ks": ks_adv(li, lo)[0]}, "m": m, "n": n}


# =============================================================================
# 7. Self-test of the closed forms
# =============================================================================
def selftest():
    rng = np.random.default_rng(123)
    W = World(2, 1.0, 3.0)
    print("== self-test ==")
    # (i) E_theta chi2(P_{1,theta} || P_0) = I_chi2 / n   (Theorem 4, equality step).
    #     chi2 is tail-dominated: plain Monte Carlo only converges when the between-user
    #     spread is small (here sigma_mu = 0.5); for sigma_mu = 3 its variance is infinite.
    Wl = World(2, 1.0, 0.5)
    for (m, k, n) in ((1, np.inf, 5), (1, 4, 5), (2, 8, 3)):
        vals = []
        for _ in range(2000):
            model = draw_model(rng, Wl, n, k=k)
            y = math.sqrt(Wl.sigma_mu**2 + Wl.sigma**2 / m) * rng.standard_normal((20000, Wl.d))
            vals.append(np.mean(np.expm1(lam_opt(model, y, m)) ** 2))
        est, se = np.mean(vals), np.std(vals, ddof=1) / math.sqrt(len(vals))
        print(f" chi2 identity m={m} k={k} n={n}: MC {est:.4f} +- {se:.4f}  closed form "
              f"{chi2_information(m, k, Wl.snr, Wl.d) / n:.4f}")
    #     deterministic check at the base parameters (d = 1): per-user chi2 for a FIXED user
    #     by quadrature in y; the average over users is then an exact Gaussian mgf.
    y = np.arange(-80, 80, 0.002)
    for (m, k, mu) in ((1, np.inf, 2.0), (4, np.inf, 5.0), (4, 8, 5.0)):
        a = 1.0 / m
        s2k = 0.0 if np.isinf(k) else 9.0 * (1.0 / k) / (9.0 + 1.0 / k)
        at, c = a + s2k, 9.0 + a
        val = np.exp(2 * log_iso((y - mu) ** 2, at, 1) - log_iso(y**2, c, 1)).sum() * 0.002
        cf = c / math.sqrt(at * (2 * c - at)) * math.exp(mu**2 / (2 * c - at))
        print(f" per-user 1+chi2, m={m} k={k} mu={mu}: quadrature {val:.5f}  closed form {cf:.5f}")
    # (ii) kandpal == opt at m = 1 (equal weights, exact components, population reference)
    model = draw_model(rng, W, 50)
    X, _, _ = draw_targets(rng, model, 1000, 1, True)
    kd = (per_doc_logp(model, X) - log_iso((X**2).sum(-1), W.s2, W.d))[:, 0]
    print(" max |kandpal - opt| at m=1:", float(np.abs(kd - lam_opt(model, X[:, 0, :], 1)).max()))
    # (iii) Gauss-Hermite plateau vs brute force
    Z, wz = gh_nodes(20, W.d)
    mu_t = model.mu[:5]
    Xq = mu_t[:, None, :] + W.sigma * Z[None]
    gh = (per_doc_logp(model, Xq) - log_iso((Xq**2).sum(-1), W.s2, W.d)) @ wz
    Xb = mu_t[:, None, :] + W.sigma * rng.standard_normal((5, 400000, W.d))
    bf = (per_doc_logp(model, Xb) - log_iso((Xb**2).sum(-1), W.s2, W.d)).mean(1)
    print(" GH-20 vs MC (tau of 5 members):", np.round(gh, 4), np.round(bf, 4))
    # (iv) AUROC envelope for the uniform-shift example  (upper bound is attained)
    u0 = rng.random(400000)
    for J in (0.2, 0.6):
        print(f" uniform shift J={J}: AUROC {auroc(u0[:200000] + J, u0[200000:]):.4f}"
              f"  upper envelope {1 - 0.5 * (1 - J) ** 2:.4f}")


# =============================================================================
# 8. Driver
# =============================================================================
def save(name, obj):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f)


def load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+")
    ap.add_argument("--quick", action="store_true", help="tiny budgets, for smoke tests")
    args = ap.parse_args()
    W = World(2, 1.0, 3.0)
    q = args.quick
    todo = set(args.what)
    if "all" in todo:
        todo = {"E1", "E2a", "E2b", "E2c", "E3b", "E5", "E6", "illus"}
        # ui_figs.py / make_numbers.py belong to the working note and are not in this repo;
        # only run them if present. Stage 0 figures come from stage0_checks.py.
        for extra, mod in (("figs", "ui_figs"), ("numbers", "make_numbers")):
            if os.path.exists(os.path.join(HERE, mod + ".py")):
                todo.add(extra)
    if "selftest" in todo:
        selftest()
    if "E1" in todo:
        save("E1", exp_m_sweep(W, B=4 if q else 40, R=1000 if q else 5000))
    if "E2a" in todo:
        save("E2a", exp_crowd(W, n_grid=(5, 50, 500) if q else (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000),
                              R=1000 if q else 5000, R_opt=2000 if q else 20000))
    if "E2b" in todo:
        save("E2b", exp_k(W, B=3 if q else 24, R=1000 if q else 4000))
    if "E2c" in todo:
        save("E2c", exp_hetero(W, B=6 if q else 80, R=1000 if q else 5000))
    if "E3b" in todo:
        save("E3b", exp_snr(B=3 if q else 40, R=1000 if q else 5000))
    if "E5" in todo:
        save("E5", exp_two_world(reps=2000 if q else 20000))
    if "E6" in todo:
        save("E6", exp_estimation(W, big=100_000 if q else 1_000_000, reps=40 if q else 400))
    if "illus" in todo:
        save("illus", exp_illustration(W))
    if "figs" in todo:
        import ui_figs
        ui_figs.make_all()
    if "numbers" in todo:
        import make_numbers
        make_numbers.main()


if __name__ == "__main__":
    main()
