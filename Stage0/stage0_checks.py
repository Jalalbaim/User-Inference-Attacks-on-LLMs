#!/usr/bin/env python3
"""
stage0_checks.py -- the Stage-0 questions that ui_toy.py does not answer directly.

Same hierarchical Gaussian world as ui_toy.py (imported, not copied):
    mu_u ~ N(0, sigma_mu^2 I_d),  x | u ~ N(mu_u, sigma^2 I_d),
    p_theta = sum_u alpha_u N(muhat_u, sigma^2 I),  p_ref = P_pop = N(0, (sigma_mu^2+sigma^2) I).
Game B given theta: member target = uniform member, non-member = fresh user, m fresh docs.

  Q1  Claim 1 (strict suboptimality of T-hat for m >= 2), with the FEASIBLE Lambda_m:
        Adv*_B(theta)   = P1(Lambda_m > 0) - P0(Lambda_m > 0)        (exact LLR, unbiased MC)
        J(T-hat)        in-sample best threshold (biased UP, so the comparison is conservative)
                        and 2-fold cross-fitted threshold
        feasible        log sum_u alpha_u prod_i N(x_i; muhat_u, sigma^2) - sum_i log p_ref(x_i)
                        (model components, product of per-document references: no population prior)
  Q2  Proposition 6: Var(S_m|b) = v_B + v_W/m (exact), eq. (3) vs measured AUROC, ceiling;
      plus a pilot of plan Step F at Enron scale (n1 = n0 in {34, 68} users, 32 held-out docs).
  Q3  Corollary 2: per member, tau_u = own + typicality, the sandwich
        alpha_u KL(D_u||D_-u) <= own <= log(1 + alpha_u chi2(D_u||D_-u)),
      and per-user TPR(m) -> 1{tau_u > t_gamma} as m grows, for T-hat and the oracle LLR.

Run   python stage0_checks.py all            (~10-15 min on a laptop CPU)
      python stage0_checks.py q1 q2 --quick  (smoke test)
      python stage0_checks.py figs summary   (re-render from results/stage0_checks_*.json)
Outputs go to $UIA_OUT_DIR/results and $UIA_OUT_DIR/figs (default: this folder).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.special import logsumexp, ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ui_toy as T  # noqa: E402

RES, FIG = T.RES, T.FIG
M_GRID = (1, 2, 4, 8, 16, 32)


def _ms(x):
    x = np.asarray(x, float)
    return x.mean(0), x.std(0, ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.zeros(x.shape[1:])


# =============================================================================
# Q1. Claim 1 + feasible Lambda_m
# =============================================================================
Q1_CONFIGS = [  # (label, d, sigma_mu, n, k); k = inf is Assumption 2 (C2: p_theta = D_task)
    ("base d2 smu3 n50", 2, 3.0, 50, np.inf),
    ("d2 smu1 n50", 2, 1.0, 50, np.inf),
    ("d1 smu3 n50", 1, 3.0, 50, np.inf),
    ("d8 smu1 n50", 8, 1.0, 50, np.inf),
    ("d2 smu3 n10", 2, 3.0, 10, np.inf),
    ("d2 smu3 n50 k16 (C2 off)", 2, 3.0, 50, 16.0),
]


def q1_scores(model, X):
    """T-hat, oracle LLR, feasible Lambda for every m in M_GRID. X: (R, max m, d)."""
    W = model.W
    d = W.d
    lp = T.per_doc_logp(model, X)
    x2 = (X**2).sum(-1)
    lref = T.log_iso(x2, W.s2, d)
    cs, cl = np.cumsum(lp - lref, 1), np.cumsum(lref, 1)
    cx, cx2 = np.cumsum(X, 1), np.cumsum(x2, 1)
    out = {"that": [], "oracle": [], "feas": []}
    for m in M_GRID:
        xbar = cx[:, m - 1] / m
        scatter = cx2[:, m - 1] - m * (xbar**2).sum(1)
        out["that"].append(cs[:, m - 1] / m)
        out["oracle"].append(T.lam_opt(model, xbar, m))
        # log prod_i N(x_i; mu, s^2) = log N(xbar; mu, s^2/m) + (d/2) log(2 pi s^2/m)
        #                              - scatter/(2 s^2) - (m d/2) log(2 pi s^2)
        s2 = W.sigma**2
        num = (T.log_mixture(xbar, model.mu_hat, np.log(model.alpha), s2 / m)
               + 0.5 * d * math.log(2 * math.pi * s2 / m) - scatter / (2 * s2)
               - 0.5 * m * d * math.log(2 * math.pi * s2))
        out["feas"].append((num - cl[:, m - 1]) / m)
    return {k: np.stack(v, 1) for k, v in out.items()}


def run_q1(B=30, R=4000, seed=10):
    rng = np.random.default_rng(seed)
    res = {"m_grid": list(M_GRID), "B": B, "R": R, "configs": []}
    for label, d, smu, n, k in Q1_CONFIGS:
        t0 = time.time()
        W = T.World(d, 1.0, smu)
        acc = {key: [] for key in ("adv_star", "that_Jin", "that_Jcf", "feas_Jin", "feas_Jcf",
                                   "oracle_Jin", "that_auc", "feas_auc", "oracle_auc", "m1_feas_eq_that")}
        for _ in range(B):
            model = T.draw_model(rng, W, n, k=k)
            Xi, _, _ = T.draw_targets(rng, model, R, max(M_GRID), True)
            Xo, _, _ = T.draw_targets(rng, model, R, max(M_GRID), False)
            Si, So = q1_scores(model, Xi), q1_scores(model, Xo)
            acc["adv_star"].append((Si["oracle"] > 0).mean(0) - (So["oracle"] > 0).mean(0))
            acc["m1_feas_eq_that"].append(float(np.abs(Si["feas"][:, 0] - Si["that"][:, 0]).max()))
            for s in ("that", "feas", "oracle"):
                jin, jcf, auc = [], [], []
                for j in range(len(M_GRID)):
                    a, c = Si[s][:, j], So[s][:, j]
                    jin.append(T.ks_adv(a, c)[0])
                    auc.append(T.auroc(a, c))
                    if s != "oracle":
                        jcf.append(T.crossfit(a, c, rng)[0])
                acc[f"{s}_Jin"].append(jin)
                acc[f"{s}_auc"].append(auc)
                if s != "oracle":
                    acc[f"{s}_Jcf"].append(jcf)
        row = {"label": label, "d": d, "sigma_mu": smu, "n": n, "k": None if np.isinf(k) else k,
               "assumption2": bool(np.isinf(k)),
               "max_abs_feas_minus_that_m1": float(max(acc.pop("m1_feas_eq_that")))}
        for key, v in acc.items():
            mu, se = _ms(v)
            row[key], row[key + "_se"] = mu.tolist(), se.tolist()
        # paired, conservative gap: Adv* - J_in(T-hat) per model
        gap = np.asarray(acc["adv_star"]) - np.asarray(acc["that_Jin"])
        mu, se = _ms(gap)
        row["gap_conservative"], row["gap_conservative_se"] = mu.tolist(), se.tolist()
        row["gap_frac_models_positive"] = (gap > 0).mean(0).tolist()
        res["configs"].append(row)
        print(f"  [q1] {label}: {time.time() - t0:.0f}s", flush=True)
    return res


# =============================================================================
# Q2. Proposition 6 + Step-F pilot
# =============================================================================
Q2_CONFIGS = [
    ("base d2 smu3 n50", 2, 3.0, 50, np.inf),
    ("d2 smu1 n50", 2, 1.0, 50, np.inf),
    ("d2 smu3 n50 k16 (C2 off)", 2, 3.0, 50, 16.0),
]


def user_moments(model, mu_t, Z, wz):
    """tau_u = E[s|u] and Var(s|u) by Gauss-Hermite, s = log p_theta - log p_ref."""
    W = model.W
    Xq = mu_t[:, None, :] + W.sigma * Z[None]
    s = T.per_doc_logp(model, Xq) - T.log_iso((Xq**2).sum(-1), W.s2, W.d)
    tau = s @ wz
    return tau, np.maximum((s**2) @ wz - tau**2, 0.0)


def pop_quantities(model, rng, Z, wz, n_fresh):
    tau1, vw1 = user_moments(model, model.mu, Z, wz)           # members: exact (uniform targets)
    mu0 = model.W.sigma_mu * rng.standard_normal((n_fresh, model.W.d))
    tau0, vw0 = user_moments(model, mu0, Z, wz)
    return dict(tau1=tau1, tau0=tau0, vB1=tau1.var(), vB0=tau0.var(), vW1=vw1.mean(), vW0=vw0.mean(),
                delta=tau1.mean() - tau0.mean())


def eq3(delta, vB1, vB0, vW1, vW0, m):
    return float(ndtr(delta / math.sqrt(vB1 + vB0 + (vW1 + vW0) / m)))


def run_q2(B=20, R=5000, n_fresh=4000, seed=20, pilot_reps=300):
    rng = np.random.default_rng(seed)
    Z, wz = T.gh_nodes(24, 2)
    res = {"m_grid": list(M_GRID), "B": B, "R": R, "configs": []}
    for label, d, smu, n, k in Q2_CONFIGS:
        t0 = time.time()
        W = T.World(d, 1.0, smu)
        acc = {key: [] for key in ("auc_meas", "auc_pred", "var_ratio1", "var_ratio0",
                                   "ceil_exact", "ceil_pred")}
        for _ in range(B):
            model = T.draw_model(rng, W, n, k=k)
            P = pop_quantities(model, rng, Z, wz, n_fresh)
            Xi, _, _ = T.draw_targets(rng, model, R, max(M_GRID), True)
            Xo, _, _ = T.draw_targets(rng, model, R, max(M_GRID), False)
            si = q1_scores(model, Xi)["that"]
            so = q1_scores(model, Xo)["that"]
            acc["auc_meas"].append([T.auroc(si[:, j], so[:, j]) for j in range(len(M_GRID))])
            acc["auc_pred"].append([eq3(P["delta"], P["vB1"], P["vB0"], P["vW1"], P["vW0"], m) for m in M_GRID])
            acc["var_ratio1"].append([si[:, j].var(ddof=1) / (P["vB1"] + P["vW1"] / m) for j, m in enumerate(M_GRID)])
            acc["var_ratio0"].append([so[:, j].var(ddof=1) / (P["vB0"] + P["vW0"] / m) for j, m in enumerate(M_GRID)])
            acc["ceil_exact"].append(T.auroc(P["tau1"], P["tau0"]))
            acc["ceil_pred"].append(float(ndtr(P["delta"] / math.sqrt(P["vB1"] + P["vB0"]))))
        row = {"label": label, "d": d, "sigma_mu": smu, "n": n, "k": None if np.isinf(k) else k}
        for key, v in acc.items():
            v = np.asarray(v, float)
            mu, se = _ms(v if v.ndim > 1 else v[:, None])
            row[key], row[key + "_se"] = mu.tolist(), se.tolist()
        res["configs"].append(row)
        print(f"  [q2] {label}: {time.time() - t0:.0f}s", flush=True)
    res["pilot"] = run_stepF_pilot(rng, Z, wz, reps=pilot_reps)
    return res


def run_stepF_pilot(rng, Z, wz, n_members=68, K=32, n_models=4, reps=300):
    """Plan Step F with Enron-like counts: does the plug-in prediction land near the truth?"""
    W = T.World(2, 1.0, 3.0)
    out = {"n_members": n_members, "K_heldout": K, "models": []}
    for _ in range(n_models):
        model = T.draw_model(rng, W, n_members)
        P = pop_quantities(model, rng, Z, wz, 4000)
        R = 6000
        Xi, _, _ = T.draw_targets(rng, model, R, max(M_GRID), True)
        Xo, _, _ = T.draw_targets(rng, model, R, max(M_GRID), False)
        si, so = q1_scores(model, Xi)["that"], q1_scores(model, Xo)["that"]
        truth = {"auc_meas": [T.auroc(si[:, j], so[:, j]) for j in range(len(M_GRID))],
                 "ceil_exact": T.auroc(P["tau1"], P["tau0"]),
                 "eq3": [eq3(P["delta"], P["vB1"], P["vB0"], P["vW1"], P["vW0"], m) for m in M_GRID]}
        est = {}
        for n_cls in (34, 68):
            pred_m, pred_c, meas_small = [], [], []
            for _ in range(reps):
                users = rng.permutation(n_members)[:n_cls]
                X1 = model.mu[users][:, None, :] + W.sigma * rng.standard_normal((n_cls, K, W.d))
                mu0 = W.sigma_mu * rng.standard_normal((n_cls, W.d))
                X0 = mu0[:, None, :] + W.sigma * rng.standard_normal((n_cls, K, W.d))
                f = lambda X: T.per_doc_logp(model, X) - T.log_iso((X**2).sum(-1), W.s2, W.d)
                s1, s0 = f(X1), f(X0)
                th1, th0 = s1.mean(1), s0.mean(1)
                vw1, vw0 = s1.var(1, ddof=1), s0.var(1, ddof=1)
                vB1 = max(0.0, th1.var(ddof=1) - (vw1 / K).mean())
                vB0 = max(0.0, th0.var(ddof=1) - (vw0 / K).mean())
                dl = th1.mean() - th0.mean()
                pred_m.append([eq3(dl, vB1, vB0, vw1.mean(), vw0.mean(), m) for m in M_GRID])
                pred_c.append(float(ndtr(dl / math.sqrt(max(vB1 + vB0, 1e-12)))))
                # what Stage 2 would MEASURE at this scale (one subset draw per user)
                meas_small.append([T.auroc(s1[:, :m].mean(1), s0[:, :m].mean(1)) for m in M_GRID])
            q = lambda a: {"mean": np.mean(a, 0).tolist(), "q05": np.quantile(a, 0.05, 0).tolist(),
                           "q95": np.quantile(a, 0.95, 0).tolist()}
            est[str(n_cls)] = {"pred_auc": q(pred_m), "pred_ceil": q(pred_c), "meas_auc": q(meas_small)}
        out["models"].append({"truth": truth, "est": est})
    print("  [q2] Step-F pilot done", flush=True)
    return out


# =============================================================================
# Q3. Corollary 2
# =============================================================================
Q3_CONFIGS = [("base d2 smu3", 2, 3.0), ("overlap d2 smu1", 2, 1.0)]
Q3_M = (1, 4, 16, 64, 256)
K_EDGES = (2, 8, 32, 128, 513)


def run_q3(B=8, n=50, Ru=200, R0=10000, gamma=0.05, seed=30):
    rng = np.random.default_rng(seed)
    Zm, wm = T.gh_nodes(30, 2)
    Z0, w0 = T.gh_nodes(16, 2)
    res = {"m_grid": list(Q3_M), "k_edges": list(K_EDGES), "gamma": gamma, "n": n, "configs": []}
    for label, d, smu in Q3_CONFIGS:
        t0 = time.time()
        W = T.World(d, 1.0, smu)
        users, tpr_that, tpr_orac = [], [], []
        for _ in range(B):
            kvec = np.floor(2.0 ** rng.uniform(1, 9, n))
            model = T.draw_model(rng, W, n, kvec=kvec, exact=True)     # Assumption 2: C2, C3 hold
            al, mu = model.alpha, model.mu
            # ---- non-member tau law and its (1-gamma) quantile ----
            mu0 = smu * rng.standard_normal((R0, d))
            X0q = mu0[:, None, :] + W.sigma * Z0[None]
            tau0 = (T.per_doc_logp(model, X0q) - T.log_iso((X0q**2).sum(-1), W.s2, d)) @ w0
            t_gam = float(np.quantile(tau0, 1 - gamma))
            # ---- per-member decomposition by Gauss-Hermite over D_u ----
            for u in range(n):
                x = mu[u] + W.sigma * Zm
                oth = np.arange(n) != u
                lDu = T.log_iso(((x - mu[u]) ** 2).sum(1), W.sigma**2, d)
                lTask = T.log_mixture(x, mu, np.log(al), W.sigma**2)
                lMinus = T.log_mixture(x, mu[oth], np.log(al[oth] / (1 - al[u])), W.sigma**2)
                lPop = T.log_iso((x**2).sum(1), W.s2, d)
                tau, own, typ = wm @ (lTask - lPop), wm @ (lTask - lMinus), wm @ (lMinus - lPop)
                kl = wm @ (lDu - lMinus)
                log1chi = float(logsumexp(np.log(wm) + lDu - lMinus))      # log(1 + chi2)
                upper = float(np.logaddexp(math.log(al[u]) + log1chi, math.log1p(-al[u])))
                users.append(dict(k=float(kvec[u]), alpha=float(al[u]), tau=float(tau), own=float(own),
                                  typ=float(typ), lower=float(al[u] * kl), upper=upper, t_gamma=t_gam,
                                  limit=int(tau > t_gam), never=int(upper + typ < t_gam)))
            # ---- per-user TPR(m), threshold at FPR gamma from non-member scores ----
            mmax = max(Q3_M)
            nn = min(4000, R0)
            Xn = mu0[:nn, None, :] + W.sigma * rng.standard_normal((nn, mmax, d))
            ln = T.per_doc_logp(model, Xn) - T.log_iso((Xn**2).sum(-1), W.s2, d)
            thr_t = [np.quantile(ln[:, :m].mean(1), 1 - gamma) for m in Q3_M]
            thr_o = [np.quantile(T.lam_opt(model, Xn[:, :m].mean(1), m), 1 - gamma) for m in Q3_M]
            for u in range(n):
                Xu = mu[u] + W.sigma * rng.standard_normal((Ru, mmax, d))
                lu = T.per_doc_logp(model, Xu) - T.log_iso((Xu**2).sum(-1), W.s2, d)
                tpr_that.append([(lu[:, :m].mean(1) > thr_t[j]).mean() for j, m in enumerate(Q3_M)])
                tpr_orac.append([(T.lam_opt(model, Xu[:, :m].mean(1), m) > thr_o[j]).mean()
                                 for j, m in enumerate(Q3_M)])
        U = {key: np.array([r[key] for r in users]) for key in users[0]}
        tt, to = np.array(tpr_that), np.array(tpr_orac)
        tol = 1e-6
        row = {"label": label, "d": d, "sigma_mu": smu, "n_users": len(users),
               "identity_max_abs": float(np.abs(U["tau"] - U["own"] - U["typ"]).max()),
               "sandwich_lower_violations": int((U["lower"] > U["own"] + tol).sum()),
               "sandwich_upper_violations": int((U["own"] > U["upper"] + tol).sum()),
               "never_count": int(U["never"].sum()),
               "limit_count": int(U["limit"].sum()),
               "abs_dev_from_limit": np.abs(tt - U["limit"][:, None]).mean(0).tolist(),
               "never_users_tpr_that": tt[U["never"] == 1].mean(0).tolist() if U["never"].any() else None,
               "never_users_tpr_that_max": tt[U["never"] == 1].max(0).tolist() if U["never"].any() else None,
               "users": {key: v.tolist() for key, v in U.items()}}
        bins = np.searchsorted(np.asarray(K_EDGES), U["k"], side="right") - 1
        row["bin_count"] = np.bincount(bins, minlength=len(K_EDGES) - 1).tolist()
        row["bin_tpr_that"] = [tt[bins == b].mean(0).tolist() for b in range(len(K_EDGES) - 1)]
        row["bin_tpr_oracle"] = [to[bins == b].mean(0).tolist() for b in range(len(K_EDGES) - 1)]
        row["bin_limit_frac"] = [float(U["limit"][bins == b].mean()) for b in range(len(K_EDGES) - 1)]
        res["configs"].append(row)
        print(f"  [q3] {label}: {time.time() - t0:.0f}s", flush=True)
    return res


# =============================================================================
# Figures and summary
# =============================================================================
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3, fixed order


def _style(ax, title):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


KW = dict(lw=2, ms=6)


def _plt():
    import matplotlib.pyplot as plt
    return plt

def fig_q1(r):
    plt, kw = _plt(), KW
    cf = r["configs"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5), facecolor="#fcfcfb")
    for ax, c in zip(axs.ravel(), cf):
        m = r["m_grid"]
        ax.errorbar(m, c["adv_star"], c["adv_star_se"], color=C3, marker="s", label="Adv*_B (oracle LLR)", **kw)
        ax.errorbar(m, c["feas_Jcf"], c["feas_Jcf_se"], color=C2, marker="^", label="J(feasible Λ_m), cross-fit", **kw)
        ax.errorbar(m, c["that_Jin"], c["that_Jin_se"], color=C1, marker="o", label="J(T̂), in-sample", **kw)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("m (documents per target)", fontsize=8, color=INK2)
        _style(ax, c["label"])
    axs[0, 0].set_ylabel("advantage", fontsize=8, color=INK2)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Q1  Claim 1: J(T̂) vs Adv*_B(θ) vs feasible Λ_m (mean ± SE over models)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_q2(r):
    plt, kw = _plt(), KW
    cf = r["configs"]
    fig, axs = plt.subplots(1, len(cf) + 1, figsize=(4 * (len(cf) + 1), 3.6), facecolor="#fcfcfb")
    m = r["m_grid"]
    for ax, c in zip(axs, cf):
        ax.errorbar(m, c["auc_meas"], c["auc_meas_se"], color=C1, marker="o", label="measured AUROC(T̂)", **kw)
        ax.plot(m, c["auc_pred"], color=C2, marker="^", label="eq. (3) prediction", **kw)
        ax.axhline(c["ceil_exact"][0], color=C1, ls="--", lw=1.5, label="exact ceiling AUROC(τ)")
        ax.axhline(c["ceil_pred"][0], color=C2, ls=":", lw=1.5, label="predicted ceiling")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("m", fontsize=8, color=INK2)
        _style(ax, c["label"])
    axs[0].legend(fontsize=7, frameon=False)
    ax = axs[-1]
    p = r["pilot"]["models"][0]
    ax.plot(m, p["truth"]["auc_meas"], color=INK, marker="o", lw=2, label="truth (model 1)")
    for n_cls, col, mk in (("68", C1, "s"), ("34", C2, "^")):
        e = p["est"][n_cls]["pred_auc"]
        ax.fill_between(m, e["q05"], e["q95"], color=col, alpha=0.18, lw=0)
        ax.plot(m, e["mean"], color=col, marker=mk, lw=2, label=f"Step-F prediction, {n_cls}/class (90% band)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "Step-F pilot, 32 held-out docs/user")
    fig.suptitle("Q2  Proposition 6: measured vs predicted AUROC of mean aggregation", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_q3(r):
    plt, kw = _plt(), KW
    cf = r["configs"]
    fig, axs = plt.subplots(2, len(cf), figsize=(5.2 * len(cf), 7.4), facecolor="#fcfcfb")
    m = r["m_grid"]
    for j, c in enumerate(cf):
        U = c["users"]
        ax = axs[0, j]
        a = np.asarray(U["alpha"])
        ax.scatter(a, np.maximum(U["upper"], 1e-12), s=10, color=C2, marker="^", label="log(1+α χ²) (upper)")
        ax.scatter(a, np.maximum(U["own"], 1e-12), s=10, color=C1, marker="o", label="own signal")
        ax.scatter(a, np.maximum(U["lower"], 1e-12), s=10, color=C3, marker="s", label="α KL (lower)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("α_u (share of fine-tuning docs)", fontsize=8, color=INK2)
        _style(ax, f"{c['label']}: sandwich per member")
        ax = axs[1, j]
        edges = r["k_edges"]
        for b, (col, mk) in enumerate(zip((C1, C2, C3, "#eda100"), ("o", "^", "s", "D"))):
            if c["bin_count"][b] == 0:
                continue
            ax.plot(m, c["bin_tpr_that"][b], color=col, marker=mk, lw=2,
                    label=f"T̂, k∈[{edges[b]},{edges[b + 1] - 1}] (limit {c['bin_limit_frac'][b]:.2f})")
            ax.plot(m, c["bin_tpr_oracle"][b], color=col, marker=mk, lw=1.2, ls="--")
        ax.axhline(r["gamma"], color=INK2, lw=1, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("m   (solid: T̂, dashed: oracle LLR, dotted: FPR γ)", fontsize=8, color=INK2)
        ax.set_ylabel("per-user TPR at FPR γ", fontsize=8, color=INK2)
        _style(ax, f"{c['label']}: TPR by contribution size")
        ax.legend(fontsize=6.5, frameon=False)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Q3  Corollary 2: sandwich and light users", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def make_figs(show=False):
    """Render every figure whose result file exists; save PNGs to FIG. Returns {name: figure}."""
    plt = _plt()
    os.makedirs(FIG, exist_ok=True)
    figs = {}
    for name, fn, png in (("stage0_q1", fig_q1, "stage0_q1_claim1.png"),
                          ("stage0_q2", fig_q2, "stage0_q2_prop6.png"),
                          ("stage0_q3", fig_q3, "stage0_q3_cor2.png")):
        path = os.path.join(RES, name + ".json")
        if os.path.exists(path):
            with open(path) as f:
                fig = fn(json.load(f))
            fig.savefig(os.path.join(FIG, png), dpi=150)
            figs[name] = fig
            if not show:
                plt.close(fig)
    print(f"  figures -> {FIG}", flush=True)
    return figs


def _f(v, p=3):
    return "–" if v is None else f"{v:.{p}f}"


def make_summary():
    L = ["# Stage 0 summary (auto-generated by stage0_checks.py)", ""]
    p1 = os.path.join(RES, "stage0_q1.json")
    if os.path.exists(p1):
        r = json.load(open(p1))
        L += ["## Q1 Claim 1: Adv*_B(θ) − J_in(T̂)  (J_in is biased up, so this gap is conservative)", "",
              "| config | A2 | " + " | ".join(f"m={m}" for m in r["m_grid"]) + " | holds for all m≥2 |",
              "|---|---|" + "---|" * len(r["m_grid"]) + "---|"]
        for c in r["configs"]:
            g, se = np.array(c["gap_conservative"]), np.array(c["gap_conservative_se"])
            holds = bool(np.all(g[1:] - 2 * se[1:] > 0))
            L.append(f"| {c['label']} | {'yes' if c['assumption2'] else 'no'} | "
                     + " | ".join(f"{a:+.3f}±{b:.3f}" for a, b in zip(g, se)) + f" | {'YES' if holds else 'no'} |")
        L += ["", "Feasible Λ_m (cross-fitted J) vs T̂ (cross-fitted J), at the largest m:", ""]
        for c in r["configs"]:
            L.append(f"- {c['label']}: J_cf(T̂)={_f(c['that_Jcf'][-1])}, J_cf(feas)={_f(c['feas_Jcf'][-1])}, "
                     f"Adv*={_f(c['adv_star'][-1])}; m=1 check max|feas−T̂|={c['max_abs_feas_minus_that_m1']:.1e}")
        L.append("")
    p2 = os.path.join(RES, "stage0_q2.json")
    if os.path.exists(p2):
        r = json.load(open(p2))
        L += ["## Q2 Proposition 6", "",
              "| config | max abs(Var ratio − 1), b=1 / b=0 | max abs(eq.3 − measured) | ceiling exact / eq.3 |",
              "|---|---|---|---|"]
        for c in r["configs"]:
            v1 = np.max(np.abs(np.array(c["var_ratio1"]) - 1))
            v0 = np.max(np.abs(np.array(c["var_ratio0"]) - 1))
            dev = np.max(np.abs(np.array(c["auc_pred"]) - np.array(c["auc_meas"])))
            L.append(f"| {c['label']} | {v1:.3f} / {v0:.3f} | {dev:.3f} | "
                     f"{c['ceil_exact'][0]:.3f} / {c['ceil_pred'][0]:.3f} |")
        L += ["", "Step-F pilot (n=68 members, 32 held-out docs/user), model 1, m = " + str(r["m_grid"]) + ":", ""]
        p = r["pilot"]["models"][0]
        L.append(f"- truth measured AUROC: {[round(x, 3) for x in p['truth']['auc_meas']]}, "
                 f"exact ceiling {p['truth']['ceil_exact']:.3f}")
        for n_cls in ("68", "34"):
            e = p["est"][n_cls]
            L.append(f"- {n_cls}/class: predicted ceiling mean {e['pred_ceil']['mean']:.3f} "
                     f"[{e['pred_ceil']['q05']:.3f}, {e['pred_ceil']['q95']:.3f}]; "
                     f"measured AUROC at m=32 mean {e['meas_auc']['mean'][-1]:.3f} "
                     f"[{e['meas_auc']['q05'][-1]:.3f}, {e['meas_auc']['q95'][-1]:.3f}]")
        L.append("")
    p3 = os.path.join(RES, "stage0_q3.json")
    if os.path.exists(p3):
        r = json.load(open(p3))
        L += ["## Q3 Corollary 2 (FPR γ = %.2f, m = %s)" % (r["gamma"], r["m_grid"]), ""]
        for c in r["configs"]:
            L += [f"### {c['label']}  ({c['n_users']} member users)",
                  f"- identity τ = own + typicality: max abs error {c['identity_max_abs']:.1e}",
                  f"- sandwich violations: lower {c['sandwich_lower_violations']}, upper {c['sandwich_upper_violations']}",
                  f"- mean |TPR_u(m) − 1{{τ_u > t_γ}}| by m: {[round(x, 3) for x in c['abs_dev_from_limit']]}",
                  f"- users meeting the 'never detected' condition: {c['never_count']}; "
                  f"their mean TPR(T̂) by m: {None if c['never_users_tpr_that'] is None else [round(x, 3) for x in c['never_users_tpr_that']]}; "
                  f"max: {None if c['never_users_tpr_that_max'] is None else [round(x, 3) for x in c['never_users_tpr_that_max']]}",
                  "- per k-bin TPR (T̂ → oracle) at the largest m: "
                  + "; ".join(f"k∈[{r['k_edges'][b]},{r['k_edges'][b + 1] - 1}] n={c['bin_count'][b]}: "
                              f"{_f(c['bin_tpr_that'][b][-1] if c['bin_count'][b] else None, 2)} → "
                              f"{_f(c['bin_tpr_oracle'][b][-1] if c['bin_count'][b] else None, 2)}"
                              for b in range(len(r["k_edges"]) - 1)), ""]
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "stage0_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"  summary -> {path}", flush=True)


# Monte-Carlo budgets. Full-run timings on a laptop CPU (first local run): q1 ~1.2 min.
BUDGETS = {
    "full": {"q1": dict(B=30, R=4000),
             "q2": dict(B=20, R=5000, n_fresh=4000, pilot_reps=300),
             "q3": dict(B=8, Ru=200, R0=10000)},
    "quick": {"q1": dict(B=4, R=1000),
              "q2": dict(B=3, R=1000, n_fresh=1000, pilot_reps=20),
              "q3": dict(B=2, Ru=50, R0=2000)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", help="q1 q2 q3 figs summary | all")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    todo = set(a.what)
    if "all" in todo:
        todo = {"q1", "q2", "q3", "figs", "summary"}
    bud = BUDGETS["quick" if a.quick else "full"]
    os.makedirs(RES, exist_ok=True)
    for name, fn in (("q1", run_q1), ("q2", run_q2), ("q3", run_q3)):
        if name in todo:
            T.save(f"stage0_{name}", fn(**bud[name]))
    if "figs" in todo:
        import matplotlib
        matplotlib.use("Agg")          # CLI only; the notebook keeps its inline backend
        make_figs()
    if "summary" in todo:
        make_summary()


if __name__ == "__main__":
    main()
