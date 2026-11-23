"""
Certificates and metrics shared by all stages (plan section 2 and Step E).

Every function takes user-level scores: s1 = scores of member users, s0 = scores of
non-member users, one score per user (never several draws of the same user, plan D.5).
Convention: larger score = "member".
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata


def auroc(s1, s0):
    """Mann-Whitney AUROC = P(S1 > S0) + 0.5 P(S1 = S0)."""
    s1, s0 = np.asarray(s1, float), np.asarray(s0, float)
    r = rankdata(np.concatenate([s1, s0]))
    n1, n0 = len(s1), len(s0)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def youden_j(s1, s0):
    """J-hat = max over thresholds t of TPR(t) - FPR(t), rule 'member iff S >= t'.
    Returns (J-hat, argmax threshold). Ties are handled as one threshold group."""
    s1, s0 = np.asarray(s1, float), np.asarray(s0, float)
    s = np.concatenate([s1, s0])
    y = np.concatenate([np.ones(len(s1)), np.zeros(len(s0))])
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    tp, fp = np.cumsum(y) / len(s1), np.cumsum(1 - y) / len(s0)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    j = last[np.argmax((tp - fp)[last])]
    jhat = float(tp[j] - fp[j])
    return max(jhat, 0.0), float(s[j])   # the empty rule (t = +inf) always gives 0


def dkw_eps(n, eta):
    """sqrt(log(4/eta) / (2n)): DKW-Massart half-width, eta/2 per class."""
    return math.sqrt(math.log(4.0 / eta) / (2.0 * n))


def dkw_certificate(s1, s0, eta=0.05):
    """Prop. 2(ii): J(S) >= J-hat - eps(n1) - eps(n0) w.p. >= 1 - eta (all thresholds at once).
    Reported even when negative (plan E.3)."""
    jhat, _ = youden_j(s1, s0)
    return jhat - dkw_eps(len(s1), eta) - dkw_eps(len(s0), eta)


def fixed_test_certificate(s1_eval, s0_eval, threshold, eta=0.05):
    """Prop. 2(i): threshold fixed on OTHER users (FIT half). One-sided Hoeffding, eta/2 per class:
    Adv(phi) >= TPR-hat - FPR-hat - sqrt(log(2/eta)/(2 n1)) - sqrt(log(2/eta)/(2 n0))."""
    s1, s0 = np.asarray(s1_eval, float), np.asarray(s0_eval, float)
    tpr, fpr = float((s1 >= threshold).mean()), float((s0 >= threshold).mean())
    e = lambda n: math.sqrt(math.log(2.0 / eta) / (2.0 * n))
    return tpr - fpr - e(len(s1)) - e(len(s0))


def j_from_auroc(auroc_lcb):
    """Prop. 2(iii): J >= 1 - sqrt(2 (1 - AUROC_lcb)). Valid for any ROC: TPR(f) <= min(1, f + J)
    gives AUROC <= 1 - (1 - J)^2 / 2; attained by that ROC. Vacuous (< 0) for AUROC_lcb < 1/2."""
    return 1.0 - math.sqrt(2.0 * (1.0 - auroc_lcb))


def users_needed(jhat, eta=0.05):
    """Cor. 1: with n users per class, the DKW certificate is positive only if n > 2 log(4/eta) / J^2."""
    return float("inf") if jhat <= 0 else 2.0 * math.log(4.0 / eta) / jhat**2


def tpr_at_fpr(s1, s0, fpr):
    """TPR of the rule 'S > (1 - fpr)-quantile of non-member scores'.
    Below fpr = 1/n0 this is '0 or 1 false positives', not a certified rate (plan 2.2)."""
    s1, s0 = np.asarray(s1, float), np.asarray(s0, float)
    thr = np.quantile(s0, 1 - fpr)
    return float((s1 > thr).mean()), float(thr), bool(fpr < 1.0 / len(s0))


def user_bootstrap(s1, s0, stat, B=2000, alpha=0.05, seed=0):
    """Percentile CI of stat(s1, s0), resampling USERS within each class."""
    rng = np.random.default_rng(seed)
    s1, s0 = np.asarray(s1, float), np.asarray(s0, float)
    v = [stat(s1[rng.integers(len(s1), size=len(s1))], s0[rng.integers(len(s0), size=len(s0))])
         for _ in range(B)]
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def report(s1, s0, eta=0.05):
    """The per-(statistic, m, seed) row of plan Step E (without the FIT/EVAL fixed-test part)."""
    jhat, thr = youden_j(s1, s0)
    a = auroc(s1, s0)
    lo, hi = user_bootstrap(s1, s0, auroc)
    t1, _, t1_uncert = tpr_at_fpr(s1, s0, 1e-2)
    t01, _, t01_uncert = tpr_at_fpr(s1, s0, 1e-3)
    return {"n1": len(s1), "n0": len(s0), "auroc": a, "auroc_ci": [lo, hi], "j_hat": jhat,
            "j_threshold": thr, "dkw_cert": dkw_certificate(s1, s0, eta), "eta": eta,
            "tpr_at_1pct": t1, "tpr_at_1pct_below_1_over_n0": t1_uncert,
            "tpr_at_0.1pct": t01, "tpr_at_0.1pct_below_1_over_n0": t01_uncert,
            "users_needed_per_class": users_needed(jhat, eta)}


if __name__ == "__main__":   # self-check of the plan's numbers
    assert abs(2 * math.log(80) - 8.764) < 1e-3
    assert abs(math.sqrt(2 * math.log(80) / 68) - 0.359) < 1e-3          # Cor. 1 at n = 68
    # smallest integer n with n > 2 log(80) / J^2 (the plan's table rounds 219.1 and 876.4 down)
    for j, n in ((0.6, 25), (0.4, 55), (0.3, 98), (0.2, 220), (0.1, 877)):
        assert math.floor(users_needed(j)) + 1 == n, (j, users_needed(j))
    rng = np.random.default_rng(0)                                         # null: certificate <= 0
    bad = sum(dkw_certificate(rng.random(68), rng.random(68)) > 0 for _ in range(2000))
    assert bad <= 0.05 * 2000, bad
    print("certificates self-check OK; null false-positive rate", bad / 2000)
