# Stage 0: Gaussian testbed

**Plan:** `docs/experiment_plan_enron_gptneo125m.md`, section 4 row 0 (draft App. I). CPU only.
**Gate:** Claim 1 holds numerically. If it does not, rethink the C3 story before any LLM run.
**Status: CLOSED (2026-11-23), gate passed.** Conclusions are in the notes cell at the end of `stage0.ipynb`.

## Files

| File | What it is |
|---|---|
| `stage0.ipynb` | **Main entry.** Runs (or loads cached) each block, shows tables and figures, gives the gate verdict, and ends with a notes cell for the analysis. |
| `ui_toy.py` | Gaussian testbed from the working note "Optimal advantage for user inference" (experiments E1–E6; these E-numbers are the note's, not the paper's). |
| `stage0_checks.py` | Adds the Stage-0 questions that `ui_toy.py` does not answer (Q1–Q3 below). Imports `ui_toy`. Budgets are in `BUDGETS`. |
| `run.sh`, `config.env` | Only for the (parked) Kaggle pipeline. Not needed locally. |

Runs locally: `.venv/Scripts/python -m jupyter lab Stage0/stage0.ipynb` (numpy, scipy, matplotlib, pandas, jupyter).
Results are cached as JSON in `Stage0/results/` and figures saved in `Stage0/figs/` (both git-ignored).
`QUICK = True` in the first cell runs a smoke test into `Stage0/_quick/`. Its verdicts are noise (4 models), not results.
The CLI equivalents are `python stage0_checks.py all` and `python ui_toy.py E1 E2c E6`.

Full-budget timings on a laptop CPU: Q1 1.2 min (measured); Q2 + Q3 about 10 min (extrapolated from the quick run);
`ui_toy` E1 3.5 min, E2a 17 min (measured).

## Does the testbed answer the Stage-0 questions?

| Stage-0 question (plan) | Before | Now |
|---|---|---|
| Prop. 5(a): T̂ = Λ₁ at m = 1 | yes: `ui_toy selftest` (ii) and `E1.max_abs_gap_kandpal_vs_opt_m1` | unchanged |
| Oracle Λ_m vs T̂ across m | partly: `E1` (one configuration: d=2, σ_μ=3, n=50) | **Q1**: 6 configurations, paired per model |
| **Claim 1**: J(T̂) < Adv*_B(θ) for m ≥ 2 | not tested as a claim: E1 reports both, without a paired comparison or a verdict | **Q1**: per-model gap Adv*_B − J_in(T̂), mean ± SE, verdict per configuration. J_in is the in-sample best threshold, biased up, so the gap is conservative |
| Feasible Λ_m (draft App. I) | **missing** | **Q1**: `log Σ_u α_u Π_i N(x_i; μ̂_u, σ²) − Σ_i log p_ref(x_i)`. **Design choice, to confirm**: "without the population prior" is read here as replacing the joint population marginal with the product of per-document references. It equals T̂ at m = 1 (checked) |
| Prop. 6: Var(S_m \| b) = v_B + v_W/m | **missing** (E1 stores v_B, v_W but never checks the identity) | **Q2**: measured variance / (v_B + v_W/m) per m |
| Prop. 6: eq. (3) prediction vs measured AUROC | **missing** | **Q2** |
| Prop. 6: ceiling AUROC(τ) | yes: `E1 *_plateau_*` (exact, Gauss–Hermite) | **Q2** also compares it with the eq. (3) ceiling |
| Step F estimable at Enron scale? | **missing** | **Q2 pilot**: 34 or 68 users per class, 32 held-out docs each; spread of the plug-in prediction |
| Cor. 2 sandwich α KL ≤ own ≤ log(1 + α χ²) | partly: `E5` in a two-world idealisation (D_{−u} = one Gaussian) | **Q3**: per member user in the hierarchical model, with D_{−u} the real mixture; also checks τ_u = own + typicality |
| Cor. 2 limit: TPR_u(m) → 1{τ_u > t_γ} | partly: `E2c` (per-k TPR at m = 8 only) | **Q3**: m ∈ {1, 4, 16, 64, 256}, per user, T̂ vs oracle |
| Prop. 2 certificates and DKW behaviour vs n | yes: `E6` (coverage and null false-positive rate), `selftest` (iv) AUROC envelope | also `common/certificates.py` self-check |

## Problems found in the testbed

1. `ui_toy.py figs` and `numbers` import `ui_figs.py` / `make_numbers.py`, which are not in the repo. `all` now skips them when absent. Figures for Stage 0 come from `stage0_checks.py`.
2. `ui_toy` uses **uniform** targets over members. Draft Assumption 2 (C1) weights targets by α_u. These agree only when all k_u are equal. In `E2c` (heterogeneous k), T̂ is therefore not the optimal statistic even at m = 1. That is consistent with the toy's own game, but it is not the draft's setting.
3. The selftest's χ² identity at (m=2, k=8, n=3) gives 0.2116 ± 0.0048 against 0.2177 in closed form (1.3 SE). This is fine, just noted.

## Problem found in the draft

- **Cor. 2, "never detected at FPR γ, for any m"**: this only holds in the limit. At finite m, a user with τ_u < t_γ still gets a positive TPR from within-user noise. Q3 reports the TPR of such users at each m (in the quick run their maximum per-user TPR reached about 0.3 at m = 16). Proposed wording: "TPR_u(m) → 0 as m → ∞".
- The sandwich itself holds for any α and any D_u, D_{−u}, with no assumption. The lower bound follows from concavity of the log; the upper bound from Jensen's inequality and E_{D_u}[D_u/D_{−u}] = 1 + χ².
