# Stage 4: light users and caps (Cor. 2)

**Status:** 4a DONE (2026-11-25). **4b DONE (2026-11-27).** Results are in `Stage4b/` (Kaggle run of 2026-11-26, 40 models, 5 h) and in `stage4.ipynb` section 4b.

**4b result:** capping costs little and only below about 50 docs. T̂ (all users, m = 32): 0.839 uncapped vs 0.773 at c = 10 (paired Δ −0.066 [−0.093, −0.040]); at c ≥ 50 the CI contains 0. Aggregator ≈ T̂ at every cap, so the protection does not depend on the statistic. **A uniform cap does not create light users in Cor. 2's α_u sense:** α_u = 1/members ≈ 0.012 for everyone. Cor. 2's contrast is not supported in any regime tested. A real test needs mixed caps (some members capped, others not); not run, to discuss.

**Plan:** `docs/experiment_plan_enron_gptneo125m.md`, section 4 row 4, Step G.

## 4a: light users without retraining (no GPU, post-processing of Stage 2)
- Member TPR at 5% FPR per k_u quartile and α_u quartile, per statistic and m (threshold = the 95% quantile among non-members). Also Spearman correlations.
- **Result:** no dependence on k_u or α_u for T̂ or S_cat. At m = 32, TPR is 0.43 / 0.37 / 0.40 / 0.36 from the lightest to the heaviest quartile, and ρ ≈ −0.02.
- **Limit:** every Enron user has k_u ≥ 114, so there are no truly light users and Cor. 2 can't be tested without caps.

| File | What it is |
|---|---|
| `analysis.py` | `member_detection`, `tpr_by_bin`, `rank_corr` |
| `stage4.ipynb` | Tables, figure, notes |

## 4b: caps (Kaggle, GPU)
**Design.** Defaults chosen 2026-11-26; override them on the command line.
- Caps c ∈ {10, 25, 50, 100} train docs per member user, the plan's grid. Every user has ≥ 114 docs, so every cap binds for every user.
- Seeds 0–9, with the **same members as Stage 2's uncapped run**, so Stage 2 is the c = ∞ point of a paired comparison.
- Training is identical to Stage 2: 4 epochs, best member-validation epoch, batch 8, lr 5e-5, 128 tokens.
- Statistics: **no S_cat** (S_cat ≈ T̂ in Stage 2). T̂ and the aggregator come from the per-document score tables.
- The c docs are the first c of a fixed per-user permutation. They are nested across caps (10 ⊂ 25 ⊂ 50 ⊂ 100) and identical across seeds.
- Weights are deleted after scoring; only the score tables are kept.

**Code.** `run.sh` → `Stage1/train_score.py --caps "10 25 50 100" --drop-weights`.
- Loops seeds (outer) × caps (inner), so a partial run has every cap for each finished seed.
- Score tables: `run = cap<c>`, plus columns `cap`, `k_u` (= c) and `k_u_full`.

**Kaggle** (GPU T4, Internet on):
```
!curl -sSLO https://raw.githubusercontent.com/Jalalbaim/User-Inference-Attacks-on-LLMs/master/kaggle/run_kaggle.sh && bash run_kaggle.sh S4 S4_SMOKE=1 SEEDS="0"
!bash run_kaggle.sh S4
```
- **Smoke run** (about 10 min): prints `SMOKE speed ... about X h of training` per cap.
- **Full run:** about 25–35 min per seed for all 4 caps, about 5–6 h for 10 seeds (budget 11.5 h). If the budget runs out, the script stops itself (exit 75). Save Version, attach the output, and rerun the same command; finished models are skipped.
- **Download:** `/kaggle/working/outputs/S4_<time>.zip` (score tables and histories, no weights). Extracted into `Stage4/stage4b/`: the score tables in `ckpt/S4/cap<c>_s<seed>/` are the input; `ckpt/S4/smoke/` is the smoke run and is ignored.

**4b analysis code** (`analysis.py`): `load_caps`, `cap_user_values` (cached in `Stage4b/_user_values_caps.csv.gz`), `with_uncapped`, `cap_metrics`, `cap_summary`, `paired_cap_bootstrap`, `load_cap_histories`. Checked: labels identical to Stage 2, k_u = c for members, and Stage 2's seed 0 recomputed through this path matches exactly.

**Local test (2026-11-26):** `run_kaggle.sh S4 S4_SMOKE=1 S4_TINY=1 SEEDS="0 1"` exited 0 with the zip and no weights. A rerun skipped all 8 finished models. On the real splits, every cap gives exactly members × c docs, nested and identical across seeds. The uncapped path (Stage 2 behaviour) is unchanged.
