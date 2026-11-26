# Stage 4: light users and caps (Cor. 2)

**Status:** 4a DONE (2026-11-25). 4b (cap retraining, GPU) is READY TO RUN on Kaggle (code tested locally 2026-11-26).

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
- **Download:** `/kaggle/working/outputs/S4_<time>.zip` (score tables and histories, no weights). Extract it into `Stage4/results/`.

**Local test (2026-11-26):** `run_kaggle.sh S4 S4_SMOKE=1 S4_TINY=1 SEEDS="0 1"` exited 0 with the zip and no weights. A rerun skipped all 8 finished models. On the real splits, every cap gives exactly members × c docs, nested and identical across seeds. The uncapped path (Stage 2 behaviour) is unchanged.
