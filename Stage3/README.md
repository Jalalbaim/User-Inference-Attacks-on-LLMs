# Stage 3: DKW certificate vs number of users per class (Cor. 1)

**Status: DONE (2026-11-25), plan gate met.** At m = 32, T̂ needs about 26 users per class for half of the subsamples to give a positive certificate. Cor. 1's 8.76/Ĵ² predicts 31, so the corollary is slightly conservative (ratio 0.86). Details are in the notes cell of `stage3.ipynb`.

**Plan:** `docs/experiment_plan_enron_gptneo125m.md`, section 4 row 3. No GPU: post-processing of Stage 2.

| File | What it is |
|---|---|
| `analysis.py` | Subsamples n users per class without replacement (200 times) per seed, m, draw and statistic. Computes the DKW certificate, P(certificate > 0), and the empirical n₅₀ vs n_pred = 8.76/Ĵ²_full |
| `stage3.ipynb` | Tables, figure, notes |

**Inputs:** the per-user statistics of Stage 2 (`Stage2/analysis.cached_user_values`, cached in `Stage2/results/_user_values.csv.gz`). Delete that cache after adding new Stage 2 results.
**Caveats:** Ĵ_n is biased upwards at small n. Near the full sample size the subsamples overlap heavily. The aggregator uses EVAL users only (≤ 44 per class).
