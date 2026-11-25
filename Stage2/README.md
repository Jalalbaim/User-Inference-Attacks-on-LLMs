# Stage 2: core C3 (T̂ vs S_cat vs aggregator vs blind baseline)

**Plan:** `docs/experiment_plan_enron_gptneo125m.md`, section 4 row 2 (Steps D–F). Needs a GPU (Kaggle T4).
**Falsifiers (plan section 4):**
1. T̂ keeps rising with m, with no ceiling.
2. Neither S_cat nor the aggregator beats T̂ at any m.
3. The eq. (3) prediction is far from the measured AUROC.

## Pre-registered headline certificate (fixed 2026-11-24, before any Stage 2 result)
**T̂, m = 32, η = 0.05, all 175 users, main run, seed 1** (4-epoch protocol): the DKW certificate of Prop. 2(ii).
Every other seed, statistic and m is descriptive.

## Decisions (user, 2026-11-24)
- Train for 4 epochs, keeping the best member-validation epoch. Seed 0 is retrained under this protocol (the Stage 1 pilot used 10 epochs).
- 10 main seeds (0–9) and a null run for seed 0.
- S_cat: R = 4 orders. Sequences longer than 2048 tokens are scored with a sliding window (window 2048, stride 1024; option (c)).
- Aggregator features per user, from the m-document subset:
  - the mean, sd, min, max and 10/25/50/75/90% quantiles of s;
  - the mean log p_θ and the mean log p_ref.

  That is 11 features, with standardized logistic regression (C = 1) fit on FIT users and evaluated on EVAL users.
- Blind baseline: TF-IDF (sublinear, min_df = 2) + logistic regression on the subset's text, fit on FIT and evaluated on EVAL.
- T̂ and S_cat are certified on all users; the learned statistics only on EVAL users. FIT/EVAL is used only for fitting.

## Added after the results (user, 2026-11-25)
- **User-bootstrap CIs** (plan row 2): `analysis.bootstrap_vs_T`, B = 1000. Each replicate resamples the users once and reuses that resample for every seed, m, draw and statistic. For the aggregator and blind baseline, the FIT fit is held fixed. Notebook section 4b.
- **EXPLORATORY, not pre-registered:** eq. (3) on transformed scores (`slog` = sign·log1p|s|, `int` = rank-inverse-normal). This was added because eq. (3) on raw s fails, since s is heavy-tailed. Notebook section 5b.

## Files

| File | What it is |
|---|---|
| `run.sh`, `config.env` | Kaggle entry: train the main seeds, then the null seeds (`Stage1/train_score.py --keep-fp32`), then S_cat |
| `scat.py` | S_cat on the GPU, for every finished model plus the reference (computed once). It checks that S_cat at m = 1 equals the score table's s(x) |
| `../common/draws.py` | Deterministic m-doc subsets and orders per user, shared by every seed and statistic, so the comparisons are paired |
| `analysis.py` | Local: the four statistics per (seed, m, draw); AUROC, Ĵ, the DKW and fixed-test certificates; null; blind; Prop. 6; the headline |
| `stage2.ipynb` | Analysis notebook, run locally on the downloaded archives |

## Running on Kaggle
Notebook settings: **Accelerator: GPU T4**, **Internet: on**. Run the commands in this order:
```
!curl -sSLO https://raw.githubusercontent.com/Jalalbaim/User-Inference-Attacks-on-LLMs/master/kaggle/run_kaggle.sh && bash run_kaggle.sh S2 S2_SMOKE=1
!bash run_kaggle.sh S2                                   # session A: seeds 0-4 + null seed 0
!bash run_kaggle.sh S2 SEEDS="5 6 7 8 9" S2_NULL_SEEDS=  # session B: seeds 5-9 (a new session)
```
- **Smoke run (~15–20 min):** 4 users per class and 60 steps. It prints the measured training estimate (`SMOKE speed ...`) and the S_cat estimate (`S_cat SMOKE: ... min per model`).
- **Outputs:** `/kaggle/working/outputs/S2_<time>.zip`: the run outputs plus the per-seed score and S_cat tables from `ckpt/`, without model weights. Download it and extract it into `Stage2/results/<session>/`.
- **Checkpoints:** `ckpt/S2/<run>_s<seed>/`. `best/` is kept in fp32, so S_cat uses the exact weights behind the score table: about 500 MB per model, about 3 GB per session.
- **Time limit:** the script stops itself before the budget runs out (exit 75). To resume: Save Version, attach the output as an input, and rerun the same command. Finished seeds and S_cat tables are skipped.

## Local tests done
- **Full Kaggle path with the tiny random model** (`run_kaggle.sh S2 S2_SMOKE=1 S2_TINY=1`): 5 main seeds, null seed 0, reference and S_cat for 6 models, then the archive. Exit 0.
  - **S_cat at m = 1 equals the score table's s(x)** (max |diff| 1.5e-5), so the encoding, EOS mask and windows match T̂'s.
  - Sequences up to 4,128 tokens went through the sliding window.
- **`analysis.py` and `stage2.ipynb` end to end** on Stage 1's real seed-0 main and null score tables, with a fake S_cat table (s summed + noise).
  - The headline function reproduces the Stage 1 numbers (AUROC 0.881, certificate 0.350).
  - The null shows no positive certificate.

Not testable locally: real speed on the T4. The Kaggle smoke run measures it.
