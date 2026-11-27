# Stage 7: user-level DP

**7a status: DONE (2026-11-26), no GPU.** Certified lower bounds on user-level ε from the Stage 2 audits, plus the ceiling that 175 users puts on any such bound.
**7b status: READY TO RUN on Kaggle (2026-11-27).** User-level DP-SGD fine-tuning; see the 7b section below.

**Plan:** `docs/experiment_plan_enron_gptneo125m.md`, section 4 row 7.

## 7a: what it computes
The Stage 2 models are not DP (true ε = ∞), so the numbers measure **audit power**: the largest ε this audit could refute. Two routes, confidence 95%:
1. **TV inversion (Lemma 1):** ε ≥ log((1 + c − 2δ)/(1 − c)), where c is the DKW certificate.
2. **One-run audit (Steinke, Nasr, Jagielski 2023), users as units.**
   - Membership coins are independent and fair (`data/enron/splits.py`).
   - Users are ranked by the statistic, and guesses are made on k of them.
   - The 14 (rule, k) pairs are fixed in advance, with Bonferroni β/14.
   - **Pure-ε version:** the δ correction must be added before 7b.
   - The "oracle" (best pair chosen post hoc) is not valid and is shown for reference only.

One test per model (draw 0). The null run and the aggregator use EVAL users only.

## Results (T̂, m = 32)
| | TV inversion | One-run (valid) |
|---|---|---|
| One model | 0.47 (0.26–0.76) | 1.57 (1.52–1.58) |
| Pooled over 10 runs | — | 3.22 |
| Ceiling with 175 users (perfect attack), one model | 1.67 | 3.42 |
| Users needed to refute ε = 8 (perfect attack) | about 4·10⁷ | about 9,000 |

**Null:** every bound is 0. **S_cat ≈ T̂.** The aggregator is weaker, since it has only the EVAL half of the users.

**Consequence for 7b:** a nominal ε above about 3.5 cannot be refuted with this dataset, by construction.

**Open:**
- the δ correction of the one-run bound;
- adjacency: Lemma 1 is replace-one, the one-run bound is add/remove (inclusion of a user).

| File | What it is |
|---|---|
| `analysis.py` | `bounds` (per seed, m, statistic), `pooled`, `ceilings`, `users_needed` |
| `stage7.ipynb` | Tables, two figures, notes |

## 7b: user-level DP-SGD (Kaggle, GPU)
**Algorithm** (`train_dp.py`): DP-Adam, written by hand so that the privacy unit is the user, not the document. At every step t = 1..T:
1. Every member user is sampled independently with probability q (Poisson sampling).
2. For each sampled user: draw 8 of their training docs and compute g_u, the gradient of the mean token NLL. Clip the whole user gradient to L2 norm C. A non-finite g_u contributes 0.
3. g = (Σ g_u + N(0, σ²C² I)) / (q · n_members).
4. Adam step, with the learning rate decaying linearly to 0.

**Privacy and selection.**
- Adding or removing one user moves Σ g_u by at most C, so this is the Poisson-subsampled Gaussian mechanism with users as the unit (add/remove adjacency).
- σ is calibrated with Opacus' RDP accountant for the target (ε, δ, q, T). The achieved ε is stored in the score table. Opacus is used only for accounting.
- **No model selection:** the final model is scored. Stage 2's best-epoch choice uses members' validation docs, which would break DP. The validation losses at steps 0, T/2 and T are for reporting only.
- Members are those of Stage 2's main run for the same seed, so each DP model pairs with the non-DP one.

**Defaults** (`config.env`, chosen 2026-11-27; override them on the command line):

| Setting | Value | Why |
|---|---|---|
| ε (`S7_EPS`) | 1, 2, 4, 8, inf | inf = clipping only, no noise. 7a: ε above about 3.5 cannot be refuted with 175 users |
| δ (`S7_DELTA`) | 1e-5 | |
| q, T (`S7_Q`, `S7_STEPS`) | 0.5, 150 | about 42 users per step; each user takes part in about 75 steps |
| C, docs per user per step | 1.0, 8 | 8 = Stage 2's batch size |
| Learning rate | 5e-5 | Stage 2's value, not tuned (tuning would itself need privacy accounting) |
| Seeds | 0, 1, 2 | |

σ for 82–83 members: 25.0 (ε=1), 13.3 (ε=2), 7.2 (ε=4), 4.0 (ε=8). Noise std relative to the summed clipped gradients, per coordinate: σ/(q·n) ≈ 0.6, 0.32, 0.17, 0.10.
**Expectation:** utility may collapse at small ε with only about 85 users. The clipping-only run separates the effect of clipping from the effect of noise.

**Kaggle** (GPU T4, Internet on):
```
!curl -sSLO https://raw.githubusercontent.com/Jalalbaim/User-Inference-Attacks-on-LLMs/master/kaggle/run_kaggle.sh && bash run_kaggle.sh S7 S7_SMOKE=1 SEEDS="0"
!bash run_kaggle.sh S7
```
- **Smoke run** (about 10 min): 6 users per class and 6 steps per model. It prints `SMOKE speed ... about X h of training` per model.
- **Full run:** 15 models, estimated 5–7 h (budget 11.5 h). If the budget runs out, the script saves a checkpoint, stops itself (exit 75) and still writes the zip. To resume: Save Version, attach the output, rerun the same command. A resumed model is identical to an uninterrupted one (tested).
- **Output:** `/kaggle/working/outputs/S7_<time>.zip` contains every output of the S7 runs on that machine:
  - every `outputs/S7_*` folder (run.log, config, manifest, pip freeze, score table, histories);
  - all of `ckpt/S7` (per-model score tables, histories, DONE markers, reference scores, resume state);
  - weights too, if `S7_KEEP_WEIGHTS=1`.

  Extract it into `Stage7/results/`.
