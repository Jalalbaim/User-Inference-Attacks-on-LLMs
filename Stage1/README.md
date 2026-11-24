# Stage 1: reproduce Kandpal et al. on Enron (GPT-Neo 125M)

**Status: CLOSED (2026-11-24), gate passed.** Seed 0: T̂ AUROC 0.881 [0.828, 0.928] at m = 32 (Kandpal: 0.873); DKW certificate +0.350.
Null model: AUROC 0.559 [0.432, 0.678], certificate −0.256. Details are in the notes cell of `stage1.ipynb`.

**Plan:** `docs/experiment_plan_enron_gptneo125m.md`, section 4 row 1 (Steps A–C). Needs a GPU (Kaggle T4).
**Gate:** T̂ AUROC near Kandpal's 0.873, and the null model ≈ chance. Otherwise debug; do not proceed.

## Files

| File | What it is |
|---|---|
| `train_score.py` | Fine-tune, keep the best epoch, score the held-out docs. One call handles several seeds and resumes from checkpoints |
| `lm.py` | Document encoding, batching, log-probabilities (shared with later stages) |
| `analysis.py` | Score table → T̂ AUROC vs m, Ĵ, DKW certificate, gate check (local, no GPU) |
| `stage1.ipynb` | Analysis notebook, run locally on the downloaded results |
| `run.sh`, `config.env` | Kaggle entry and settings |
| `../common/enron_data.py` | Loads the committed splits and checks their sha256 |

## Settings (agreed with the user)

| | |
|---|---|
| Model | `EleutherAI/gpt-neo-125m`; the same pretrained model is the reference p_ref |
| Training | members' `train` docs; AdamW, lr 5e-5, batch 8, linear decay to 0, no warmup, no weight decay |
| Epochs | Every epoch is run, and the epoch with the lowest **member validation** loss is scored (Kandpal). Seed-0 pilot: 10 epochs, best at epoch 2. **From now on: 4 epochs** (user decision, 2026-11-24) |
| Precision | fp16 mixed precision (the T4 has no bf16); scoring in fp32 |
| Document | `[EOS] + first 128 tokens`; EOS is only a start marker, so all 128 text tokens are scored |
| Score | s(x) = log p_θ(x) − log p_ref(x), summed over the text tokens |
| Runs | `main`: members = b_s<seed> = 1, every user scored. `null`: members = b_s<seed> = 1 and FIT only; EVAL users are never trained on, and their b is a fake label |

**Implementation choice, not in the plan:** the EOS start token. Without it, the first token of each email can't be scored. Kandpal et al. don't say how they handle this.

## Running on Kaggle

Notebook settings: **Accelerator: GPU T4**, **Internet: on**.

```
!curl -sSLO https://raw.githubusercontent.com/Jalalbaim/User-Inference-Attacks-on-LLMs/master/kaggle/run_kaggle.sh && bash run_kaggle.sh S1 S1_SMOKE=1
!bash run_kaggle.sh S1 SEEDS=0                      # pilot (seed 0, main run)
!bash run_kaggle.sh S1 S1_RUN=null SEEDS=0          # null model
!bash run_kaggle.sh S1 SEEDS="1 2"                  # more seeds in one session (~2 h each)
```

- **Smoke run (~10 min):** 6 users, 60 steps × 2 epochs. Its log prints the measured speed and a full-run estimate
  (`SMOKE speed ... -> full run ... about X h`). Its checkpoints go to `ckpt/S1/smoke/`, apart from real runs.
- **Outputs:** `/kaggle/working/outputs/S1_<time>/` holds `scores.csv.gz` (the Step-C table), `history_<run>_s<seed>.json`,
  `run.log` and `manifest.json`, and there is a `.tar.gz` of the same folder. Download the `.tar.gz` and extract it into `Stage1/results/`.
- **Checkpoints:** `/kaggle/working/ckpt/S1/<run>_s<seed>/`.
  - `last.pt` is written after every epoch and deleted when the seed finishes.
  - `best/` keeps the best-epoch weights, saved in fp16 at the end (about 250 MB per seed; Stage 2 needs them).
  - `DONE` marks a finished seed, which is skipped if run again.
- **Time limit:** the script stops itself before the session budget runs out (exit 75, status `time_budget_reached`). To resume: Save Version,
  attach that version's output as an input, and rerun the same command.

## Local tests done (CPU, tiny random 2-layer model; numbers meaningless)

- 2 seeds trained, scored and saved, and the runtime estimate printed.
- The `null` run: EVAL users are untrained.
- Stop before the deadline (exit 75), then resume: the resumed run ends at the same validation loss as an uninterrupted one.
- The full Kaggle path run locally (`run_kaggle.sh S1 S1_SMOKE=1 S1_TINY=1`): build, header, footer, archive.
- `stage1.ipynb` runs end to end on these outputs.

Not testable locally: the real 125M model on a GPU, fp16 behaviour, and the real speed. The Kaggle smoke run checks all three.
