#!/usr/bin/env bash
# Stage 4b entry, run by the built script with the header's environment.
# For every seed (outer loop) and cap (inner loop): train on at most c docs per member user, keep the best
# member-validation epoch, score all held-out docs (Stage1/train_score.py --caps). No S_cat.
# Weights are deleted once a model's score table is written; finished models (DONE) are skipped on a rerun.
set -euo pipefail
cd "$UIA_CODE_DIR/Stage1"
EXTRA=""
if [ "${S4_SMOKE:-0}" = 1 ]; then EXTRA="--smoke --smoke-users 4"; fi
if [ "${S4_TINY:-0}" = 1 ]; then EXTRA="$EXTRA --tiny"; fi
"$PYTHON" -c "import torch, transformers; print('[S4] torch', torch.__version__, '| transformers', transformers.__version__)"
# shellcheck disable=SC2086
"$PYTHON" train_score.py --run main --seeds "$UIA_SEEDS" --caps "$S4_CAPS" --drop-weights \
    --ckpt "$UIA_CKPT_DIR" --out "$UIA_OUT_DIR/caps" --model "$S4_MODEL" --epochs "$S4_EPOCHS" \
    --bs "$S4_BS" --lr "$S4_LR" --max-len "$S4_MAXLEN" $EXTRA
