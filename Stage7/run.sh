#!/usr/bin/env bash
# Stage 7b entry, run by the built script with the header's environment.
# For every seed (outer loop) and target epsilon (inner loop): user-level DP-SGD fine-tuning (Stage7/train_dp.py),
# then score all held-out docs with the final model and the reference. Finished models (DONE) are skipped on a rerun.
set -euo pipefail
cd "$UIA_CODE_DIR/Stage7"
# Opacus is used ONLY for the privacy accountant (noise calibration); --no-deps keeps Kaggle's torch untouched.
if ! "$PYTHON" -c "import opacus" 2>/dev/null; then
    "$PYTHON" -m pip install -q --no-deps "opacus==$S7_OPACUS" opt_einsum 2>&1 | tail -2
fi
"$PYTHON" -c "import torch, transformers, opacus; print('[S7] torch', torch.__version__, '| transformers', transformers.__version__, '| opacus', opacus.__version__)"
EXTRA=""
if [ "${S7_SMOKE:-0}" = 1 ]; then EXTRA="--smoke"; fi
if [ "${S7_TINY:-0}" = 1 ]; then EXTRA="$EXTRA --tiny"; fi
if [ "${S7_KEEP_WEIGHTS:-0}" = 1 ]; then EXTRA="$EXTRA --keep-weights"; fi
# shellcheck disable=SC2086
"$PYTHON" train_dp.py --seeds "$UIA_SEEDS" --eps "$S7_EPS" --delta "$S7_DELTA" --q "$S7_Q" --steps "$S7_STEPS" \
    --clip "$S7_CLIP" --docs-per-user "$S7_DOCS_PER_USER" --lr "$S7_LR" --model "$S7_MODEL" --max-len "$S7_MAXLEN" \
    --ckpt "$UIA_CKPT_DIR" --out "$UIA_OUT_DIR/dp" $EXTRA
