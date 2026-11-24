#!/usr/bin/env bash
# Stage 2 entry, run by the built script with the header's environment.
# 1. train + score main seeds (Stage1/train_score.py), 2. null seeds, 3. S_cat for every finished model.
# Each step skips work already done (DONE / scat.csv.gz in the checkpoints), so a rerun resumes.
set -euo pipefail
cd "$UIA_CODE_DIR/Stage2"
EXTRA=""; CAT_EXTRA=""
if [ "${S2_SMOKE:-0}" = 1 ]; then EXTRA="--smoke --smoke-users 4"; CAT_EXTRA="--smoke"; fi
if [ "${S2_TINY:-0}" = 1 ]; then EXTRA="$EXTRA --tiny"; CAT_EXTRA="$CAT_EXTRA --tiny"; fi
"$PYTHON" -c "import torch, transformers; print('[S2] torch', torch.__version__, '| transformers', transformers.__version__)"
TRAIN=("$UIA_CODE_DIR/Stage1/train_score.py" --ckpt "$UIA_CKPT_DIR" --model "$S2_MODEL" --epochs "$S2_EPOCHS"
       --bs "$S2_BS" --lr "$S2_LR" --max-len "$S2_MAXLEN" --keep-fp32)
# shellcheck disable=SC2086
"$PYTHON" "${TRAIN[@]}" --run main --seeds "$UIA_SEEDS" --out "$UIA_OUT_DIR/main" $EXTRA
if [ -n "${S2_NULL_SEEDS:-}" ]; then
    # shellcheck disable=SC2086
    "$PYTHON" "${TRAIN[@]}" --run null --seeds "$S2_NULL_SEEDS" --out "$UIA_OUT_DIR/null" $EXTRA
fi
# shellcheck disable=SC2086
"$PYTHON" scat.py --main-seeds "$UIA_SEEDS" --null-seeds "${S2_NULL_SEEDS:-}" --ckpt "$UIA_CKPT_DIR" \
    --out "$UIA_OUT_DIR/scat" --model "$S2_MODEL" --max-len "$S2_MAXLEN" --draws "$S2_CAT_DRAWS" --R "$S2_CAT_R" \
    --window "$S2_WINDOW" $CAT_EXTRA
