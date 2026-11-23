#!/usr/bin/env bash
# Stage 1 entry, run by the built script with the header's environment.
set -euo pipefail
cd "$UIA_CODE_DIR/Stage1"
EXTRA=""; [ "${S1_SMOKE:-0}" = 1 ] && EXTRA="--smoke"; [ "${S1_TINY:-0}" = 1 ] && EXTRA="$EXTRA --tiny"
"$PYTHON" -c "import torch, transformers; print('[S1] torch', torch.__version__, '| transformers', transformers.__version__)"
# shellcheck disable=SC2086
"$PYTHON" train_score.py --seeds "$UIA_SEEDS" --run "$S1_RUN" --out "$UIA_OUT_DIR" --ckpt "$UIA_CKPT_DIR" \
    --model "$S1_MODEL" --epochs "$S1_EPOCHS" --bs "$S1_BS" --lr "$S1_LR" --max-len "$S1_MAXLEN" $EXTRA
