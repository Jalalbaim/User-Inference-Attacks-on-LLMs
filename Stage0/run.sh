#!/usr/bin/env bash
# Stage 0 entry, run by the built script with the header's environment.
set -euo pipefail
cd "$UIA_CODE_DIR/Stage0"
Q=""; [ "${S0_QUICK:-0}" = 1 ] && Q="--quick"

echo "[S0] certificate module self-check"
"$PYTHON" "$UIA_CODE_DIR/common/certificates.py"

echo "[S0] ui_toy self-test"
"$PYTHON" ui_toy.py selftest | tee "$UIA_OUT_DIR/ui_toy_selftest.txt"

echo "[S0] ui_toy experiments: $S0_TOY $Q"
# shellcheck disable=SC2086
"$PYTHON" ui_toy.py $S0_TOY $Q

echo "[S0] stage0_checks: $S0_CHECKS $Q"
# shellcheck disable=SC2086
"$PYTHON" stage0_checks.py $S0_CHECKS $Q
