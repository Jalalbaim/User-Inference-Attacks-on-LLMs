#!/usr/bin/env bash
# build.sh <stage number 0-7> <output script>
# Assembles ONE self-contained bash script:
#   preamble (ids, commit, overrides)  ->  repo code embedded as base64 tar.gz  ->  header.sh
#   ->  Stage<N>/run.sh under a time-budget timeout  ->  footer.sh (always runs)
# The built script needs nothing but bash, tar, base64, coreutils and the stage's Python deps.
set -euo pipefail

N="${1:?stage number}"
OUT="${2:?output path}"
KDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$KDIR/.." && pwd)"
STAGE="Stage${N}"

[ -f "$REPO/$STAGE/run.sh" ]     || { echo "[build] $STAGE/run.sh missing" >&2; exit 2; }
[ -f "$REPO/$STAGE/config.env" ] || { echo "[build] $STAGE/config.env missing" >&2; exit 2; }
if grep -q "UIA_NOT_IMPLEMENTED" "$REPO/$STAGE/run.sh"; then
    echo "[build] $STAGE is not implemented yet (see $STAGE/README.md)." >&2
    exit 2
fi

COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY="$(git -C "$REPO" status --porcelain 2>/dev/null | grep -qv '^??' && echo "+dirty" || true)"

PAYLOAD="$(mktemp)"
trap 'rm -f "$PAYLOAD"' EXIT
# Code only: no results, figures, caches or data dumps.
# Stages 2+ reuse Stage 1's training and scoring code (Stage1/train_score.py, Stage1/lm.py).
SHARED=""; [ "$N" -ge 2 ] && SHARED="Stage1"
tar -C "$REPO" -czf "$PAYLOAD" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='results' --exclude='figs' \
    --exclude='data/raw' --exclude='data/processed' --exclude='*.ipynb' \
    common data kaggle $SHARED "$STAGE" $( [ -d "$REPO/docs" ] && echo docs )

{
    echo '#!/usr/bin/env bash'
    echo "# Built by kaggle/build.sh: $STAGE, commit $COMMIT$DIRTY, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# Usage: bash $(basename "$OUT") [KEY=VALUE ...]"
    echo 'set -euo pipefail'
    echo "export UIA_STAGE=$N UIA_ID=S$N UIA_GIT_COMMIT='$COMMIT$DIRTY'"
    echo 'UIA_OVERRIDES=("$@")'
    echo 'UIA_START=$(date +%s)'
    echo 'export UIA_CODE_DIR="${UIA_CODE_DIR:-/tmp/uia_code_S'"$N"'}"'
    echo 'rm -rf "$UIA_CODE_DIR" && mkdir -p "$UIA_CODE_DIR"'
    echo 'base64 -d <<'"'"'__UIA_PAYLOAD__'"'"' | tar -xz -C "$UIA_CODE_DIR"'
    base64 "$PAYLOAD"
    echo '__UIA_PAYLOAD__'
    echo
    echo '# ======================= header.sh ======================='
    cat "$KDIR/header.sh"
    echo
    echo '# ======================= stage ==========================='
    echo 'set +e'
    echo 'timeout --signal=TERM --kill-after=120 "$UIA_TIME_LEFT" bash "$UIA_CODE_DIR/Stage${UIA_STAGE}/run.sh" 2>&1 | tee -a "$UIA_LOG"'
    echo 'UIA_RC=${PIPESTATUS[0]}'
    echo 'set -e'
    echo
    echo '# ======================= footer.sh ======================='
    cat "$KDIR/footer.sh"
    echo
    echo 'exit "$UIA_RC"'
} > "$OUT"
chmod +x "$OUT"
echo "[build] $OUT  ($STAGE, commit $COMMIT$DIRTY, $(wc -c < "$OUT") bytes)"
