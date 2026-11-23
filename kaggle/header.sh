# header.sh -- sourced inline by the built script (not run on its own).
# Sets up directories, the stage config + overrides, seeds/determinism, deps, GPU check,
# the time budget, and checkpoint resume.
#
# Exported to the stage (StageN/run.sh and its Python code):
#   UIA_CODE_DIR  extracted repo code (also on PYTHONPATH)
#   UIA_OUT_DIR   this run's outputs  -> /kaggle/working/outputs/<S#>_<UTC time>/
#   UIA_CKPT_DIR  checkpoints, stable per stage -> /kaggle/working/ckpt/<S#>/  (resume source)
#   UIA_DATA_DIR  prepared data (see data/README.md)
#   UIA_SEEDS     space-separated seeds;  UIA_ETA  confidence level of certificates
#   UIA_DEADLINE  epoch seconds by which the stage must have checkpointed and exited
#   PYTHON        interpreter to use

# ---------- where we are ----------
if [ -d /kaggle/working ]; then UIA_ON_KAGGLE=1; UIA_WORK=/kaggle/working
else UIA_ON_KAGGLE=0; UIA_WORK="${UIA_WORK:-$PWD/_runs}"; fi
export UIA_ON_KAGGLE UIA_WORK

# ---------- stage config, then KEY=VALUE overrides from the command line ----------
set -a
# shellcheck disable=SC1090
. "$UIA_CODE_DIR/Stage${UIA_STAGE}/config.env"
for _kv in "${UIA_OVERRIDES[@]+"${UIA_OVERRIDES[@]}"}"; do
    case "$_kv" in *=*) eval "${_kv%%=*}=\"\${_kv#*=}\"" ;;
                   *) echo "[header] ignoring argument '$_kv' (expected KEY=VALUE)" >&2 ;; esac
done
set +a

export UIA_RUN_TAG="${UIA_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
export UIA_OUT_DIR="$UIA_WORK/outputs/$UIA_RUN_TAG"
export UIA_CKPT_DIR="${UIA_CKPT_DIR:-$UIA_WORK/ckpt/$UIA_ID}"
export UIA_DATA_DIR="${UIA_DATA_DIR:-$UIA_WORK/data}"
export UIA_LOG="$UIA_OUT_DIR/run.log"
mkdir -p "$UIA_OUT_DIR" "$UIA_CKPT_DIR" "$UIA_DATA_DIR"

# ---------- seeds and determinism ----------
export UIA_SEEDS="${SEEDS:-0}"
export UIA_ETA="${ETA:-0.05}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # deterministic cuBLAS (torch.use_deterministic_algorithms)
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$UIA_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/tmp/hf_home}"   # model cache outside /kaggle/working (not saved)
export PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

# ---------- time budget (Kaggle sessions are capped; footer needs a margin) ----------
UIA_MAX_SECS=$(awk -v h="${MAX_HOURS:-11.5}" 'BEGIN{printf "%d", h*3600}')
UIA_MARGIN=$(( ${FOOTER_MARGIN_MIN:-10} * 60 ))
export UIA_DEADLINE=$(( UIA_START + UIA_MAX_SECS - UIA_MARGIN ))
export UIA_TIME_LEFT=$(( UIA_DEADLINE - $(date +%s) ))
[ "$UIA_TIME_LEFT" -gt 60 ] || { echo "[header] MAX_HOURS too small" >&2; exit 2; }

{
echo "=================================================================="
echo " UIA experiment ${UIA_ID}: ${STAGE_TITLE:-Stage ${UIA_STAGE}}"
echo " commit ${UIA_GIT_COMMIT} | run ${UIA_RUN_TAG} | kaggle=${UIA_ON_KAGGLE}"
echo " seeds: ${UIA_SEEDS} | eta: ${UIA_ETA} | GPU needed: ${NEEDS_GPU:-0}"
echo " estimated runtime: ${EST_RUNTIME:-unknown} | hard budget: ${MAX_HOURS:-11.5} h"
echo " outputs: ${UIA_OUT_DIR}"
echo " checkpoints: ${UIA_CKPT_DIR}"
echo "=================================================================="
} | tee "$UIA_LOG"

# ---------- dependencies ----------
if [ -f "$UIA_CODE_DIR/Stage${UIA_STAGE}/requirements.txt" ] && [ "${SKIP_PIP:-0}" != 1 ]; then
    echo "[header] pip install -r Stage${UIA_STAGE}/requirements.txt" | tee -a "$UIA_LOG"
    "$PYTHON" -m pip install -q -r "$UIA_CODE_DIR/Stage${UIA_STAGE}/requirements.txt" 2>&1 | tail -5 | tee -a "$UIA_LOG"
fi

# ---------- GPU ----------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi > "$UIA_OUT_DIR/nvidia-smi.txt" 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/[header] GPU: /' | tee -a "$UIA_LOG"
elif [ "${NEEDS_GPU:-0}" = 1 ]; then
    echo "[header] ERROR: this stage needs a GPU. In Kaggle: Settings > Accelerator > GPU." | tee -a "$UIA_LOG" >&2
    exit 3
fi

# ---------- resume from an earlier run's checkpoints ----------
# Kaggle keeps /kaggle/working only for the session. To resume: "Save Version" the notebook,
# then attach that version's output as an input; we look for <input>/ckpt/<S#>/ automatically,
# or set RESUME_FROM=/path/to/ckpt/<S#>.
if [ -z "$(ls -A "$UIA_CKPT_DIR" 2>/dev/null)" ]; then
    _src="${RESUME_FROM:-}"
    if [ -z "$_src" ] && [ -d /kaggle/input ]; then
        _src="$(find /kaggle/input -maxdepth 4 -type d -path "*/ckpt/${UIA_ID}" 2>/dev/null | head -1 || true)"
    fi
    if [ -n "$_src" ] && [ -d "$_src" ]; then
        echo "[header] resuming: copying checkpoints from $_src" | tee -a "$UIA_LOG"
        cp -r "$_src/." "$UIA_CKPT_DIR/"
    fi
fi

env | grep -E '^(UIA_|SEEDS=|ETA=|MAX_HOURS=|NEEDS_GPU=|EST_RUNTIME=|S[0-9]_)' | sort > "$UIA_OUT_DIR/config.resolved.env"
cp "$UIA_CODE_DIR/Stage${UIA_STAGE}/config.env" "$UIA_OUT_DIR/config.env"
