#!/usr/bin/env bash
# Entry point. From a Kaggle notebook cell:
#   !curl -sSLO https://raw.githubusercontent.com/Jalalbaim/User-Inference-Attacks-on-LLMs/master/kaggle/run_kaggle.sh && bash run_kaggle.sh S0
#   !bash run_kaggle.sh S2 SEEDS="0 1 2" MAX_HOURS=11      # KEY=VALUE pairs override StageN/config.env
# Ids: S0..S7 (= folders Stage0..Stage7); "Stage3", "s3" and "3" also work.
#
# Env knobs: REPO_URL, REPO_REF (branch/tag/commit, default master), GITHUB_TOKEN (private repo;
# also read from the Kaggle secret GITHUB_TOKEN), UIA_REPO_DIR (where to clone, default /tmp/uia_repo).
set -euo pipefail

ID_RAW="${1:-}"
[ -n "$ID_RAW" ] || { echo "usage: bash run_kaggle.sh S<0-7> [KEY=VALUE ...]" >&2; exit 2; }
shift
N="$(printf '%s' "$ID_RAW" | sed -E 's/^([Ss]tage|[Ss])?//')"
[[ "$N" =~ ^[0-7]$ ]] || { echo "unknown experiment id '$ID_RAW' (expected S0..S7)" >&2; exit 2; }

REPO_URL="${REPO_URL:-https://github.com/Jalalbaim/User-Inference-Attacks-on-LLMs.git}"
REPO_REF="${REPO_REF:-master}"

# 1. Find the code: use the clone this script lives in, else clone/update a disposable copy.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HERE/build.sh" ] && [ -d "$HERE/../common" ]; then
    REPO="$(cd "$HERE/.." && pwd)"
    echo "[run_kaggle] using local checkout $REPO"
else
    REPO="${UIA_REPO_DIR:-/tmp/uia_repo}"
    if [ -z "${GITHUB_TOKEN:-}" ]; then   # optional Kaggle secret, silently skipped if absent
        GITHUB_TOKEN="$(python3 -c 'from kaggle_secrets import UserSecretsClient as U; print(U().get_secret("GITHUB_TOKEN"))' 2>/dev/null || true)"
    fi
    URL="$REPO_URL"
    [ -n "${GITHUB_TOKEN:-}" ] && URL="${REPO_URL/https:\/\//https://x-access-token:${GITHUB_TOKEN}@}"
    if [ -d "$REPO/.git" ]; then
        git -C "$REPO" fetch -q --depth 1 origin "$REPO_REF"
        git -C "$REPO" checkout -q --force FETCH_HEAD
    else
        rm -rf "$REPO"
        git clone -q --depth 1 --branch "$REPO_REF" "$URL" "$REPO" 2>/dev/null \
          || { git clone -q "$URL" "$REPO" && git -C "$REPO" checkout -q "$REPO_REF"; }
    fi
    echo "[run_kaggle] code at $REPO ($(git -C "$REPO" rev-parse --short HEAD))"
fi

# 2. Assemble the single runnable script, 3. run it (overrides passed through).
if [ -d /kaggle/working ]; then WORK=/kaggle/working; else WORK="${UIA_WORK:-$PWD/_runs}"; fi
mkdir -p "$WORK/build"
OUT="$WORK/build/run_S${N}.sh"
bash "$REPO/kaggle/build.sh" "$N" "$OUT"
exec bash "$OUT" "$@"
