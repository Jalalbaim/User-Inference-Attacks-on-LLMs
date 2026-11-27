# footer.sh -- inlined at the end of the built script; runs even if the stage failed or timed out.
# Writes the manifest, checks the per-document score table, archives everything to /kaggle/working.
set +e +o pipefail          # the footer must never abort: it is what saves the outputs

UIA_END=$(date +%s)
UIA_RUNTIME=$(( UIA_END - UIA_START ))
case "$UIA_RC" in
    0)   UIA_STATUS=ok ;;
    124|75) UIA_STATUS=time_budget_reached ;;   # 124: killed by timeout; 75: stage stopped itself to resume
    *)   UIA_STATUS=failed ;;
esac

# ---------- per-document score table (plan Step C) ----------
UIA_SCORE_STATUS="not_expected"
if [ -n "${SCORE_TABLE:-}" ]; then
    if [ -f "$UIA_OUT_DIR/$SCORE_TABLE" ]; then
        UIA_SCORE_STATUS="present ($(du -h "$UIA_OUT_DIR/$SCORE_TABLE" | cut -f1))"
    else
        UIA_SCORE_STATUS="MISSING"
    fi
fi

# ---------- manifest + environment ----------
"$PYTHON" -m pip freeze > "$UIA_OUT_DIR/pip_freeze.txt" 2>/dev/null || true
"$PYTHON" - <<__UIA_PY__ > "$UIA_OUT_DIR/manifest.json" 2>/dev/null || true
import json, platform, sys
print(json.dumps({
    "id": "$UIA_ID", "run_tag": "$UIA_RUN_TAG", "git_commit": "$UIA_GIT_COMMIT",
    "status": "$UIA_STATUS", "exit_code": $UIA_RC, "runtime_s": $UIA_RUNTIME,
    "start_epoch": $UIA_START, "end_epoch": $UIA_END, "seeds": "$UIA_SEEDS", "eta": "$UIA_ETA",
    "on_kaggle": $UIA_ON_KAGGLE, "score_table": "${SCORE_TABLE:-}", "score_table_status": "$UIA_SCORE_STATUS",
    "python": sys.version.split()[0], "platform": platform.platform(),
}, indent=2))
__UIA_PY__

# ---------- summary (written BEFORE the zip, so the zipped run.log is complete) ----------
UIA_ZIP="$UIA_WORK/outputs/$UIA_RUN_TAG.zip"
{
echo "=================================================================="
echo " ${UIA_ID} finished: ${UIA_STATUS} (exit ${UIA_RC}) after $(( UIA_RUNTIME / 3600 ))h$(( UIA_RUNTIME % 3600 / 60 ))m$(( UIA_RUNTIME % 60 ))s"
echo " per-document score table: ${UIA_SCORE_STATUS}"
echo " outputs: ${UIA_OUT_DIR}"
[ "$UIA_STATUS" = time_budget_reached ] && echo " budget reached: checkpoints in ${UIA_CKPT_DIR}; Save Version, attach the output, rerun the same command to resume."
_used=$(du -sm "$UIA_WORK" 2>/dev/null | cut -f1 || echo 0)
[ "${_used:-0}" -gt 15000 ] && echo " WARNING: ${UIA_WORK} uses ${_used} MB (Kaggle output limit ~20 GB)."
} | tee -a "$UIA_LOG"

# ---------- archive: ONE .zip with every output of this experiment id ----------
#   outputs/<ID>_*/   every run folder of this id on this machine (this run, earlier runs, smoke runs)
#   ckpt/<ID>/        everything: score tables, histories, DONE markers, reference scores, resume state
# Model weights (.safetensors/.bin/.pt/...) are included only with ZIP_WEIGHTS=1; otherwise they are listed as skipped.
"$PYTHON" - "$UIA_WORK/outputs" "$UIA_ID" "${UIA_CKPT_DIR:-}" "$UIA_ZIP" "${ZIP_WEIGHTS:-0}" <<'__UIA_ZIP__' 2>&1 | tee -a "$UIA_LOG" || true
import glob, os, sys, zipfile
outs, uid, ckpt, dst, zw = sys.argv[1:6]
WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
tops = [(d, os.path.join("outputs", os.path.basename(d))) for d in sorted(glob.glob(os.path.join(outs, uid + "_*")))
        if os.path.isdir(d)]
if ckpt and os.path.isdir(ckpt):
    tops.append((ckpt, os.path.join("ckpt", uid)))
n, skipped, skipped_mb = 0, 0, 0.0
tmp = dst + ".part"
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
    for top, arc in tops:
        for d, dirs, files in os.walk(top):
            for f in sorted(files):
                p = os.path.join(d, f)
                if f.endswith(WEIGHTS) and zw != "1":
                    skipped += 1
                    skipped_mb += os.path.getsize(p) / 1e6
                    continue
                z.write(p, os.path.join(arc, os.path.relpath(p, top)))
                n += 1
os.replace(tmp, dst)
print(f" zip: {n} files from {len(tops)} folders, {os.path.getsize(dst) / 1e6:.1f} MB -> {dst}")
if skipped:
    print(f" zip: {skipped} model-weight files ({skipped_mb:.0f} MB) NOT included; rerun the footer with ZIP_WEIGHTS=1 to include them")
__UIA_ZIP__

{
echo " archive (download this): ${UIA_ZIP}"
[ "$UIA_ON_KAGGLE" = 1 ] && echo " Kaggle: /kaggle/working is kept only if you 'Save Version' (or download the archive now)."
echo "=================================================================="
} | tee -a "$UIA_LOG"
