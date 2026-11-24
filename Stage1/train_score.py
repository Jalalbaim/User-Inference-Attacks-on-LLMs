"""
Stage 1 (plan section 4, row 1; Steps A-C): fine-tune GPT-Neo 125M on one seed's members, keep the epoch with the
best member-validation loss, then score every held-out document with that model and the pretrained reference.

Settings agreed with the user (Kandpal et al.): batch 8, lr 5e-5, AdamW, linear decay, no warmup, all epochs run (4 from 2026-11-24; the seed-0 pilot used 10),
best epoch by validation loss (no early stopping), fp16 mixed precision on GPU, 128 tokens per document.

Runs:
  main  members = users with b_s<seed> = 1; all 175 users are scored.
  null  members = b_s<seed> = 1 AND half = FIT; EVAL users are never trained on (plan Step B.4).
        All users are scored; the null analysis uses EVAL users only, with their b_s<seed> as fake labels.

Checkpoints (resume across Kaggle sessions): <ckpt>/<run>_s<seed>/
  last.pt         model + optimizer + scheduler + scaler + history, written after every epoch
  best/           best-epoch weights (save_pretrained); after scoring, kept in fp16 unless --keep-fp32
  scores.csv.gz   the per-document score table of this run/seed; DONE marks a finished seed
Reference scores (pretrained model, seed-independent) are cached in <ckpt>/ref_scores_<len>.csv.gz.

    python train_score.py --seeds 0 --run main --out OUT --ckpt CKPT [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.enron_data import load_splits, members   # noqa: E402
import lm                                              # noqa: E402

EXIT_RESUME = 75   # stopped before the deadline with a checkpoint: rerun the same command to resume


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_model(name_or_path, device, tiny=False, tokenizer=None):
    from transformers import AutoModelForCausalLM, GPTNeoConfig, GPTNeoForCausalLM
    if tiny:   # local CPU tests only: a 2-layer random GPT-Neo with the real tokenizer
        cfg = GPTNeoConfig(vocab_size=len(tokenizer), hidden_size=64, num_layers=2, num_heads=2,
                           attention_types=[[["global", "local"], 1]], max_position_embeddings=256,
                           window_size=64)
        torch.manual_seed(0)
        return GPTNeoForCausalLM(cfg).to(device)
    return AutoModelForCausalLM.from_pretrained(name_or_path).to(device)


def deadline_left():
    d = os.environ.get("UIA_DEADLINE")
    return float(d) - time.time() if d else float("inf")


def run_seed(a, seed, docs, users, tok, device, ref_cache):
    tag = f"{a.run}_s{seed}"
    cdir = os.path.join(a.ckpt, tag)
    os.makedirs(cdir, exist_ok=True)
    done = os.path.join(cdir, "DONE")
    if os.path.exists(done):
        log(f"{tag}: already done, reusing its score table")
        return pd.read_csv(os.path.join(cdir, "scores.csv.gz"))

    mem = members(users, seed, a.run)
    full_train = sum(1 for d in docs if d["role"] == "train" and d["user"] in mem)
    if a.smoke:   # a few users, a few docs each: exercises every code path in minutes
        rng = random.Random(seed)
        keep = set(rng.sample(sorted(mem), a.smoke_users)) | set(rng.sample(sorted(set(users) - mem), a.smoke_users))
        docs = [d for d in docs if d["user"] in keep]
        mem = mem & keep
    train = [d for d in docs if d["role"] == "train" and d["user"] in mem]
    if a.smoke:
        train = train[:a.smoke_train_docs]
    val_in = [d for d in docs if d["role"] == "val" and d["user"] in mem]
    val_out = [d for d in docs if d["role"] == "val" and d["user"] not in mem]
    held = [d for d in docs if d["role"] == "heldout"]
    log(f"{tag}: {len(mem)} members, {len(train)} train docs, val {len(val_in)} in / {len(val_out)} out, "
        f"{len(held)} held-out docs to score")

    enc = lambda ds: lm.encode(tok, [d["text"] for d in ds], a.max_len)
    tr_seqs, vi_seqs, vo_seqs = enc(train), enc(val_in), enc(val_out)
    ntok = np.array([len(s) - 1 for s in tr_seqs])
    log(f"{tag}: train tokens per doc mean {ntok.mean():.1f}, {100 * (ntok == a.max_len).mean():.1f}% truncated")

    set_seed(1000 * seed + 17)
    model = load_model(a.model, device, a.tiny, tok)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    steps_per_epoch = lm.n_steps(len(tr_seqs), a.bs)
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / total))
    amp = device.type == "cuda" and not a.fp32
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    hist, start_epoch, best = [], 0, {"epoch": None, "val_in": float("inf")}

    last = os.path.join(cdir, "last.pt")
    if os.path.exists(last):
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        hist, start_epoch, best = st["hist"], st["epoch"] + 1, st["best"]
        log(f"{tag}: resumed after epoch {st['epoch'] + 1}/{a.epochs}")

    if start_epoch == 0:   # epoch-0 validation = the pretrained model, for reference
        hist.append({"epoch": 0, "train_loss": None, "val_in": lm.mean_nll(model, vi_seqs, device, amp=amp),
                     "val_out": lm.mean_nll(model, vo_seqs, device, amp=amp), "secs": 0.0})
        log(f"{tag}: pretrained val loss in {hist[0]['val_in']:.4f} out {hist[0]['val_out']:.4f}")

    for ep in range(start_epoch, a.epochs):
        prev = [h["secs"] for h in hist if h["epoch"] > 0]
        need = 1.3 * (prev[-1] if prev else 0) + 120
        if deadline_left() < need:
            log(f"{tag}: {deadline_left():.0f}s left, the next epoch needs about {need:.0f}s; stopping to resume later")
            sys.exit(EXIT_RESUME)
        t0 = time.time()
        model.train()
        order = np.random.default_rng([seed, ep]).permutation(len(tr_seqs)).tolist()
        run_loss, run_n = 0.0, 0
        for step, (_, ids, mask) in enumerate(lm.batches(tr_seqs, a.bs, order)):
            ids, mask = ids.to(device), mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = lm.lm_loss(model, ids, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep + 1} step {step}")
            run_loss += loss.item()
            run_n += 1
            if step % 500 == 0:
                log(f"{tag}: epoch {ep + 1} step {step}/{steps_per_epoch} loss {loss.item():.4f} "
                    f"({(time.time() - t0) / (step + 1):.3f} s/step)")
            if a.smoke and step + 1 >= a.smoke_steps:
                break
        t_train = time.time() - t0
        vi = lm.mean_nll(model, vi_seqs, device, amp=amp)
        vo = lm.mean_nll(model, vo_seqs, device, amp=amp)
        secs = time.time() - t0
        hist.append({"epoch": ep + 1, "train_loss": run_loss / max(run_n, 1), "val_in": vi, "val_out": vo,
                     "secs": secs, "train_secs": t_train, "steps": run_n,
                     "sec_per_step": t_train / max(run_n, 1)})
        log(f"{tag}: epoch {ep + 1}/{a.epochs} train {run_loss / max(run_n, 1):.4f} val in {vi:.4f} "
            f"out {vo:.4f} ({secs / 60:.1f} min)")
        if vi < best["val_in"]:
            best = {"epoch": ep + 1, "val_in": vi}
            tmp = os.path.join(cdir, "best.tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained(tmp)
            shutil.rmtree(os.path.join(cdir, "best"), ignore_errors=True)
            os.replace(tmp, os.path.join(cdir, "best"))
        atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                           "scaler": scaler.state_dict(), "hist": hist, "epoch": ep, "best": best}, last)

    if a.smoke:   # extrapolate the full-run time from the measured speed
        sps = [h["sec_per_step"] for h in hist if h["epoch"] > 0]
        est = lm.n_steps(full_train, a.bs) * float(min(sps)) * a.full_epochs
        log(f"{tag}: SMOKE speed {min(sps):.3f} s/step -> full run ({full_train} train docs, "
            f"{a.full_epochs} epochs) "
            f"about {est / 3600:.2f} h of training, plus validation and scoring")

    # ---- score held-out docs with the best model and the reference ----
    log(f"{tag}: best epoch {best['epoch']} (val in {best['val_in']:.4f}); scoring {len(held)} held-out docs")
    del model, opt
    torch.cuda.empty_cache() if device.type == "cuda" else None
    model = load_model(os.path.join(cdir, "best"), device)   # saved config: works for the tiny model too
    h_seqs = enc(held)
    lp_t, nt = lm.score(model, h_seqs, device)
    ids = [d["doc_id"] for d in held]
    ref = ref_cache(h_seqs, ids)
    lp_r = ref.loc[ids, "logp_ref"].to_numpy()
    assert (ref.loc[ids, "n_tokens"].to_numpy() == nt).all()

    k = {u: users[u]["n_train"] for u in users}
    ktot = sum(k[u] for u in mem)
    df = pd.DataFrame({
        "run": a.run, "seed": seed, "doc_id": ids, "user": [d["user"] for d in held],
        "b": [users[d["user"]][f"b_s{seed}"] for d in held],
        "trained": [int(d["user"] in mem) for d in held],
        "half": [users[d["user"]]["half"] for d in held],
        "k_u": [k[d["user"]] for d in held],
        "alpha_u": [k[d["user"]] / ktot if d["user"] in mem else 0.0 for d in held],
        "date": [d["date"] for d in held], "n_tokens": nt,
        "logp_theta": lp_t, "logp_ref": lp_r, "s": lp_t - lp_r,
        "best_epoch": best["epoch"], "model": a.model if not a.tiny else "tiny-random", "max_len": a.max_len,
    })
    df.to_csv(os.path.join(cdir, "scores.csv.gz"), index=False)
    json.dump({"hist": hist, "best": best, "n_members": len(mem), "n_train_docs": len(train),
               "args": vars(a)}, open(os.path.join(cdir, "history.json"), "w"), indent=1)

    # keep only what later stages need: best weights in fp16; drop the optimizer state
    if not a.keep_last:
        os.remove(last)
    if device.type == "cuda" and not a.tiny and not a.keep_fp32:
        model.half().save_pretrained(os.path.join(cdir, "best"))
    open(done, "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    gap = df.groupby("trained").s.mean()
    log(f"{tag}: done. mean s(x) trained users {gap.get(1, float('nan')):.3f}, "
        f"untrained users {gap.get(0, float('nan')):.3f}")
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--run", default="main", choices=["main", "null"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="EleutherAI/gpt-neo-125m")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--fp32", action="store_true", help="disable fp16 mixed precision")
    ap.add_argument("--keep-last", action="store_true", help="keep last.pt (optimizer) after a seed finishes")
    ap.add_argument("--keep-fp32", action="store_true",
                    help="keep best/ in fp32 (Stage 2: S_cat must use the exact weights that produced the score table)")
    ap.add_argument("--smoke", action="store_true", help="6 users, few docs and steps; prints a runtime estimate")
    ap.add_argument("--smoke-steps", type=int, default=60)
    ap.add_argument("--smoke-users", type=int, default=3, help="smoke: users per class")
    ap.add_argument("--smoke-train-docs", type=int, default=600)
    ap.add_argument("--tiny", action="store_true", help="random 2-layer model (local CPU test only)")
    a = ap.parse_args()
    a.full_epochs = a.epochs
    if a.smoke:
        a.epochs = 2
    if a.smoke:   # smoke runs must never mix with real checkpoints
        a.ckpt = os.path.join(a.ckpt, "smoke")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.ckpt, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + f", torch {torch.__version__}")
    if device.type != "cuda" and not a.tiny:
        log("WARNING: no GPU; the real model on CPU is far too slow for a full run")

    docs, users, sdir = load_splits()
    log(f"splits from {sdir}: {len(docs)} docs, {len(users)} users (sha256 checked)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    def ref_cache(h_seqs, ids):
        p = os.path.join(a.ckpt, f"ref_scores_{a.max_len}{'_tiny' if a.tiny else ''}.csv.gz")
        if os.path.exists(p):
            r = pd.read_csv(p).set_index("doc_id")
            if set(ids) <= set(r.index):
                return r
        log(f"scoring {len(h_seqs)} held-out docs with the reference (pretrained) model")
        m = load_model(a.model, device, a.tiny, tok)
        lp, nt = lm.score(m, h_seqs, device)
        del m
        r = pd.DataFrame({"doc_id": ids, "logp_ref": lp, "n_tokens": nt}).set_index("doc_id")
        r.to_csv(p)
        return r

    frames = []
    for s in [int(x) for x in a.seeds.split()]:
        frames.append(run_seed(a, s, docs, users, tok, device, ref_cache))
        pd.concat(frames).to_csv(os.path.join(a.out, "scores.csv.gz"), index=False)   # updated after each seed
        cdir = os.path.join(a.ckpt, f"{a.run}_s{s}")
        shutil.copy(os.path.join(cdir, "history.json"), os.path.join(a.out, f"history_{a.run}_s{s}.json"))
    log(f"all seeds done: {os.path.join(a.out, 'scores.csv.gz')}")


if __name__ == "__main__":
    main()
