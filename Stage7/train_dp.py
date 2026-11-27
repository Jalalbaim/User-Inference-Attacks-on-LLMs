"""
Stage 7b (plan section 4 row 7): user-level DP fine-tuning of GPT-Neo 125M on Enron, then the same scoring as Stage 2.

User-level DP-SGD (DP-Adam), written out by hand so the privacy unit is the USER, not the document:
  every step t = 1..T:
    each member user is sampled independently with probability q (Poisson sampling over users);
    for each sampled user u: draw `docs_per_user` of u's training docs, g_u = grad of the mean token NLL on them,
        clip the WHOLE user gradient: g_u <- g_u * min(1, C / ||g_u||)   (non-finite g_u -> contributes 0);
    g = (sum_u g_u + N(0, sigma^2 C^2 I)) / (q * n_members)          (fixed denominator)
    Adam step on g, linear learning-rate decay to 0 over T.
  Adding or removing one user changes sum_u g_u by at most C in L2, so the Poisson-subsampled Gaussian mechanism
  analysis applies with users as the unit (add/remove adjacency). sigma is calibrated with Opacus' RDP accountant
  (get_noise_multiplier) for the target (eps, delta), q and T; the achieved eps is recomputed and stored.
  eps = inf ("clip") = clipping only, sigma = 0: separates the effect of clipping from the effect of noise.

No model selection: the released model is the one after step T. (Stage 2 picks the best epoch on member validation
docs, which is a function of members' data and would break the DP guarantee.) Member / non-member validation losses are
computed at steps 0, T/2, T for REPORTING ONLY.
Members are those of Stage 2's main run for the same seed (same coin flips), so 7b pairs with Stage 2 (eps = inf, no clip).

Checkpoints: <ckpt>/<label>_s<seed>/last.pt every --ckpt-every steps (resume across Kaggle sessions, exit 75 before
the deadline); the per-step sampling and noise RNGs are keyed by (seed, label, step), so a resumed run is identical.
After scoring: scores.csv.gz + history.json + DONE; last.pt is deleted; weights are kept (fp16) only with --keep-weights.

    python train_dp.py --seeds "0 1 2" --eps "1 2 4 8 inf" --out OUT --ckpt CKPT [--smoke] [--tiny]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "Stage1"), ROOT]
from common.enron_data import load_splits, members                                   # noqa: E402
import lm                                                                              # noqa: E402
from train_score import (EXIT_RESUME, atomic_torch_save, deadline_left, load_model,   # noqa: E402
                         log, set_seed)


# ------------------------------------------------------------------ privacy accounting
def calibrate(eps, delta, q, steps):
    """(sigma, achieved eps) for the Poisson-subsampled Gaussian mechanism, RDP accountant (Opacus)."""
    if math.isinf(eps):
        return 0.0, math.inf
    from opacus.accountants import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier
    sigma = get_noise_multiplier(target_epsilon=eps, target_delta=delta, sample_rate=q, steps=steps,
                                 accountant="rdp", epsilon_tolerance=0.01)
    acc = RDPAccountant()
    acc.history = [(sigma, q, steps)]
    return float(sigma), float(acc.get_epsilon(delta))


def label_of(eps):
    return "dpclip" if math.isinf(eps) else f"dp{eps:g}"


# ------------------------------------------------------------------ one model
def run_model(a, seed, eps, docs, users, tok, device, ref_cache, full_members):
    label = label_of(eps)
    tag = f"{label}_s{seed}"
    cdir = os.path.join(a.ckpt, tag)
    os.makedirs(cdir, exist_ok=True)
    done = os.path.join(cdir, "DONE")
    if os.path.exists(done):
        log(f"{tag}: already done, reusing its score table")
        return pd.read_csv(os.path.join(cdir, "scores.csv.gz"), keep_default_na=False, na_values=[""])

    mem = members(users, seed, "main")
    if a.smoke:   # a few users per class: exercises every code path in minutes
        rng0 = np.random.default_rng(seed)
        keep = set(rng0.choice(sorted(mem), a.smoke_users, replace=False)) | \
            set(rng0.choice(sorted(set(users) - mem), a.smoke_users, replace=False))
        docs = [d for d in docs if d["user"] in keep]
        mem = mem & keep
    mem_list = sorted(mem)
    n_mem = len(mem_list)
    T, q, C = a.steps, a.q, a.clip
    sigma, eps_ach = calibrate(eps, a.delta, q, T)

    enc = lambda ds: lm.encode(tok, [d["text"] for d in ds], a.max_len)
    by_user = {u: [] for u in mem_list}
    for d in docs:
        if d["role"] == "train" and d["user"] in mem:
            by_user[d["user"]].append(d)
    user_seqs = {u: enc(sorted(ds, key=lambda d: d["doc_id"])) for u, ds in by_user.items()}
    val_in = enc([d for d in docs if d["role"] == "val" and d["user"] in mem])
    val_out = enc([d for d in docs if d["role"] == "val" and d["user"] not in mem])
    held = [d for d in docs if d["role"] == "heldout"]
    log(f"{tag}: {n_mem} members ({sum(map(len, user_seqs.values()))} train docs), q={q}, T={T}, C={C}, "
        f"docs/user/step={a.docs_per_user}, sigma={sigma:.4f}, eps target {eps:g} achieved {eps_ach:.4f} "
        f"(delta={a.delta:g}, RDP, user-level add/remove); val {len(val_in)} in / {len(val_out)} out")

    set_seed(1000 * seed + 17)
    model = load_model(a.model, device, a.tiny, tok)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / T))
    amp = device.type == "cuda" and not a.fp32
    scale = 1024.0 if amp else 1.0        # static loss scale: fp16 grads, unscaled before the norm
    key = zlib.crc32(label.encode())
    hist, stats, start = [], {"users_sampled": [], "nonfinite_users": 0, "clipped_frac": [], "norm_median": []}, 0

    last = os.path.join(cdir, "last.pt")
    if os.path.exists(last):
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        hist, stats, start = st["hist"], st["stats"], st["step"]
        log(f"{tag}: resumed at step {start}/{T}")

    def evaluate(step):
        model.eval()
        r = {"step": step, "val_in": lm.mean_nll(model, val_in, device, amp=amp),
             "val_out": lm.mean_nll(model, val_out, device, amp=amp)}
        model.train()
        hist.append(r)
        log(f"{tag}: step {step}/{T} val in {r['val_in']:.4f} out {r['val_out']:.4f} (reporting only)")

    if not any(h["step"] == 0 for h in hist):
        evaluate(0)
    evals = {T // 2, T}
    acc = [torch.zeros_like(p) for p in params]
    model.train()
    t0, t_steps, n_steps_timed = time.time(), 0.0, 0
    for step in range(start, T):
        avg = t_steps / max(n_steps_timed, 1)
        if deadline_left() < max(300.0, 3 * avg + 60):
            atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                               "hist": hist, "stats": stats, "step": step}, last)
            log(f"{tag}: {deadline_left():.0f}s left; checkpoint at step {step}, stopping to resume later")
            sys.exit(EXIT_RESUME)
        ts = time.time()
        rng = np.random.default_rng([seed, key, step])
        sampled = [mem_list[i] for i in np.flatnonzero(rng.random(n_mem) < q)]
        for x in acc:
            x.zero_()
        norms = []
        for u in sampled:
            seqs = user_seqs[u]
            idx = sorted(rng.choice(len(seqs), min(a.docs_per_user, len(seqs)), replace=False).tolist())
            _, ids, mask = next(lm.batches(seqs, len(idx), idx))
            model.zero_grad(set_to_none=False)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = lm.lm_loss(model, ids.to(device), mask.to(device))
            (loss * scale).backward()
            grads = [p.grad for p in params]
            norm = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)).float())) / scale
            if not math.isfinite(norm):
                stats["nonfinite_users"] += 1        # contributes 0: still sensitivity <= C
                continue
            norms.append(norm)
            torch._foreach_add_(acc, grads, alpha=min(1.0, C / (norm + 1e-6)) / scale)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(np.random.default_rng([seed, key, step, 1]).integers(2**62)))
        denom = q * n_mem
        for p, x in zip(params, acc):
            if sigma > 0:
                x.add_(torch.randn(x.shape, generator=gen, device=device, dtype=x.dtype), alpha=sigma * C)
            p.grad = x / denom
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=False)
        stats["users_sampled"].append(len(sampled))
        stats["clipped_frac"].append(float(np.mean([n > C for n in norms])) if norms else float("nan"))
        stats["norm_median"].append(float(np.median(norms)) if norms else float("nan"))
        t_steps += time.time() - ts
        n_steps_timed += 1
        if step % 10 == 0 or step + 1 == T:
            log(f"{tag}: step {step + 1}/{T} users {len(sampled)} median ||g_u|| {stats['norm_median'][-1]:.3f} "
                f"clipped {stats['clipped_frac'][-1]:.2f} ({t_steps / n_steps_timed:.2f} s/step)")
        if step + 1 in evals and not any(h["step"] == step + 1 for h in hist):
            evaluate(step + 1)
        if (step + 1) % a.ckpt_every == 0 and step + 1 < T:
            atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                               "hist": hist, "stats": stats, "step": step + 1}, last)
        if a.smoke and step + 1 - start >= a.smoke_steps:
            break

    if a.smoke:   # extrapolate the full run from the measured time per sampled user
        per_user = t_steps / max(sum(stats["users_sampled"][-n_steps_timed:]), 1)
        est = per_user * a.q * full_members * a.full_steps
        log(f"{tag}: SMOKE speed {per_user:.3f} s per sampled user -> full model ({full_members} members, "
            f"q={a.q}, {a.full_steps} steps) about {est / 3600:.2f} h of training, plus 3 evaluations and scoring")

    # ---- score held-out docs with the FINAL model and the reference ----
    log(f"{tag}: scoring {len(held)} held-out docs with the final model")
    h_seqs = enc(held)
    lp_t, nt = lm.score(model, h_seqs, device)
    ids = [d["doc_id"] for d in held]
    ref = ref_cache(h_seqs, ids)
    lp_r = ref.loc[ids, "logp_ref"].to_numpy()
    assert (ref.loc[ids, "n_tokens"].to_numpy() == nt).all()
    k = {u: users[u]["n_train"] for u in users}
    ktot = sum(k[u] for u in mem)
    df = pd.DataFrame({
        "run": label, "seed": seed, "doc_id": ids, "user": [d["user"] for d in held],
        "b": [users[d["user"]][f"b_s{seed}"] for d in held],
        "trained": [int(d["user"] in mem) for d in held],
        "half": [users[d["user"]]["half"] for d in held],
        "k_u": [k[d["user"]] for d in held],
        "alpha_u": [k[d["user"]] / ktot if d["user"] in mem else 0.0 for d in held],
        "date": [d["date"] for d in held], "n_tokens": nt,
        "logp_theta": lp_t, "logp_ref": lp_r, "s": lp_t - lp_r,
        "eps_target": eps, "eps_achieved": eps_ach, "delta": a.delta, "sigma": sigma, "q": q, "steps": T,
        "clip": C, "docs_per_user": a.docs_per_user, "lr": a.lr,
        "model": a.model if not a.tiny else "tiny-random", "max_len": a.max_len,
    })
    df.to_csv(os.path.join(cdir, "scores.csv.gz"), index=False)
    json.dump({"hist": hist, "stats": stats, "n_members": n_mem, "sigma": sigma, "eps_target": eps,
               "eps_achieved": eps_ach, "args": vars(a)}, open(os.path.join(cdir, "history.json"), "w"), indent=1)
    if os.path.exists(last):
        os.remove(last)
    if a.keep_weights:
        (model.half() if device.type == "cuda" and not a.tiny else model).save_pretrained(os.path.join(cdir, "final"))
    open(done, "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    gap = df.groupby("trained").s.mean()
    log(f"{tag}: done in {(time.time() - t0) / 60:.1f} min. mean s(x) trained users {gap.get(1, float('nan')):.3f}, "
        f"untrained users {gap.get(0, float('nan')):.3f}")
    del model, opt, acc
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0 1 2")
    ap.add_argument("--eps", default="1 2 4 8 inf", help='target epsilons; "inf" = clipping only (sigma = 0)')
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--q", type=float, default=0.5, help="per-step sampling probability of each member user")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--clip", type=float, default=1.0, help="L2 clip norm C of each user's gradient")
    ap.add_argument("--docs-per-user", type=int, default=8, help="docs drawn from a sampled user per step")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="EleutherAI/gpt-neo-125m")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--keep-weights", action="store_true", help="save the final model (fp16 on GPU) in <tag>/final/")
    ap.add_argument("--smoke", action="store_true", help="few users and steps; prints a runtime estimate per model")
    ap.add_argument("--smoke-steps", type=int, default=6)
    ap.add_argument("--smoke-users", type=int, default=6, help="smoke: users per class")
    ap.add_argument("--tiny", action="store_true", help="random 2-layer model (local CPU test only)")
    a = ap.parse_args()
    a.full_steps = a.steps
    eps_list = [math.inf if x.lower() == "inf" else float(x) for x in a.eps.split()]
    if not eps_list or any(e <= 0 for e in eps_list):
        raise SystemExit(f"--eps: positive numbers or inf (got {a.eps!r})")
    if not (0 < a.q <= 1) or a.steps < 1 or a.clip <= 0:
        raise SystemExit("need 0 < q <= 1, steps >= 1, clip > 0")
    if a.smoke:   # smoke runs never mix with real checkpoints
        a.ckpt = os.path.join(a.ckpt, "smoke")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.ckpt, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + f", torch {torch.__version__}")
    docs, users, sdir = load_splits()
    log(f"splits from {sdir}: {len(docs)} docs, {len(users)} users (sha256 checked)")

    seeds = [int(x) for x in a.seeds.split()]
    # noise table for the FULL member sets, before any training (q and T fixed; the member count does not enter sigma)
    for e in eps_list:
        s, ea = calibrate(e, a.delta, a.q, a.steps)
        log(f"calibration eps {e:g}: sigma {s:.4f}, achieved eps {ea:.4f} (delta {a.delta:g}, q {a.q}, T {a.steps}); "
            + ", ".join(f"seed {sd}: {len(members(users, sd, 'main'))} members, noise/signal "
                        f"{s / (a.q * len(members(users, sd, 'main'))):.3f}" for sd in seeds))

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
    for s in seeds:                       # seed-major: a partial run has every eps for its finished seeds
        for e in eps_list:
            frames.append(run_model(a, s, e, docs, users, tok, device, ref_cache, len(members(users, s, "main"))))
            pd.concat(frames).to_csv(os.path.join(a.out, "scores.csv.gz"), index=False)
            tag = f"{label_of(e)}_s{s}"
            shutil.copy(os.path.join(a.ckpt, tag, "history.json"), os.path.join(a.out, f"history_{tag}.json"))
    log(f"all models done: {os.path.join(a.out, 'scores.csv.gz')}")


if __name__ == "__main__":
    main()
