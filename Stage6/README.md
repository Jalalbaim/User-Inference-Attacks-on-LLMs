# Stage 6: pool identity, Prop. 3 (sanity check only)

**Status: DONE (2026-11-25), sanity check passed.**
- At π = 0 the bound was positive in 1 of 30,000 pools.
- At π = 1 the lower bound on J and π is +0.10 with 35-user pools (positive in 87% of pools) and ≈ 0 with 15-user pools.
- Report this as the method's limit at Enron's scale. Details are in the notes cell of `stage6.ipynb`.

**Plan:** `docs/experiment_plan_enron_gptneo125m.md`, section 4 row 6, Step I. No GPU: post-processing of Stage 2 (statistic at m = 32).

| File | What it is |
|---|---|
| `analysis.py` | Per seed: the reference set is a random half of the non-members. The pool has n_p users ∈ {15, 25, 35}, a fraction π ∈ {0, .25, .5, .75, 1} of them members, 500 pools each. The bound is the one-sided KS-hat − ε(n_p) − ε(n_ref) |
| `stage6.ipynb` | Tables, figure, notes |
