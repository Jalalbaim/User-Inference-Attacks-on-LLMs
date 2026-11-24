"""
Deterministic m-document subsets and orderings per user (plan Step D, Stage 2 onwards).

They depend only on (user, m, draw, order), never on the seed or the model. So:
  - the reference model's S_cat is computed once and shared by every seed;
  - T-hat, S_cat, the aggregator and the blind baseline all see the SAME documents for a given (user, m, draw),
    which makes their comparison paired.
At m >= the number of held-out docs (m = 32) there is a single draw (all docs). At m = 1 there is a single order.
"""
from __future__ import annotations

import zlib

import numpy as np

M_GRID = (1, 2, 4, 8, 16, 32)
SEP = "|"   # joins doc ids in tables (Enron doc ids are tarball paths: no '|')


def _key(user: str) -> int:
    return zlib.crc32(user.encode("utf-8"))


def subsets(heldout_ids, user, m, draws):
    """List of subsets (each a sorted list of doc ids) of size m."""
    ids = sorted(heldout_ids)
    if m >= len(ids):
        return [ids]
    out = []
    for d in range(draws):
        rng = np.random.default_rng([7, _key(user), m, d])
        out.append([ids[i] for i in sorted(rng.choice(len(ids), m, replace=False))])
    return out


def orders(user, m, d, R):
    """R permutations of range(m) (one if m == 1)."""
    if m == 1:
        return [[0]]
    return [np.random.default_rng([8, _key(user), m, d, r]).permutation(m).tolist() for r in range(R)]


def units(heldout_by_user, m_grid=M_GRID, draws=4, R=4):
    """Rows (user, m, draw, order, [doc ids in concatenation order])."""
    rows = []
    for u in sorted(heldout_by_user):
        for m in m_grid:
            for d, sub in enumerate(subsets(heldout_by_user[u], u, m, draws)):
                for r, o in enumerate(orders(u, len(sub), d, R)):
                    rows.append((u, m, d, r, [sub[i] for i in o]))
    return rows
