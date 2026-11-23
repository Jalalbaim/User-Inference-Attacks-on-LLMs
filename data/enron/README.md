# data/enron

**Decisions (agreed 2026-11-23):**
1. **Source:** the CMU original, https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
   (443,254,787 bytes, Last-Modified 2015-05-07). Put it in `data/raw/`, which is git-ignored.
2. **User:** the sender (the `From:` address), as in Kandpal et al. Senders outside the 150 mailboxes count too.
   Merging aliases of one person is optional and comes later.
3. **Cleaning:** strip the headers; cut each body at the quote or forward markers (keep only the sender's own
   words); remove duplicates on (sender, cleaned text); drop empty and very short bodies.
4. **Filter:** at least 150 docs per user (Kandpal), plus 32 fixed held-out docs per user and a validation set.
   Report the counts after every step. If far fewer than about 136 users survive, stop and rethink (plan, section 6, item 4).
5. **Document:** one cleaned body, truncated to the first 128 tokens. Keep the date of each email (needed for Stage 5).

Known from Kandpal et al. (Table 1 / App. D): 136 users, about 91K emails, examples per user
P0/P25/P50/P75/P100 = 28/107/279/604/4280, minimum 150 per user. These numbers are internally
inconsistent (P0 = 28 < 150; the appendix says 138 users). We report our own counts.

## Step 1 results (`prepare.py`, defaults, 2026-11-23)

The first run takes about 5 min (one pass over the tarball, cached in `data/processed/enron/_parsed.pkl.gz`).
Reruns take about 30 s; use `--reparse` to ignore the cache.

| Step | Count |
|---|---|
| Files in the tarball | 517,401 |
| Bodies cut at a quote or forward marker | 228,458 |
| After dropping bodies under 5 words | 435,741 |
| Aliases merged (same display name, `X-From`) | 1,990 addresses folded into 1,230 people (89 of them among the kept users) |
| After deduplicating on (user, text) | 205,682 docs, 17,706 users |
| Users with ≥ 50 / 100 / 150 / 200 / 300 docs | 564 / 271 / **175** / 135 / 95 |
| Kept (≥ 150 docs) | **175 users, 101,817 docs**; docs per user P0/25/50/75/100 = 152/205/342/619/4164 |

Words per doc: P25/50/75 = 15/30/67. Most docs are shorter than 128 tokens.
Dates: 202 docs fall outside 1997–2002 (e.g. 1979). This matters for the Stage 5 temporal split.

**Decided (2026-11-23):**
- **Keep non-person and bulk senders** (`outlook.team@`, `enron.announcements@`, newsletters, …).
- **Merge aliases.** The key is the display name reduced to "first last", with Jr/Sr/II/III/IV removed.
  The canonical id is the address with the most emails. Every merge is listed in `data/processed/enron/merges.txt`
  (`*` = kept user). Kept merges that look uncertain, left merged:
  - `mjones7@txu.com` + `melissa.jones@enron.com`;
  - three PMA addresses under "The Association";
  - `webmaster@trader.ca` (1 email) under John Zufferli.

Files in `data/processed/enron/`:
- `docs.jsonl.gz`: one line per kept doc, with `user`, `address`, `date`, `msg_id`, `path`, `text`;
- `counts.json`;
- `merges.txt`;
- `samples.txt`: cleaning examples, raw vs kept.

## Step 2: splits (`splits.py`, about 10 s)

**Fixed for all seeds:**
- Per user: 32 random **held-out** docs (attacker only, never trained); **val** = 5% of the rest, at least 4; **train** = the rest.
- FIT/EVAL: a random half of the users, 87 FIT and 88 EVAL.

**Per seed s = 0..9:** an independent fair coin per user, b = 1 means member. Only members' train docs are used for training.

Certification rule (agreed 2026-11-23): T̂ and S_cat have nothing to fit, so they are certified on all users.
FIT/EVAL is used only by learned statistics (the aggregator). One headline certificate is chosen before the runs;
every other row is descriptive.

| | Count |
|---|---|
| Docs by role | 5,600 held-out / 4,810 val / 91,407 train |
| Train docs per user | min 114, median 294, max 3,925 |
| Members per seed | 70 to 100 (seeds 8 and 7). The certificate threshold changes by < 0.01 vs a balanced split |
| Member train docs per seed | 33K to 55K (Kandpal: 41K) |
| α_u max per seed | 0.067 to 0.097 |
| Non-member held-out docs found verbatim in member train docs | 3 to 20 per seed (2 to 6 users) |

**Committed to git** (option A, 2026-11-23): `data/processed/enron/splits/` is the only processed folder tracked.
`docs.jsonl.gz` sha256 = `115dabb94c400a9bcd9962b41a1212622de8fd0b94c704d4a59c7f4918f87ac1`. Stage 1 checks this before training.
If the splits are regenerated, update this hash.

Outputs in `data/processed/enron/splits/`: `docs.jsonl.gz` (doc_id, user, role, date, address, text), `users.csv`, `summary.json`.
