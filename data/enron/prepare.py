"""
Enron preparation, step 1: raw CMU tarball -> one cleaned document per (sender, email), with counts.

Decisions: data/enron/README.md (agreed 2026-11-23). This step does NOT split users or tokenize;
splits (held-out 32, validation, FIT/EVAL) and the 128-token truncation come later.

Reads the .tar.gz directly (no unpacking: the repo lives in OneDrive).
Writes data/processed/enron/{docs.jsonl.gz, counts.json, samples.txt}.

    .venv/Scripts/python data/enron/prepare.py [--min-words 5] [--min-docs 150] [--heldout 32]
"""
from __future__ import annotations

import argparse
import collections
import email
import email.utils
import gzip
import json
import os
import pickle
import random
import re
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(ROOT, "data", "raw", "enron_mail_20150507.tar.gz")
OUT = os.path.join(ROOT, "data", "processed", "enron")

# A line that starts text the sender did not write: everything from it on is cut.
CUT = re.compile(
    r"^\s*(-{2,}\s*Original Message"          # Outlook reply
    r"|-{2,}.*Forwarded by"                     # Lotus / Outlook forward
    r"|-{5,}\s*Forwarded"                       # "---------- Forwarded message"
    r"|From:\s"                                 # quoted header block
    r"|To:\s"                                   # Lotus Notes reply header (name / date / To:)
    r"|Sent by:\s"
    r"|>"                                       # '>' quoting
    r"|.*\bwrote:\s*$)",                        # "On ..., X wrote:"
    re.IGNORECASE)
# Lotus Notes puts "<name>\n<MM/DD/YYYY HH:MM AM>" just above "To:"; drop those lines too.
LOTUS_DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)", re.IGNORECASE)


def clean_body(body: str) -> tuple[str, bool]:
    """Keep only the sender's own words. Returns (text, was_cut)."""
    lines = body.replace("\r\n", "\n").split("\n")
    cut = next((i for i, l in enumerate(lines) if CUT.match(l)), None)
    if cut is not None:
        lines = lines[:cut]
        # strip a trailing Lotus "name / date" block left above the cut point
        tail = [i for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1) if lines[i].strip()]
        if tail and LOTUS_DATE.search(lines[tail[0]]):
            lines = lines[:tail[-1]] if len(tail) > 1 else lines[:tail[0]]
    text = "\n".join(l.rstrip() for l in lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, cut is not None


def body_of(msg) -> str:
    if msg.is_multipart():   # rare in this release
        parts = [p.get_payload(decode=True) for p in msg.walk() if p.get_content_type() == "text/plain"]
        return b"\n".join(p for p in parts if p).decode("latin-1")
    p = msg.get_payload(decode=True)
    return p.decode("latin-1") if isinstance(p, bytes) else str(msg.get_payload())


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def pct(v, qs=(0, 25, 50, 75, 100)):
    v = sorted(v)
    return {f"P{q}": v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] for q in qs} if v else {}


def parse_tarball(t0):
    recs, n_files, n_nofrom, n_cut = [], 0, 0, 0
    with tarfile.open(RAW, "r:gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            n_files += 1
            msg = email.message_from_bytes(tar.extractfile(m).read())
            sender = email.utils.parseaddr(msg.get("From", ""))[1].strip().lower()
            if not sender:
                n_nofrom += 1
                continue
            raw_body = body_of(msg)
            text, was_cut = clean_body(raw_body)
            n_cut += was_cut
            try:
                date = email.utils.parsedate_to_datetime(msg.get("Date")).isoformat()
            except Exception:
                date = None
            recs.append({"address": sender, "xfrom": msg.get("X-From", "") or "", "date": date,
                         "msg_id": msg.get("Message-ID", "").strip(), "path": m.name, "text": text,
                         "_cut": was_cut})
            if n_files % 100000 == 0:
                print(f"  parsed {n_files} files ({time.time() - t0:.0f}s)", flush=True)
    return recs, n_files, n_nofrom, n_cut


SUFFIX = {"jr", "sr", "ii", "iii", "iv"}


def name_key(xfrom: str):
    """'Lavorato, John J. </O=ENRON/...>' and 'John J Lavorato@ECT' -> 'john lavorato'. None if no usable name."""
    s = re.sub(r"<.*?>|\(.*?\)|[\"']", " ", str(xfrom)).split("<")[0]
    s = re.sub(r"@\S*", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", " ", s, flags=re.IGNORECASE).strip(" ,")
    if "," in s:
        last, first = s.split(",", 1)
    else:
        parts = s.split()
        if len(parts) < 2:
            return None
        first, last = " ".join(parts[:-1]), parts[-1]
    f = [t for t in re.findall(r"[a-z]+", first.lower()) if len(t) > 1]
    l = [t for t in re.findall(r"[a-z]+", last.lower()) if len(t) > 1 and t not in SUFFIX]
    f = [t for t in f if t not in SUFFIX]
    return f"{f[0]} {l[-1]}" if f and l else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-words", type=int, default=5, help="drop cleaned bodies with fewer words")
    ap.add_argument("--min-docs", type=int, default=150, help="Kandpal's per-user minimum")
    ap.add_argument("--heldout", type=int, default=32, help="fixed held-out docs per user")
    ap.add_argument("--seed", type=int, default=0, help="only for picking the printed samples")
    ap.add_argument("--reparse", action="store_true", help="ignore the parse cache")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    c = collections.OrderedDict()
    t0 = time.time()

    # 1. parse every file (cached: the tarball pass takes about 5 min)
    cache = os.path.join(OUT, "_parsed.pkl.gz")
    parsed_now = not (os.path.exists(cache) and not a.reparse)
    if not parsed_now:
        with gzip.open(cache, "rb") as f:
            recs, n_files, n_nofrom, n_cut = pickle.load(f)
    else:
        recs, n_files, n_nofrom, n_cut = parse_tarball(t0)
        with gzip.open(cache, "wb") as f:
            pickle.dump((recs, n_files, n_nofrom, n_cut), f)
    c["1_files_in_tarball"] = n_files
    c["1_no_sender_dropped"] = n_nofrom
    c["1_bodies_cut_at_quote_or_forward"] = n_cut

    # 2. drop empty / very short own text
    recs = [r for r in recs if len(r["text"].split()) >= a.min_words]
    c[f"2_after_min_{a.min_words}_words"] = len(recs)

    # 3a. merge addresses of one person: same display-name key (most common X-From name per address)
    names = collections.defaultdict(collections.Counter)
    for r in recs:
        names[r["address"]][name_key(r["xfrom"])] += 1
    n_addr = collections.Counter(r["address"] for r in recs)
    addr_key = {ad: cn.most_common(1)[0][0] for ad, cn in names.items()}
    groups = collections.defaultdict(list)
    for ad, k in addr_key.items():
        if k is not None:
            groups[k].append(ad)
    canon = {ad: ad for ad in n_addr}
    merges = []
    for k, ads in groups.items():
        if len(ads) > 1:
            ads.sort(key=lambda ad: -n_addr[ad])
            for ad in ads:
                canon[ad] = ads[0]
            merges.append((k, [(ad, n_addr[ad]) for ad in ads]))
    for r in recs:
        r["user"] = canon[r["address"]]
    c["3a_name_groups_merged"] = len(merges)
    c["3a_addresses_merged_away"] = sum(len(m) - 1 for _, m in merges)

    # 3b. deduplicate on (user, normalized text); keep the earliest dated copy
    recs.sort(key=lambda r: (r["date"] is None, r["date"] or ""))
    seen, uniq = set(), []
    for r in recs:
        k = (r["user"], norm(r["text"]))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    recs = uniq
    c["3_after_dedup_sender_text"] = len(recs)
    by_text = collections.defaultdict(set)
    for r in recs:
        by_text[norm(r["text"])].add(r["user"])
    c["3_info_texts_shared_by_2plus_senders"] = sum(len(s) > 1 for s in by_text.values())   # not removed
    per_user = collections.Counter(r["user"] for r in recs)
    c["3_senders"] = len(per_user)

    # 4. per-user filter
    for t in (50, 100, a.min_docs, 200, 300):
        c[f"4_users_with_>={t}_docs"] = sum(v >= t for v in per_user.values())
    keep = {u for u, v in per_user.items() if v >= a.min_docs and v - a.heldout > 0}
    kept = [r for r in recs if r["user"] in keep]
    c["4_users_kept"] = len(keep)
    c["4_docs_kept"] = len(kept)
    c["4_docs_per_kept_user"] = pct([per_user[u] for u in keep])
    c["4_train_docs_per_kept_user_after_heldout"] = pct([per_user[u] - a.heldout for u in keep])
    c["4_non_enron_domain_users_kept"] = sorted(u for u in keep if not u.endswith("@enron.com"))
    dates = sorted(r["date"][:10] for r in kept if r["date"])
    c["4_date_range_kept"] = [dates[0], dates[-1]] if dates else None
    c["4_docs_dated_outside_1997_2002"] = sum(not ("1997" <= d[:4] <= "2002") for d in dates)
    c["4_docs_without_date"] = sum(r["date"] is None for r in kept)
    c["4_words_per_doc_kept"] = pct([len(r["text"].split()) for r in kept])
    c["3a_merges_touching_kept_users"] = sum(any(canon[ad] in keep for ad, _ in m) for _, m in merges)
    c["params"] = vars(a)
    c["seconds"] = round(time.time() - t0, 1)

    # outputs
    with gzip.open(os.path.join(OUT, "docs.jsonl.gz"), "wt", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({k: r[k] for k in ("user", "address", "date", "msg_id", "path", "text")}) + "\n")
    with open(os.path.join(OUT, "counts.json"), "w") as f:
        json.dump(c, f, indent=1)
    rng = random.Random(a.seed)
    if parsed_now:   # cleaning examples need a second tarball pass, so only when (re)parsing
        pick = {r["path"]: r for r in rng.sample([r for r in kept if r["_cut"]], 6)}
        with open(os.path.join(OUT, "samples.txt"), "w", encoding="utf-8") as f, \
                tarfile.open(RAW, "r:gz") as tar:
            for m in tar:
                if m.name in pick:
                    raw = body_of(email.message_from_bytes(tar.extractfile(m).read()))
                    f.write(f"=== {m.name}  user={pick[m.name]['user']}\n--- RAW BODY ---\n{raw[:2500]}\n"
                            f"--- KEPT ---\n{pick[m.name]['text']}\n\n")
    with open(os.path.join(OUT, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("# name key -> addresses (emails before dedup); first address = canonical user id\n")
        f.write("# '*' = the merged user is kept (>= min-docs)\n")
        for k, m in sorted(merges, key=lambda x: -sum(n for _, n in x[1])):
            star = "*" if canon[m[0][0]] in keep else " "
            f.write(f"{star} {k!r}: " + ", ".join(f"{ad} ({n})" for ad, n in m) + "\n")
    print(json.dumps(c, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
