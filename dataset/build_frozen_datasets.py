"""
build_frozen_datasets.py
========================
One-shot builder that freezes each dataset into a normalized JSONL of
`N` (default 1000) questions, selected deterministically with a fixed seed.

Run on a machine WITH HuggingFace access (e.g. your cluster login node).

Usage
-----
    # 1) Inspect first (no write): see raw schema + what the adapter extracts
    python build_frozen_datasets.py --inspect triviaqa
    python build_frozen_datasets.py --inspect morehopqa

    # 2) Build one, or all
    python build_frozen_datasets.py --build triviaqa
    python build_frozen_datasets.py --build all

Output
------
    data/frozen/<name>.jsonl      # the frozen data (commit these to git)
    data/frozen/manifest.json     # provenance + sha256 per dataset

After freezing, pipelines load via:
    from frozen_utils import load_frozen
    ds = load_frozen("triviaqa")          # list[dict]
    question = ex["question"]; golds = ex["gold_answers"]; qid = ex["id"]
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import datetime

from dataset.frozen_utils import (
    normalize_golds, select_unique, assign_ids,
    write_jsonl, sha256_file,
)

SEED      = 42
N_TARGET  = 1000
FROZEN_DIR = "data/frozen"

# ===========================================================================
# Adapters
# ---------------------------------------------------------------------------
# Each adapter maps ONE raw HF example -> (question:str, gold_answers, meta:dict).
# `gold_answers` may be a str or list; frozen_utils.normalize_golds cleans it.
# If a row is unusable, return (None, [], {}) and it will be skipped.
# ===========================================================================
def adapt_triviaqa(ex: dict):
    ans = ex.get("answer", {}) or {}
    aliases = ans.get("aliases") or []
    value   = ans.get("value")
    golds   = list(aliases)
    if value:
        golds.append(value)
    meta = {"orig_value": value, "question_id": ex.get("question_id")}
    return ex.get("question"), golds, meta


def adapt_nq_open(ex: dict):
    # nq_open: question:str, answer:list[str]
    return ex.get("question"), ex.get("answer") or [], {}


def adapt_morehopqa(ex: dict):
    # alabnii/morehopqa (human-verified default): multi-hop question + last-hop answer.
    # Closed-book: `context` deliberately NOT stored.
    meta = {
        "previous_question":      ex.get("previous_question"),
        "previous_answer":        ex.get("previous_answer"),
        "answer_type":            ex.get("answer_type"),
        "previous_answer_type":   ex.get("previous_answer_type"),
        "no_of_hops":             ex.get("no_of_hops"),          # was wrongly "num_hop"
        "reasoning_type":         ex.get("reasoning_type"),      # Symbolic/Arithmetic/Commonsense
        "question_decomposition": ex.get("question_decomposition"),
    }
    return ex.get("question"), ex.get("answer"), meta


def adapt_2wiki(ex: dict):
    # framolfese/2WikiMultihopQA: HotpotQA layout, answer is a single string
    meta = {"type": ex.get("type"), "orig_id": ex.get("id")}
    return ex.get("question"), ex.get("answer"), meta


# ===========================================================================
# Registry
# ---------------------------------------------------------------------------
# split=None  -> auto-resolve (prefer validation > dev > test > train > only-split)
# dedupe=True -> select N *unique* questions (we re-run all conditions, so this
#                is uniform across every dataset including TriviaQA).
# ===========================================================================
REGISTRY = {
    "triviaqa": {
        "hf_id":   "mandarjoshi/trivia_qa",
        "config":  "rc.nocontext",
        "split":   "validation",
        "adapter": adapt_triviaqa,
        "dedupe":  True,
    },
    "nq_open": {
        "hf_id":   "google-research-datasets/nq_open",
        "config":  None,
        "split":   "validation",
        "adapter": adapt_nq_open,
        "dedupe":  True,
    },
    "morehopqa": {
        "hf_id":   "alabnii/morehopqa",
        "config":  None,
        "split":   None,        # auto-resolve; verify with --inspect
        "adapter": adapt_morehopqa,
        "dedupe":  True,
    },
    "2wikimultihopqa": {
        "hf_id":   "framolfese/2WikiMultihopQA",
        "config":  None,
        "split":   "validation",
        "adapter": adapt_2wiki,
        "dedupe":  True,
    },
}


# ===========================================================================
# HF loading (only part that needs the `datasets` library / network)
# ===========================================================================
def _resolve_split(dsdict_or_ds, requested):
    import datasets as hfds
    # Single Dataset already (split was given to load_dataset)
    if isinstance(dsdict_or_ds, hfds.Dataset):
        return dsdict_or_ds
    avail = list(dsdict_or_ds.keys())
    if requested and requested in avail:
        return dsdict_or_ds[requested]
    for pref in ("validation", "dev", "test", "train"):
        if pref in avail:
            print(f"  [split] requested={requested!r} not found; using {pref!r} "
                  f"(available: {avail})")
            return dsdict_or_ds[pref]
    # fall back to the only / first split
    print(f"  [split] using {avail[0]!r} (available: {avail})")
    return dsdict_or_ds[avail[0]]


def load_raw(name: str):
    import datasets as hfds
    spec = REGISTRY[name]
    kwargs = {}
    if spec["config"]:
        kwargs["name"] = spec["config"]
    if spec["split"]:
        kwargs["split"] = spec["split"]
    try:
        ds = hfds.load_dataset(spec["hf_id"], **kwargs)
    except Exception as e:
        # Some repos ship a loading script -> needs trust_remote_code
        print(f"  [load] retrying with trust_remote_code=True ({type(e).__name__})")
        ds = hfds.load_dataset(spec["hf_id"], trust_remote_code=True, **kwargs)
    ds = _resolve_split(ds, spec["split"])
    return ds


# ===========================================================================
# Inspect / Build
# ===========================================================================
def inspect(name: str, k: int = 3):
    spec = REGISTRY[name]
    print(f"\n=== INSPECT {name}  ({spec['hf_id']}, config={spec['config']}, "
          f"split={spec['split']}) ===")
    ds = load_raw(name)
    print(f"rows in split : {len(ds)}")
    print(f"raw columns   : {ds.column_names}")
    for i in range(min(k, len(ds))):
        ex = ds[i]
        q, golds, meta = spec["adapter"](ex)
        print(f"\n--- raw example {i} ---")
        print(json.dumps(ex, ensure_ascii=False, indent=2)[:1200])
        print(f"--- adapter output {i} ---")
        print(f"  question     : {q}")
        print(f"  gold_answers : {normalize_golds(golds)}")
        print(f"  meta         : {meta}")
    print("\nIf the adapter output looks correct, run with --build.")


def build(name: str, seed: int, n: int):
    import datasets as hfds
    spec = REGISTRY[name]
    print(f"\n=== BUILD {name} ===")
    ds = load_raw(name)
    source_size = len(ds)
    print(f"  source rows  : {source_size}")

    adapter = spec["adapter"]
    records = []
    for ex in ds:
        q, golds, meta = adapter(ex)
        golds = normalize_golds(golds)
        if not q or not golds:
            continue
        records.append({"question": q, "gold_answers": golds, "meta": meta})
    print(f"  usable rows  : {len(records)}")

    selected = select_unique(records, n=n, seed=seed, dedupe=spec["dedupe"])
    if len(selected) < n:
        print(f"  WARNING: only {len(selected)} unique rows available (< {n}).")
    rows = assign_ids(selected, name)

    path = os.path.join(FROZEN_DIR, f"{name}.jsonl")
    write_jsonl(rows, path)
    digest = sha256_file(path)

    n_unique = len({r["question"].strip().lower() for r in rows})
    print(f"  written      : {len(rows)} rows ({n_unique} unique) -> {path}")
    print(f"  sha256       : {digest}")

    # update manifest
    man_path = os.path.join(FROZEN_DIR, "manifest.json")
    man = {}
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            man = json.load(f)
    man[name] = {
        "hf_id":            spec["hf_id"],
        "config":           spec["config"],
        "split":            spec["split"],
        "seed":             seed,
        "n_target":         n,
        "n_written":        len(rows),
        "n_unique":         n_unique,
        "dedupe":           spec["dedupe"],
        "source_split_size": source_size,
        "datasets_version": hfds.__version__,
        "sha256":           digest,
        "built_at":         datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print(f"  manifest     : updated {man_path}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inspect", metavar="NAME", help="dry-run: show schema + adapter output")
    g.add_argument("--build",   metavar="NAME", help="freeze NAME, or 'all'")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n",    type=int, default=N_TARGET)
    args = ap.parse_args()

    if args.inspect:
        if args.inspect not in REGISTRY:
            sys.exit(f"Unknown dataset {args.inspect!r}. Known: {list(REGISTRY)}")
        inspect(args.inspect)
        return

    targets = list(REGISTRY) if args.build == "all" else [args.build]
    for name in targets:
        if name not in REGISTRY:
            sys.exit(f"Unknown dataset {name!r}. Known: {list(REGISTRY)}")
    for name in targets:
        build(name, seed=args.seed, n=args.n)


if __name__ == "__main__":
    main()
