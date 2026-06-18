"""
frozen_utils.py
================
Runtime helpers shared by the pipelines and the freeze builder.

Design goals
------------
* Zero dependency on the `datasets` library at *read* time. The frozen JSONL
  file IS the data; reading it back must not depend on any HF version. This is
  the whole point of freezing (we are escaping shuffle/version drift).
* A single normalized schema for every dataset, so pipelines stop carrying
  dataset-specific gold extraction like `ex["answer"]["aliases"]`.

Frozen schema (one JSON object per line):
    {
      "id":           "triviaqa-000042",   # stable, assigned at freeze time
      "dataset":      "triviaqa",
      "question":     "....",
      "gold_answers": ["...", "..."],       # always a non-empty list[str]
      "meta":         { ... provenance / dataset-specific extras ... }
    }
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Callable

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def normalize_question_key(q: str) -> str:
    """Key used ONLY for de-duplication. The stored question text is untouched."""
    return _WS.sub(" ", q.strip().lower())


def normalize_golds(golds) -> list[str]:
    """Coerce gold answers into a clean, non-empty, de-duplicated list[str].

    Accepts a str or any iterable of str. Order is preserved.
    """
    if isinstance(golds, str):
        golds = [golds]
    out, seen = [], set()
    for g in golds:
        if g is None:
            continue
        g = str(g).strip()
        if not g:
            continue
        if g.lower() in seen:
            continue
        seen.add(g.lower())
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# Selection (pure, version-independent)
# ---------------------------------------------------------------------------
def select_unique(records: list[dict], n: int, seed: int, dedupe: bool) -> list[dict]:
    """Shuffle `records` deterministically and select up to `n` items.

    `records` is a list of {"question", "gold_answers", "meta"} dicts in the
    dataset's native order. We shuffle the *indices* with Python's stdlib RNG
    (Mersenne Twister is stable across Python versions, unlike Dataset.shuffle),
    then walk the shuffled order.

    If dedupe=True we skip a record whose normalized question we have already
    taken, until we have `n` unique questions. If dedupe=False we take the first
    `n` in shuffled order (duplicates allowed).
    """
    import random

    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)

    selected, seen = [], set()
    for i in idx:
        rec = records[i]
        if not rec["question"] or not rec["gold_answers"]:
            continue  # unusable row (empty question or no gold)
        if dedupe:
            k = normalize_question_key(rec["question"])
            if k in seen:
                continue
            seen.add(k)
        selected.append(rec)
        if len(selected) >= n:
            break
    return selected


def assign_ids(records: list[dict], dataset: str) -> list[dict]:
    """Attach a stable id + dataset field based on final position."""
    out = []
    for pos, rec in enumerate(records):
        out.append({
            "id":           f"{dataset}-{pos:06d}",
            "dataset":      dataset,
            "question":     rec["question"],
            "gold_answers": rec["gold_answers"],
            "meta":         rec.get("meta", {}),
        })
    return out


# ---------------------------------------------------------------------------
# IO + integrity
# ---------------------------------------------------------------------------
def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Runtime loader (used by C1-C4 pipelines)
# ---------------------------------------------------------------------------
def load_frozen(name: str, frozen_dir: str = "data/frozen", verify: bool = True) -> list[dict]:
    """Load a frozen dataset as a list[dict].

    If verify=True and a manifest entry exists, the file's sha256 is checked so
    a silently-modified frozen file is caught immediately.
    """
    path = os.path.join(frozen_dir, f"{name}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Frozen dataset not found: {path}. "
            f"Run `python build_frozen_datasets.py --build {name}` first."
        )

    if verify:
        man_path = os.path.join(frozen_dir, "manifest.json")
        if os.path.exists(man_path):
            with open(man_path, encoding="utf-8") as f:
                man = json.load(f)
            entry = man.get(name)
            if entry and "sha256" in entry:
                actual = sha256_file(path)
                if actual != entry["sha256"]:
                    raise ValueError(
                        f"sha256 mismatch for {name}: file has been modified "
                        f"since freezing.\n  manifest: {entry['sha256']}\n  file    : {actual}"
                    )

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
