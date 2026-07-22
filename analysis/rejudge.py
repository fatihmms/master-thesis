"""
rejudge.py
==========
Fixed-judge re-labeling of saved pipeline results, IN-PLACE.

Every pipeline labels with the GENERATOR judging its own answer (self-judge),
so the correctness label z(x) depends on which model produced the response.
This script re-labels all saved results with a single FIXED judge
(Llama-3.1-8B-Instruct, following Nikitin et al. 2024), making hallucination
rates and AUROC comparable across generator models.

Only the label changes. The uncertainty scores (kle_full / vne / agg) are
produced upstream of the judge and are left byte-identical; the judge output is
terminal in every pipeline (it flows into judge_correct / is_halluc and nothing
else), so post-hoc re-labeling is exactly equivalent to having run the
pipelines with a fixed judge from the start.

Because of that equivalence the result files are left looking like native
fixed-judge pipeline output: the labels, hallucination_rate and auroc are
updated, config records which judge produced them, and nothing else is added.
All provenance of the re-labeling -- per-file agreement, Cohen's kappa, the old
and new hallucination rate and AUROC, the disagreeing questions, and the full
original label set -- goes into ONE separate report:

    results/rejudge_report.json

That report is what the thesis cites for the label-robustness numbers, and it
is also what makes the operation reversible and --force safe.

Two modes, selected automatically from config.generator:

  full           generator != judge (Mistral runs) -> every record re-judged
  metadata_only  generator == judge (Llama-8B runs) -> labels are identical by
                 construction (greedy, max_new_tokens=8); no GPU work, only
                 config stamping. --verify N re-judges N random records to
                 demonstrate the agreement empirically.

Safety, because this overwrites the inputs:
  * Before touching a file, AUROC and hallucination_rate are recomputed from
    the OLD labels and compared against the stored values. A mismatch means the
    record adapter is wrong for that file, and the file is skipped untouched.
  * Writes are atomic (tmp file + os.replace), and the report is flushed after
    every file.
  * Files already carrying config.judge_mode == "fixed" are skipped unless
    --force; with --force the original labels are taken from the report, so a
    redo can never mistake already-rejudged labels for the originals.
  * Re-judging is resumable via a per-file checkpoint JSONL.

Usage (from repo root, GPU node):
    python analysis/rejudge.py --verify 50 \
        results/black-box/*/c[1-4]_*_results.json \
        results/white-box/*/c[1-4]_*_results.json

    python analysis/rejudge.py --dry-run results/*-box/*/c[1-4]_*_results.json
"""

import os
import re
import sys
import json
import time
import random
import argparse
import datetime

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)
from dataset.frozen_utils import load_frozen

FROZEN_DIR = os.path.join(root_dir, "dataset")
CKPT_DIR = os.path.join(root_dir, "results", "rejudge_checkpoints")
REPORT_PATH = os.path.join(root_dir, "results", "rejudge_report.json")

# The fixed judge. Same model for every generator, every condition, every dataset.
FIXED_JUDGE = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SEED = 42

# ids in the frozen nqopen file carry the build-time prefix "nq_open"
DATASET_ALIAS = {"nq_open": "nqopen"}

# recomputed metrics must reproduce the stored ones to this tolerance
METRIC_TOL = 1e-6

_ap = argparse.ArgumentParser()
_ap.add_argument("--verify", type=int, default=50,
                 help="on metadata_only files, re-judge this many random records to "
                      "confirm the labels really are identical (0 disables)")
_ap.add_argument("--limit", type=int, default=None,
                 help="only the first N records per file (smoke test)")
_ap.add_argument("--force", action="store_true",
                 help="redo files already marked judge_mode=fixed (needs the report)")
_ap.add_argument("--dry-run", action="store_true",
                 help="no judge, no writes: only run the metric self-check and report the plan")
_ap.add_argument("files", nargs="+", help="*_results.json files to re-label")
_args = _ap.parse_args()

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("No HF Token!")


# =====================================================================
# MUST MIRROR the pipelines
# =====================================================================
def extract_final_answer(text):
    lower_text = text.lower()
    marker = "final answer:"
    if marker in lower_text:
        idx = lower_text.rfind(marker) + len(marker)
        return text[idx:].strip()
    sentences = text.split('.')
    return sentences[-2].strip() if len(sentences) > 1 else text.strip()


@torch.no_grad()
def llm_judge(tokenizer, model, question, gold_answers, predicted_answer):
    correct = "; ".join(gold_answers)
    prompt = (
        "We are assessing the quality of answers to the following question:\n"
        f"{question}\n"
        f"The following are expected answers to this question: {correct}.\n"
        f"The proposed answer is: {predicted_answer}\n"
        "Within the context of the question, does the proposed answer mean "
        "the same as any of the expected answers?\n"
        "Respond only with yes or no.\nResponse:"
    )
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, do_sample=False, max_new_tokens=8,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()
    return gen.startswith("yes")


# =====================================================================
# Per-condition record adapters
# =====================================================================
def infer_cond(path):
    m = re.search(r"(c[1-4])_", os.path.basename(path))
    if m is None:
        raise ValueError(f"cannot infer condition (c1-c4) from filename: {path}")
    return m.group(1)


def infer_box(path):
    p = os.path.abspath(path)
    if os.sep + "black-box" + os.sep in p or "_BB_" in os.path.basename(p):
        return "black-box"
    if os.sep + "white-box" + os.sep in p or "_WB_" in os.path.basename(p):
        return "white-box"
    raise ValueError(f"cannot infer box type (black/white) from path: {path}")


def valid_records(cond, details):
    """C4 stores per-question errors; C1-C3 abort the run instead."""
    if cond == "c4":
        return [r for r in details if r.get("error") is None]
    return list(details)


def rec_id(cond, r):
    return r["qid"] if cond == "c4" else r["id"]


def judged_string(cond, r):
    """Reconstruct exactly what the pipeline's judge saw."""
    if cond == "c4":
        # answer already extracted from the reference chain
        return r.get("candidate") or ""
    cand = r.get("judge_candidate") or ""
    if cond == "c3":
        # C3 judges the extracted span; WB stores it, BB does not
        return r.get("judge_extracted") or extract_final_answer(cand)
    return cand


def cur_halluc(cond, r):
    """The label currently in the file."""
    if cond == "c4":
        return int(r["is_halluc"])
    return int(not r["judge_correct"])


def apply_label(cond, box, r, is_correct):
    """Write the new label. Nothing else is added: the file must stay
    indistinguishable from native fixed-judge pipeline output."""
    if cond == "c4":
        r["is_halluc"] = int(not is_correct)
        if box == "white-box":
            li = r.get("label_info")
            if isinstance(li, dict):
                li["judge_correct"] = bool(is_correct)
    else:
        r["judge_correct"] = bool(is_correct)


def agg_keys(valid):
    """C4 AUROC keys in first-seen order, matching the pipeline's dict order."""
    keys = []
    for r in valid:
        for k in (r.get("agg") or {}):
            if k not in keys:
                keys.append(k)
    return keys


def safe_auroc(labels, scores):
    if len(set(labels)) < 2:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def compute_metrics(cond, box, valid, labels):
    """Reproduce the pipeline's own AUROC layout: float for C1-C3, dict for C4."""
    if cond == "c4":
        return {k: safe_auroc(labels, [r["agg"][k] for r in valid])
                for k in agg_keys(valid)}
    score_key = "kle_full" if box == "black-box" else "vne"
    return safe_auroc(labels, [r[score_key] for r in valid])


def metrics_match(recomputed, stored):
    """Stored metrics may be a float (C1-C3) or a dict (C4)."""
    if isinstance(stored, dict) != isinstance(recomputed, dict):
        return False
    if isinstance(stored, dict):
        if set(stored) != set(recomputed):
            return False
        return all(_close(recomputed[k], stored[k]) for k in stored)
    return _close(recomputed, stored)


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= METRIC_TOL


# =====================================================================
# Judge (loaded lazily: metadata_only files need no GPU)
# =====================================================================
_judge_tok = None
_judge_model = None


def get_judge():
    global _judge_tok, _judge_model
    if _judge_model is None:
        print(f"Loading fixed judge: {FIXED_JUDGE}")
        tok = AutoTokenizer.from_pretrained(FIXED_JUDGE)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            FIXED_JUDGE, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        _judge_tok, _judge_model = tok, model
    return _judge_tok, _judge_model


# =====================================================================
# Gold answers
# =====================================================================
_frozen_cache = {}


def gold_for(qid):
    prefix = str(qid).rsplit("-", 1)[0]
    name = str(DATASET_ALIAS.get(prefix, prefix))
    if name not in _frozen_cache:
        _frozen_cache[name] = {ex["id"]: ex
                               for ex in load_frozen(name, frozen_dir=FROZEN_DIR)}
    ex = _frozen_cache[name].get(qid)
    return (ex["question"], ex["gold_answers"]) if ex else (None, None)


# =====================================================================
# Report: the single place where the re-labeling is documented
# =====================================================================
def load_report():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, encoding="utf-8") as f:
            rep = json.load(f)
        rep.setdefault("files", {})
        return rep
    return {"judge_model": FIXED_JUDGE, "files": {}}


def save_report(rep):
    rep["judge_model"] = FIXED_JUDGE
    rep["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    write_atomic(REPORT_PATH, rep)


def report_key(path):
    return os.path.relpath(os.path.abspath(path), root_dir)


# =====================================================================
# Checkpoint IO (per file, keyed on qid)
# =====================================================================
def ckpt_path(path):
    base = os.path.basename(path).replace(".json", "")
    return os.path.join(CKPT_DIR, f"{base}.jsonl")


def load_ckpt(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["qid"]] = bool(rec["correct"])
    return done


def append_ckpt(path, qid, correct):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"qid": qid, "correct": bool(correct)}) + "\n")


def write_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# =====================================================================
# Per-file processing
# =====================================================================
def process(path, report):
    base = os.path.basename(path)
    key = report_key(path)
    cond = infer_cond(path)
    box = infer_box(path)

    d = json.load(open(path, encoding="utf-8"))
    cfg = d.get("config") or {}

    already = cfg.get("judge_mode") == "fixed"
    if already and not _args.force:
        print(f"[skip] {base}: already judge_mode=fixed (use --force to redo)")
        return None

    details = d.get("details") or []
    valid = valid_records(cond, details)
    if _args.limit:
        valid = valid[:_args.limit]
    if not valid:
        print(f"[skip] {base}: no valid records")
        return None

    qids = [rec_id(cond, r) for r in valid]
    prev = report["files"].get(key) or {}

    # ---- recover the self-judge labels ------------------------------------
    # Normally they are simply the labels in the file. On a --force redo the
    # file already holds fixed-judge labels, so the originals must come from
    # the report; without it the comparison would be meaningless.
    if already:
        prior = prev.get("labels_selfjudge")
        if not prior:
            print(f"[ABORT] {base}: already rejudged but the report has no original "
                  f"labels for it. Restore the file from git before using --force.")
            return None
        missing = [q for q in qids if q not in prior]
        if missing:
            print(f"[ABORT] {base}: report is missing original labels for "
                  f"{len(missing)} records. File untouched.")
            return None
        old_labels = [int(prior[q]) for q in qids]
        ref_auroc = prev.get("auroc_selfjudge")
        ref_rate = prev.get("hallucination_rate_selfjudge")
    else:
        old_labels = [cur_halluc(cond, r) for r in valid]
        ref_auroc = d.get("auroc")
        ref_rate = d.get("hallucination_rate")

    # ---- self-check: the adapter must reproduce the stored metrics ---------
    # Only meaningful on a complete file; --limit changes the population.
    if not _args.limit:
        rec_auroc = compute_metrics(cond, box, valid, old_labels)
        rec_rate = sum(old_labels) / len(old_labels)
        if not metrics_match(rec_auroc, ref_auroc):
            print(f"[ABORT] {base}: AUROC self-check failed "
                  f"(recomputed {rec_auroc} vs expected {ref_auroc}). File untouched.")
            return None
        if not _close(rec_rate, ref_rate):
            print(f"[ABORT] {base}: hallucination_rate self-check failed "
                  f"({rec_rate} vs {ref_rate}). File untouched.")
            return None

    # ---- gold answers must resolve for EVERY record before anything runs ---
    # Resolving lazily inside the judging loop would mean a systematic id
    # mismatch (a broken dataset alias, say) silently leaves the labels
    # untouched while the file still gets stamped judge_mode=fixed. Failing up
    # front instead costs nothing and is checkable by --dry-run.
    golds_by_qid = {}
    missing_gold = []
    for q in qids:
        question, golds = gold_for(q)
        if question is None:
            missing_gold.append(q)
        else:
            golds_by_qid[q] = (question, golds)
    if missing_gold:
        print(f"[ABORT] {base}: {len(missing_gold)} of {len(qids)} records are not in "
              f"the frozen set (e.g. {missing_gold[:3]}). File untouched.")
        return None

    generator = cfg.get("generator")
    mode = "metadata_only" if generator == FIXED_JUDGE else "full"

    print(f"\n== {base} ==")
    print(f"   cond={cond} box={box} n={len(valid)} generator={generator}")
    print(f"   mode={mode}")

    if _args.dry_run:
        print("   [dry-run] metric self-check and gold lookup passed, nothing written")
        return None

    t0 = time.perf_counter()
    verify_n, verify_agree = 0, None

    if mode == "full":
        # ---- re-judge every record ----------------------------------------
        os.makedirs(CKPT_DIR, exist_ok=True)
        cpath = ckpt_path(path)
        done = load_ckpt(cpath)
        if done:
            print(f"   resuming: {len(done)} records already judged")

        tok, model = get_judge()
        new_labels = []
        for i, r in enumerate(valid):
            qid = qids[i]
            if qid in done:
                is_correct = done[qid]
            else:
                question, golds = golds_by_qid[qid]
                is_correct = llm_judge(tok, model, question, golds,
                                       judged_string(cond, r))
                append_ckpt(cpath, qid, is_correct)
                done[qid] = is_correct
            apply_label(cond, box, r, is_correct)
            new_labels.append(int(not is_correct))
            if (i + 1) % 100 == 0:
                print(f"   [{i+1}/{len(valid)}]")
    else:
        # ---- labels are identical by construction; nothing to write --------
        new_labels = list(old_labels)

        if _args.verify > 0:
            rng = random.Random(SEED)
            idxs = rng.sample(range(len(valid)), min(_args.verify, len(valid)))
            tok, model = get_judge()
            hits = 0
            for i in idxs:
                question, golds = golds_by_qid[qids[i]]
                is_correct = llm_judge(tok, model, question, golds,
                                       judged_string(cond, valid[i]))
                verify_n += 1
                hits += int(int(not is_correct) == old_labels[i])
            verify_agree = (hits / verify_n) if verify_n else None
            print(f"   verify: {hits}/{verify_n} agree with the stored self-judge labels")

    # ---- recompute metrics under the new labels ---------------------------
    new_auroc = compute_metrics(cond, box, valid, new_labels)
    new_rate = sum(new_labels) / len(new_labels)

    n = len(new_labels)
    agreement = sum(int(a == b) for a, b in zip(old_labels, new_labels)) / n
    try:
        kappa = float(cohen_kappa_score(old_labels, new_labels))
        if not np.isfinite(kappa):
            kappa = None      # sklearn returns nan for degenerate labelings
    except Exception:
        kappa = None
    if kappa is None and old_labels == new_labels:
        kappa = 1.0           # identical labelings, degenerate only because of one class

    # ---- the result file itself stays clean -------------------------------
    cfg["judge_model"] = FIXED_JUDGE
    cfg["judge_mode"] = "fixed"
    d["config"] = cfg
    d["auroc"] = new_auroc
    d["hallucination_rate"] = new_rate
    write_atomic(path, d)

    # ---- everything about the re-labeling goes to the report --------------
    entry = {
        "cond": cond,
        "box": box,
        "generator": generator,
        "judge_model": FIXED_JUDGE,
        "mode": mode,
        "n": n,
        "agreement": agreement,
        "kappa": kappa,
        "hallucination_rate_selfjudge": sum(old_labels) / n,
        "hallucination_rate_fixed": new_rate,
        "auroc_selfjudge": ref_auroc,
        "auroc_fixed": new_auroc,
        "n_verified": verify_n,
        "verify_agreement": verify_agree,
        "t_seconds": time.perf_counter() - t0,
        "disagreements": [
            {"qid": qids[i], "selfjudge_halluc": old_labels[i],
             "fixed_halluc": new_labels[i],
             "judged": judged_string(cond, valid[i])[:200]}
            for i in range(n) if old_labels[i] != new_labels[i]
        ],
        # full original label set: makes the operation reversible and --force safe
        "labels_selfjudge": {qids[i]: old_labels[i] for i in range(n)},
    }
    report["files"][key] = entry
    save_report(report)

    print(f"   agreement={agreement:.3f} "
          f"kappa={'n/a' if kappa is None else round(kappa, 3)}  "
          f"halluc {entry['hallucination_rate_selfjudge']:.3f} -> {new_rate:.3f}")
    print(f"   auroc {ref_auroc} -> {new_auroc}")
    print(f"   file rewritten in place, provenance -> {REPORT_PATH}")

    return {"file": base, **entry}


# =====================================================================
# Main
# =====================================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Fixed judge : {FIXED_JUDGE}")
    print(f"Files       : {len(_args.files)}")
    print(f"Report      : {REPORT_PATH}")
    if _args.dry_run:
        print("DRY RUN: no judge is loaded and no file is written.")

    report = load_report()

    summaries = []
    for path in _args.files:
        try:
            s = process(path, report)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(path)}: {type(e).__name__}: {e}. File untouched.")
            continue
        if s:
            summaries.append(s)

    if summaries:
        print("\n===== REJUDGE SUMMARY =====")
        print(f"{'file':46s} {'mode':14s} {'n':>5s} {'agree':>6s} {'kappa':>6s} "
              f"{'hall_old':>9s} {'hall_new':>9s}")
        for s in summaries:
            k = "n/a" if s["kappa"] is None else f"{s['kappa']:.3f}"
            print(f"{s['file']:46s} {s['mode']:14s} {s['n']:5d} {s['agreement']:6.3f} "
                  f"{k:>6s} {s['hallucination_rate_selfjudge']:9.3f} "
                  f"{s['hallucination_rate_fixed']:9.3f}")

        verified = [s for s in summaries if s["n_verified"]]
        if verified:
            print("\n--- metadata_only verification (expected: 1.000) ---")
            for s in verified:
                print(f"{s['file']:46s} {s['n_verified']:4d} records  "
                      f"agreement={s['verify_agreement']:.3f}")


if __name__ == "__main__":
    main()
