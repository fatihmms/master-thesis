"""
c4_error_localization.py
========================
POST-HOC error localization for C4 (stepwise) runs on MoreHopQA.

Reads a saved C4 checkpoint JSONL (black-box or white-box, unchanged
pipelines) + the frozen morehopqa meta, and asks, per question:

  1. Which gold hop (from `question_decomposition`, details flattened)
     does the reference chain get wrong FIRST?  -> first_error_hop
  2. Which model step does that correspond to?  -> first_error_step
  3. Does the C4 step-entropy profile peak at/near that step?

Hop correctness is decided by an LLM judge (same style as the pipelines'
llm_judge): for hop j we scan the chain's steps in order and ask whether
the step states or correctly uses the hop's expected sub-answer; the
answer line is scanned last. First hop with no supporting segment is the
first error.

This script NEVER touches pipeline outputs; it only reads them.
Needs a GPU (judge model). Typical cost: ~15-25 short judge calls per
question. Resumable via its own JSONL checkpoint keyed on qid.

Input is either a C4 checkpoint JSONL or a C4 results JSON; both carry the
same per-question fields. Prefer --results: after rejudge.py the results
JSON is the only source holding fixed-judge labels, while the checkpoint
still holds the stale self-judge ones.

Usage (from repo root):
    python analysis/c4_error_localization.py --model 8b \
        --results results/black-box/morehopqa/c4_morehopqa_8b_BB_results.json

    python analysis/c4_error_localization.py \
        --ckpt results/c4_morehopqa_8b_BB_checkpoint.jsonl --model 8b
"""

import os
import re
import sys
import json
import time
import argparse
import collections

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)
from dataset.frozen_utils import load_frozen

# =====================================================================
# Config
# =====================================================================
DATASET    = "morehopqa"
FROZEN_DIR = os.path.join(root_dir, "dataset")
SEED       = 42

MODEL_MAP = {
    "8b":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "70b":     "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="8b", help="judge model alias (8b/70b/mistral) or full HF id")
_ap.add_argument("--ckpt",    help="path to a c4_morehopqa_*_checkpoint.jsonl")
_ap.add_argument("--results", help="path to a c4_morehopqa_*_results.json (preferred: "
                                   "holds the fixed-judge labels after rejudge.py)")
_args = _ap.parse_args()
if bool(_args.ckpt) == bool(_args.results):
    _ap.error("give exactly one of --ckpt / --results")
JUDGE_MODEL_NAME = str(MODEL_MAP.get(_args.model, _args.model))
JUDGE_TAG        = _args.model if _args.model in MODEL_MAP else "custom"
SRC_IN           = _args.ckpt or _args.results

PRECISION = "bf16"

# --- MUST MIRROR the C4 pipelines (segmentation is re-derived from `chain`) ---
STEP_RE   = re.compile(r"(?im)^\s*step\s*\d+\s*:")
ANSWER_RE = re.compile(r"(?im)^\s*(final\s+answer|answer)\s*:")
MAX_STEPS = 5

_base    = re.sub(r"_(checkpoint\.jsonl|results\.json)$", "", os.path.basename(SRC_IN))
OUT_DIR  = os.path.dirname(os.path.abspath(SRC_IN))
LOC_CKPT = os.path.join(OUT_DIR, f"{_base}_loc{JUDGE_TAG}_checkpoint.jsonl")
LOC_OUT  = os.path.join(OUT_DIR, f"{_base}_loc{JUDGE_TAG}_results.json")

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("No HF Token!")


# =====================================================================
# Segmentation (mirrors c4_*_pipeline.segment_steps incl. fallback + cap)
# =====================================================================
def segment_steps(chain):
    ans_match = ANSWER_RE.search(chain)
    ans_start = ans_match.start() if ans_match else len(chain)
    step_labels = [m for m in STEP_RE.finditer(chain) if m.start() < ans_start]

    steps = []
    for k, m in enumerate(step_labels):
        body_start = m.end()
        body_end = step_labels[k + 1].start() if k + 1 < len(step_labels) else ans_start
        steps.append(chain[body_start:body_end].strip())

    if len(steps) == 0:
        steps = [chain[:ans_start].strip()]
    steps = steps[:MAX_STEPS]

    answer_text = chain[ans_start:].strip() if ans_match else ""
    return steps, answer_text


# =====================================================================
# Gold hops: flatten question_decomposition (details replace their parent)
# =====================================================================
def flatten_hops(meta):
    hops = []
    for sq in meta.get("question_decomposition") or []:
        details = sq.get("details") or []
        items = details if details else [sq]
        for it in items:
            q, a = it.get("question"), it.get("answer")
            if q is None or a is None:
                continue
            hops.append({"sub_q": str(q), "sub_a": str(a)})
    return hops


# =====================================================================
# Judge
# =====================================================================
def load_judge():
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return tok, model


@torch.no_grad()
def judge_hop_in_segment(tokenizer, model, sub_q, sub_a, segment_text):
    prompt = (
        "We are assessing one segment of a step-by-step reasoning chain.\n"
        f"The sub-question for this hop is: {sub_q}\n"
        f"The expected answer to this sub-question is: {sub_a}\n"
        f"The reasoning segment says: {segment_text}\n"
        "Does this segment state, or correctly use, the expected answer to "
        "the sub-question?\n"
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


def localize(tokenizer, model, steps, answer_text, hops):
    """Scan segments (steps, then answer line) per hop, in hop order.

    Returns (hop_resolved_step, first_error_hop, first_error_step, n_calls).
    hop_resolved_step[j] = index of the first segment supporting hop j
    (n_steps == the answer line), or None. A step may resolve several hops
    (models merge hops); order across hops is not assumed.
    """
    segments = list(steps) + ([answer_text] if answer_text else [])
    resolved, n_calls = [], 0
    for hop in hops:
        found = None
        for k, seg in enumerate(segments):
            if not seg:
                continue
            n_calls += 1
            if judge_hop_in_segment(tokenizer, model, hop["sub_q"], hop["sub_a"], seg):
                found = k
                break
        resolved.append(found)

    first_error_hop, first_error_step = None, None
    for j, r in enumerate(resolved):
        if r is None:
            first_error_hop = j
            prev = [x for x in resolved[:j] if x is not None]
            # error begins right after the last resolved step (0 if none)
            first_error_step = min((max(prev) + 1) if prev else 0, len(steps) - 1)
            break
    return resolved, first_error_hop, first_error_step, n_calls


# =====================================================================
# Input: checkpoint JSONL and results JSON carry the same record fields
# =====================================================================
def load_c4_records(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        src = "checkpoint"
    else:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        recs = d.get("details") or []
        judge = (d.get("config") or {}).get("judge_mode", "self")
        src = f"results (judge_mode={judge})"
    return [r for r in recs if r.get("error") is None and r.get("chain")], src


# =====================================================================
# Checkpoint IO (same pattern as the C4 pipelines, keyed on qid)
# =====================================================================
def load_loc_checkpoint(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["qid"]] = rec
    return done


def append_loc_checkpoint(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =====================================================================
# Main
# =====================================================================
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    frozen = {ex["id"]: ex for ex in load_frozen(DATASET, frozen_dir=FROZEN_DIR)}

    c4, src = load_c4_records(SRC_IN)
    print(f"C4 records to localize: {len(c4)}  (from {SRC_IN}, {src})")

    print(f"Loading judge: {JUDGE_MODEL_NAME} ({PRECISION})")
    tok, model = load_judge()

    done = load_loc_checkpoint(LOC_CKPT)
    if done:
        print(f"Resuming: {len(done)} questions already localized.")
        # Labels can change between runs (rejudge.py rewrites is_halluc in the
        # results files). The expensive part -- hop localization -- is
        # label-independent, so resumed records are kept, but their label is
        # refreshed from the CURRENT input before any summary uses it.
        labels_now = {r["qid"]: r.get("is_halluc") for r in c4}
        for qid, rec in done.items():
            if qid in labels_now:
                rec["is_halluc"] = labels_now[qid]

    for i, rec in enumerate(c4):
        qid = rec["qid"]
        if qid in done:
            continue
        t0 = time.perf_counter()

        ex = frozen.get(qid)
        if ex is None:
            print(f"  [skip {qid}] not in frozen set")
            continue
        hops = flatten_hops(ex["meta"])
        steps, answer_text = segment_steps(rec["chain"])

        # sanity: segmentation must reproduce the pipeline's step count
        seg_mismatch = (len(steps) != rec.get("n_steps"))

        resolved, err_hop, err_step, n_calls = localize(
            tok, model, steps, answer_text, hops
        )

        step_scores = rec.get("step_scores") or []
        argmax_step = int(np.argmax(step_scores)) if step_scores else None

        out = {
            "qid":               qid,
            "is_halluc":         rec.get("is_halluc"),
            "no_of_hops":        ex["meta"].get("no_of_hops"),
            "reasoning_type":    ex["meta"].get("reasoning_type"),
            "n_gold_hops":       len(hops),
            "n_steps":           len(steps),
            "seg_mismatch":      seg_mismatch,
            # the model never emitted the step scaffold: one segment for a
            # multi-hop question. Step-wise analysis is meaningless on these,
            # so they are reported as their own category, not localized.
            "format_failure":    len(steps) <= 1 and len(hops) > 1,
            "hop_resolved_step": resolved,
            "first_error_hop":   err_hop,
            "first_error_step":  err_step,
            "step_scores":       step_scores,
            "argmax_step":       argmax_step,
            "n_judge_calls":     n_calls,
            "t_seconds":         time.perf_counter() - t0,
        }
        append_loc_checkpoint(LOC_CKPT, out)
        done[qid] = out

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:4d}/{len(c4)}]  err_hop={err_hop}  "
                  f"err_step={err_step}  argmax={argmax_step}  "
                  f"halluc={out['is_halluc']}")

    # ---- Summary ---------------------------------------------------------
    # Format failures are excluded from the step-wise numbers and reported
    # separately; they carry no step structure to localize an error in.
    all_recs = list(done.values())
    fmt_fail = [r for r in all_recs if r.get("format_failure")]
    recs = [r for r in all_recs if not r.get("format_failure")]
    halluc = [r for r in recs if r["is_halluc"] == 1]
    located = [r for r in halluc if r["first_error_step"] is not None]
    clean = [r for r in recs if r["is_halluc"] == 0]
    clean_flagged = [r for r in clean if r["first_error_hop"] is not None]

    def agree(rs, tol):
        hits = [r for r in rs if r["argmax_step"] is not None
                and abs(r["argmax_step"] - r["first_error_step"]) <= tol]
        return len(hits) / len(rs) if rs else float("nan")

    def chance(rs, tol):
        """Agreement a uniformly random step pick would reach on this cohort.

        Without it the raw agreement is unreadable: chains are capped at 5
        steps, so top-1 alone is already 20% by luck and +/-1 around 50%.
        """
        ps = []
        for r in rs:
            n = len(r["step_scores"] or [])
            if n:
                e = r["first_error_step"]
                ps.append(sum(1 for k in range(n) if abs(k - e) <= tol) / n)
        return float(np.mean(ps)) if ps else float("nan")

    # The headline answer: WHERE do hallucinations start?  Distributions over
    # hallucinated chains ("none" = every gold hop was supported somewhere in
    # the chain, i.e. the final answer failed without a localizable hop error).
    hop_dist = collections.Counter(
        "none" if r["first_error_hop"] is None else r["first_error_hop"]
        for r in halluc)
    step_dist = collections.Counter(r["first_error_step"] + 1 for r in located)
    # 5-step cap => raw step indices confound "late error" with "long chain";
    # the normalized position (step / n_steps) is the cap-safe view.
    norm_pos = [(r["first_error_step"] + 1) / r["n_steps"]
                for r in located if r["n_steps"]]

    err_vs_other = []
    for r in located:
        ss = r["step_scores"]
        e = r["first_error_step"]
        if len(ss) > 1 and e < len(ss):
            others = [s for k, s in enumerate(ss) if k != e]
            err_vs_other.append(ss[e] - float(np.mean(others)))

    summary = {
        "source":                  SRC_IN,
        "judge":                   JUDGE_MODEL_NAME,
        "n_total":                 len(all_recs),
        "n_format_failure":        len(fmt_fail),
        "format_failure_halluc_rate": (sum(r["is_halluc"] for r in fmt_fail) / len(fmt_fail))
                                      if fmt_fail else float("nan"),
        "n":                       len(recs),
        "n_halluc":                len(halluc),
        "n_halluc_error_located":  len(located),
        "loc_rate_in_halluc":      (len(located) / len(halluc)) if halluc else float("nan"),
        "n_clean_flagged":         len(clean_flagged),   # judge found a broken hop despite correct final answer
        "first_error_hop_dist":    {str(k): v for k, v in sorted(hop_dist.items(), key=lambda kv: str(kv[0]))},
        "first_error_step_dist":   {str(k): v for k, v in sorted(step_dist.items())},
        "mean_norm_error_position": float(np.mean(norm_pos)) if norm_pos else float("nan"),
        "argmax_top1_agreement":   agree(located, 0),
        "argmax_top1_chance":      chance(located, 0),
        "argmax_pm1_agreement":    agree(located, 1),
        "argmax_pm1_chance":       chance(located, 1),
        "mean_entropy_gap_at_error": float(np.mean(err_vs_other)) if err_vs_other else float("nan"),
        "n_seg_mismatch":          sum(r["seg_mismatch"] for r in recs),
    }
    print("\n===== LOCALIZATION SUMMARY =====")
    for k, v in summary.items():
        print(f"  {k:26s}: {v}")

    with open(LOC_OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": all_recs}, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {LOC_OUT}")


if __name__ == "__main__":
    main()