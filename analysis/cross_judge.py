"""
cross_judge.py
==============
POST-HOC cross-model re-labeling of saved pipeline results.

Every pipeline labels with the GENERATOR judging its own answer (self-judge).
This script re-runs the IDENTICAL judge prompt with a DIFFERENT model over the
saved candidates and reports:

  * agreement + Cohen's kappa between original and cross labels
  * hallucination rate under both label sets
  * AUROC of the stored uncertainty scores under both label sets
    (label-sensitivity of the headline numbers)

No pipeline is re-run; inputs are the saved *_results.json files (C1-C4,
BB and WB, any dataset). What the original judge saw is reconstructed
per condition, exactly as in the pipelines:
  c1/c2: the full low-temperature candidate  (judge_candidate)
  c3:    extract_final_answer(judge_candidate)  (WB stores it as judge_extracted)
  c4:    the stored `candidate` (answer extracted from the reference chain)

Usage (from repo root, GPU node; pick the OTHER model as judge):
    python analysis/cross_judge.py --model mistral results/c*_morehopqa_8b_*_results.json
    python analysis/cross_judge.py --model 8b      results/c*_triviaqa_mistral_*_results.json

Output: results/crossjudge_<judgetag>__<original name>.json per input file
(existing outputs are skipped; --force to redo), plus a printed summary table.
"""

import os
import re
import sys
import json
import time
import argparse

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

MODEL_MAP = {
    "8b":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "70b":     "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
_ap = argparse.ArgumentParser()
_ap.add_argument("--model", required=True, help="judge model alias (8b/70b/mistral) or full HF id; use a DIFFERENT model than the generator")
_ap.add_argument("--limit", type=int, default=None, help="only first N records per file (smoke)")
_ap.add_argument("--force", action="store_true", help="redo files whose output already exists")
_ap.add_argument("files", nargs="+", help="*_results.json files to re-label")
_args = _ap.parse_args()
JUDGE_MODEL_NAME = str(MODEL_MAP.get(_args.model, _args.model))
JUDGE_TAG        = _args.model if _args.model in MODEL_MAP else "custom"

# qid prefix -> frozen file name (ids in nqopen.jsonl carry the build-time
# prefix "nq_open"; the frozen file itself is nqopen.jsonl)
DATASET_ALIAS = {"nq_open": "nqopen"}

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
def parse_records(path, details):
    """Yield (qid, judged_string, orig_halluc:int, scores:dict) per record."""
    cond = None
    m = re.search(r"(c[1-4])_", os.path.basename(path))
    if m:
        cond = m.group(1)
    if cond is None:
        raise ValueError(f"cannot infer condition (c1-c4) from filename: {path}")

    out = []
    for r in details:
        if r.get("error") is not None and "error" in r:
            if r.get("error"):
                continue
        if cond == "c4":
            qid = r["qid"]
            judged = r.get("candidate") or ""
            halluc = int(r["is_halluc"])
            scores = {f"agg_{k}": v for k, v in (r.get("agg") or {}).items()}
        else:
            qid = r["id"]
            cand = r.get("judge_candidate") or ""
            if cond == "c3":
                judged = r.get("judge_extracted") or extract_final_answer(cand)
            else:
                judged = cand
            halluc = int(not r["judge_correct"])
            scores = {}
            if "kle_full" in r: scores["kle_full"] = r["kle_full"]
            if "vne" in r:      scores["vne"] = r["vne"]
        out.append({"qid": qid, "judged": judged, "orig_halluc": halluc, "scores": scores})
    return out


def load_judge():
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return tok, model


def safe_auroc(labels, scores):
    try:
        if len(set(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


# =====================================================================
# Main
# =====================================================================
def main():
    frozen_cache = {}

    def gold_for(qid):
        prefix = qid.rsplit("-", 1)[0]
        name = DATASET_ALIAS.get(prefix, prefix)
        if name not in frozen_cache:
            frozen_cache[name] = {ex["id"]: ex for ex in load_frozen(name, frozen_dir=FROZEN_DIR)}
        ex = frozen_cache[name].get(qid)
        return (ex["question"], ex["gold_answers"]) if ex else (None, None)

    print(f"Loading judge: {JUDGE_MODEL_NAME}")
    tok, model = load_judge()

    summaries = []
    for path in _args.files:
        base = os.path.basename(path)
        out_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                f"crossjudge_{JUDGE_TAG}__{base}")
        if os.path.exists(out_path) and not _args.force:
            print(f"[skip] {base}: output exists ({out_path})")
            continue

        d = json.load(open(path, encoding="utf-8"))
        recs = parse_records(path, d.get("details") or [])
        if _args.limit:
            recs = recs[:_args.limit]
        print(f"\n== {base}: {len(recs)} records ==")

        t0 = time.perf_counter()
        rows = []
        for i, r in enumerate(recs):
            question, golds = gold_for(r["qid"])
            if question is None:
                print(f"  [skip {r['qid']}] not in frozen set")
                continue
            cross_correct = llm_judge(tok, model, question, golds, r["judged"])
            rows.append({**r, "cross_halluc": int(not cross_correct)})
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(recs)}]")

        orig  = [r["orig_halluc"]  for r in rows]
        cross = [r["cross_halluc"] for r in rows]
        n = len(rows)
        agree = sum(int(a == b) for a, b in zip(orig, cross)) / n if n else float("nan")
        try:
            kappa = float(cohen_kappa_score(orig, cross))
        except Exception:
            kappa = float("nan")

        score_keys = sorted({k for r in rows for k in r["scores"]})
        auroc = {}
        for k in score_keys:
            sub = [r for r in rows if k in r["scores"]]
            sc  = [r["scores"][k] for r in sub]
            auroc[k] = {
                "orig_labels":  safe_auroc([r["orig_halluc"] for r in sub], sc),
                "cross_labels": safe_auroc([r["cross_halluc"] for r in sub], sc),
            }

        summary = {
            "file":              base,
            "judge":             JUDGE_MODEL_NAME,
            "n":                 n,
            "agreement":         agree,
            "kappa":             kappa,
            "halluc_rate_orig":  (sum(orig) / n) if n else float("nan"),
            "halluc_rate_cross": (sum(cross) / n) if n else float("nan"),
            "auroc":             auroc,
            "t_seconds":         time.perf_counter() - t0,
        }
        disagreements = [
            {"qid": r["qid"], "judged": r["judged"][:200],
             "orig_halluc": r["orig_halluc"], "cross_halluc": r["cross_halluc"]}
            for r in rows if r["orig_halluc"] != r["cross_halluc"]
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary,
                       "disagreements": disagreements,
                       "records": rows}, f, indent=2, ensure_ascii=False)
        print(f"  agreement={agree:.3f}  kappa={kappa:.3f}  "
              f"halluc {summary['halluc_rate_orig']:.2f} -> {summary['halluc_rate_cross']:.2f}")
        print(f"  saved -> {out_path}")
        summaries.append(summary)

    if summaries:
        print("\n===== CROSS-JUDGE SUMMARY =====")
        print(f"{'file':46s} {'n':>4s} {'agree':>6s} {'kappa':>6s} {'hall_o':>7s} {'hall_x':>7s}")
        for s in summaries:
            print(f"{s['file']:46s} {s['n']:4d} {s['agreement']:6.3f} {s['kappa']:6.3f} "
                  f"{s['halluc_rate_orig']:7.2f} {s['halluc_rate_cross']:7.2f}")


if __name__ == "__main__":
    main()