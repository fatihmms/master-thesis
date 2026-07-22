"""
SelfCheckGPT-NLI baseline (Manakul et al. 2023) on the C2 (holistic CoT)
setting — post-hoc, black-box, response-level.

No generation: reuses the C2 BLACK-BOX result files. Per question:
  main response = `judge_candidate` (low-temp C2 chain)
  samples       = the 10 saved stochastic `responses` (same ones KLE-BB used)
  per sample    : NLI(premise=sample, hypothesis=main);
                  P(contra) = softmax over {entail, contra} (neutral dropped,
                  faithful to Manakul)
  question score = mean_s P(contra)      -> higher = more likely hallucinated

Response-level (not sentence-level): our labels (`judge_correct`) are
question-level, so the score lives at the same granularity. Labels are reused
from the C2 BB file -> AUROC directly comparable to C2 black-box KLE.

NLI model = microsoft/deberta-v2-xlarge-mnli (same as the pipelines; fp16-safe
per project rules — never DeBERTa-v1 with fp16). Only DeBERTa is loaded; the
generator is never touched.

Baseline exception to the repo convention: `--dataset` IS parametrized
(approved) — read-only over saved results, zero per-dataset logic.

NOTE: delete the checkpoint when config (model, source file) changes.
HPC (Slurm) uyumlu.
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

# =====================================================================
# Config
# =====================================================================
MODEL_MAP = {  # generator tags — used only to locate the C2 BB source file
    "8b":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "70b":     "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
NLI_MODEL = "microsoft/deberta-v2-xlarge-mnli"

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="8b", help="generator tag of the C2 BB source (8b/mistral)")
_ap.add_argument("--dataset", required=True, choices=["triviaqa", "nqopen", "morehopqa"])
_ap.add_argument("--n", type=int, default=0, help="limit questions (0 = all); smoke tests")
_ap.add_argument("--nli-bs", type=int, default=32, help="NLI batch size (pairs)")
_args = _ap.parse_args()

MODEL_TAG = _args.model
DATASET   = _args.dataset
NLI_BS    = _args.nli_bs

OUT_DIR         = "./results/"
RESULT_FILE     = f"selfcheck_c2_{DATASET}_{MODEL_TAG}_results.json"
CHECKPOINT_FILE = os.path.join(OUT_DIR, f"selfcheck_c2_{DATASET}_{MODEL_TAG}_checkpoint.jsonl")

MAX_LENGTH = 512   # DeBERTa-v2 position limit; pair-truncated (longest_first)


# =====================================================================
# Input resolution (three naming schemes coexist in results/)
# =====================================================================
def resolve_c2_bb_file():
    cands = [
        os.path.join(OUT_DIR, f"c2_{DATASET}_{MODEL_TAG}_BB_results.json"),
        os.path.join(OUT_DIR, "black-box", DATASET, f"c2_{DATASET}_{MODEL_TAG}_BB_results.json"),
    ]
    if MODEL_TAG == "8b":  # legacy TriviaQA naming ("llama", lowercase bb)
        cands.append(os.path.join(OUT_DIR, "black-box", DATASET, f"c2_{DATASET}_llama_bb.json"))
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"No C2 BB results for {DATASET}/{MODEL_TAG}; tried: {cands}")


# =====================================================================
# Hugging Face Login (DeBERTa is not gated; harmless if absent)
# =====================================================================
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("No HF Token!")


# =====================================================================
# NLI model
# =====================================================================
def load_nli():
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    # label indices are read from the config — never hardcoded
    id2label = {int(k): v.upper() for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    if "CONTRADICTION" not in label2id or "ENTAILMENT" not in label2id:
        sys.exit(f"Unexpected NLI labels: {id2label}")
    return tok, model, label2id["ENTAILMENT"], label2id["CONTRADICTION"]


@torch.no_grad()
def p_contra(tok, model, ENT, CON, premises, hypotheses):
    """P(contradiction) among {entail, contra} for each (premise, hypothesis)."""
    out = []
    for b0 in range(0, len(premises), NLI_BS):
        enc = tok(premises[b0:b0 + NLI_BS], hypotheses[b0:b0 + NLI_BS],
                  return_tensors="pt", padding=True,
                  truncation="longest_first", max_length=MAX_LENGTH).to(model.device)
        logits = model(**enc).logits.float()
        two = logits[:, [ENT, CON]]                     # drop neutral (Manakul)
        out.extend(torch.softmax(two, dim=-1)[:, 1].cpu().tolist())
    return out


# =====================================================================
# Checkpoint (resumable, id-based; delete when config changes)
# =====================================================================
def load_checkpoint():
    done = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["id"]] = rec
        print(f"Resuming: {len(done)} questions from {CHECKPOINT_FILE}")
    return done


# =====================================================================
# Main
# =====================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    src_path = resolve_c2_bb_file()
    print(f"C2 BB source: {src_path}")
    with open(src_path, encoding="utf-8") as f:
        src = json.load(f)
    items = src["details"]
    if _args.n:
        items = items[:_args.n]

    print(f"Loading NLI: {NLI_MODEL} (fp16)")
    tok, model, ENT, CON = load_nli()

    done = load_checkpoint()
    ckpt = open(CHECKPOINT_FILE, "a", encoding="utf-8")

    details = []
    for idx, ex in enumerate(items):
        if ex["id"] in done:
            details.append(done[ex["id"]])
            continue
        t0 = time.perf_counter()
        main_resp = (ex["judge_candidate"] or "").strip()
        samples = [s for s in ex["responses"] if s and s.strip()]
        if not main_resp or not samples:
            rec = {"id": ex["id"], "question": ex["question"], "score": None,
                   "p_contra_per_sample": None,
                   "is_halluc": int(not ex["judge_correct"]),
                   "error": "empty_main_or_samples",
                   "cost": {"t_seconds": time.perf_counter() - t0, "n_pairs": 0}}
        else:
            pcs = p_contra(tok, model, ENT, CON, samples, [main_resp] * len(samples))
            rec = {"id": ex["id"], "question": ex["question"],
                   "score": float(np.mean(pcs)),
                   "p_contra_per_sample": [round(p, 6) for p in pcs],
                   "is_halluc": int(not ex["judge_correct"]),
                   "error": None,
                   "cost": {"t_seconds": time.perf_counter() - t0, "n_pairs": len(pcs)}}
        details.append(rec)
        ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ckpt.flush()
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1:4d}/{len(items)}]  score={rec['score']}")
    ckpt.close()

    # ---- AUROC (halluc = positive; higher mean P(contra) = more suspect) ----
    valid  = [r for r in details if r["score"] is not None]
    labels = [r["is_halluc"] for r in valid]
    scores = [r["score"] for r in valid]
    auroc  = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None
    rate   = sum(labels) / len(labels) if labels else float("nan")
    print(f"\nAUROC               : {auroc}")
    print(f"Hallucination rate  : {rate:.2%}   (n_valid={len(valid)}/{len(details)})")

    out = {
        "config": {
            "condition":   "selfcheckgpt_nli_c2_baseline",
            "nli":         NLI_MODEL,
            "precision":   "fp16",
            "dataset":     DATASET,
            "generator_tag": MODEL_TAG,
            "source_file": src_path,
            "n_questions": len(details),
            "granularity": "response_level",
            "reference":   "judge_candidate (low-temp C2 chain)",
            "score":       "mean over samples of P(contra) among {entail, contra}",
            "max_length":  MAX_LENGTH,
        },
        "auroc":              auroc,
        "hallucination_rate": rate,
        "details":            details,
    }
    path = os.path.join(OUT_DIR, RESULT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
