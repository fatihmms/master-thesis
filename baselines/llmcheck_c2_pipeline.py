"""
LLM-Check baseline on the C2 (holistic CoT) setting — post-hoc, white-box.

Scores the SAVED low-temperature reference chain (`judge_candidate`) of the
white-box C2 run with the official LLM-Check detectors (Sriramanan et al.,
NeurIPS 2024). No generation: one teacher-forced forward pass per question,
then the official scoring functions from the cloned repo
(LLM_Check_Hallucination_Detection/common_utils.py):

  - Hidden Score : get_svd_eval        (centered-covariance log-eigenvalues)
  - Attn Score   : get_attn_eig_prod   (log-diagonal over attention heads)
  - Perplexity   : perplexity
  - Logit Entropy: logit_entropy (top_k=50) + window_logit_entropy (w=1)

All scores are computed on the RESPONSE span only (use_toklens=True), i.e.
the full CoT chain — matching the white-box C2 granularity (holistic).
Labels (`judge_correct`) are reused from the C2 WB results file, so AUROC is
directly comparable to the C2 white-box VNE.

Baseline exception to the repo convention: `--dataset` IS parametrized here
(approved), because this script is read-only over saved results and shares
zero per-dataset logic beyond the input path + system prompt.

NOTE: delete the checkpoint when config (layer, model, source file) changes.
HPC (Slurm) uyumlu.
"""

import os
import sys
import json
import time
import argparse

import torch

from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
_LLMCHECK_DIR = os.path.join(root_dir, "LLM_Check_Hallucination_Detection")
if not os.path.isdir(_LLMCHECK_DIR):
    sys.exit(
        "LLM-Check repo not found. Clone it into the repo root (gitignored, "
        "like kernel-language-entropy):\n  git clone "
        "https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection.git"
    )
sys.path.insert(0, _LLMCHECK_DIR)

# Official LLM-Check scoring functions — never re-implement the math.
from common_utils import (
    get_model_vals,
    get_svd_eval,
    get_attn_eig_prod,
    perplexity,
    logit_entropy,
    window_logit_entropy,
)

# =====================================================================
# Config
# =====================================================================
MODEL_MAP = {
    "8b":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "70b":     "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="8b", help="alias (8b/70b/mistral) veya tam HF id")
_ap.add_argument("--dataset", required=True, choices=["triviaqa", "nqopen", "morehopqa"])
_ap.add_argument("--n", type=int, default=0, help="limit questions (0 = all); smoke tests")
_ap.add_argument("--layer", type=int, default=16, help="layer for hidden/attn scores (WB KLE default)")
_args = _ap.parse_args()

GEN_MODEL_NAME = str(MODEL_MAP.get(_args.model, _args.model))
MODEL_TAG      = _args.model if _args.model in MODEL_MAP else "custom"
DATASET        = _args.dataset
LAYER          = _args.layer
PRECISION      = "bf16"

# Zero-shot CoT prompt — must match the C2 pipelines EXACTLY (teacher forcing).
COT_TRIGGER = "Let's think step by step."
SYSTEM_MSG = {
    # triviaqa / nqopen C2: "under 3 sentences"; morehopqa C2: "under 6 sentences"
    "triviaqa":  "You are a helpful assistant. Reason step by step, but keep your reasoning concise (under 3 sentences). Always conclude with your final answer.",
    "nqopen":    "You are a helpful assistant. Reason step by step, but keep your reasoning concise (under 3 sentences). Always conclude with your final answer.",
    "morehopqa": "You are a helpful assistant. Reason step by step, but keep your reasoning concise (under 6 sentences). Always conclude with your final answer.",
}[DATASET]

OUT_DIR         = "./results/"
RESULT_FILE     = f"llmcheck_c2_{DATASET}_{MODEL_TAG}_results.json"
CHECKPOINT_FILE = os.path.join(OUT_DIR, f"llmcheck_c2_{DATASET}_{MODEL_TAG}_checkpoint.jsonl")

# Score keys; polarity ("higher = hallucination"?) is dataset/model dependent
# for the eigen scores, so raw AUROC is reported per detector (0.5-symmetric).
DETECTORS = ["hidden", "attn", "ppl", "logit_entropy", "window_entropy"]


# =====================================================================
# Input resolution (three naming schemes coexist in results/)
# =====================================================================
def resolve_c2_wb_file():
    cands = [
        os.path.join(OUT_DIR, f"c2_{DATASET}_{MODEL_TAG}_WB_results.json"),
        os.path.join(OUT_DIR, "white-box", DATASET, f"c2_{DATASET}_{MODEL_TAG}_WB_results.json"),
    ]
    if MODEL_TAG == "8b":  # legacy TriviaQA naming ("llama", lowercase wb)
        cands.append(os.path.join(OUT_DIR, "white-box", DATASET, f"c2_{DATASET}_llama_wb.json"))
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"No C2 WB results for {DATASET}/{MODEL_TAG}; tried: {cands}")


# =====================================================================
# Hugging Face Login
# =====================================================================
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("No HF Token!")


# =====================================================================
# Model
# =====================================================================
def load_generator():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # eager attention: output_attentions=True is unsupported by sdpa/flash
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return tok, model


# =====================================================================
# Prompt (IDENTICAL to the dataset's C2 pipelines)
# =====================================================================
def build_prompt(tokenizer, question):
    msgs = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"{question}\n\n{COT_TRIGGER}"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# =====================================================================
# Per-question LLM-Check scores (official functions, response span only)
# =====================================================================
@torch.no_grad()
def llmcheck_scores(tokenizer, model, question, chain):
    """One teacher-forced forward pass -> the 5 detector scores on the chain span."""
    prompt   = build_prompt(tokenizer, question)
    base_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    # Same convention as the C2 WB extraction: the prompt string already
    # carries the chat-template special tokens as text.
    tok_in = tokenizer(prompt + chain, return_tensors="pt",
                       add_special_tokens=False)["input_ids"]
    total_len = tok_in.shape[1]
    if total_len - base_len < 2:
        return None, total_len  # degenerate span: covariance/PPL undefined

    logits, hidden, attns = get_model_vals(model, tok_in.to(model.device))
    # Unpack to CPU float32, batch squeezed — exactly as the official callers do.
    logit  = logits[0].float().cpu()
    hidden = [x[0].float().cpu() for x in hidden]
    attn   = [x[0].float().cpu() for x in attns]
    del logits, attns

    tok_len = [base_len, total_len]
    scores = {
        "hidden":         float(get_svd_eval([hidden], LAYER, [tok_len])[0]),
        "attn":           float(get_attn_eig_prod([attn], LAYER, [tok_len])[0]),
        "ppl":            float(perplexity([logit], [tok_in.cpu()], [tok_len])[0]),
        "logit_entropy":  float(logit_entropy([logit], [tok_len], top_k=50)[0]),
        "window_entropy": float(window_logit_entropy([logit], [tok_len], w=1)[0]),
    }
    return scores, total_len


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

    src_path = resolve_c2_wb_file()
    print(f"C2 WB source: {src_path}")
    with open(src_path, encoding="utf-8") as f:
        src = json.load(f)
    items = src["details"]
    if _args.n:
        items = items[:_args.n]

    print(f"Loading generator: {GEN_MODEL_NAME}  ({PRECISION}, eager attn)")
    tok, model = load_generator()

    done = load_checkpoint()
    ckpt = open(CHECKPOINT_FILE, "a", encoding="utf-8")

    details = []
    for idx, ex in enumerate(items):
        if ex["id"] in done:
            details.append(done[ex["id"]])
            continue
        t0 = time.perf_counter()
        chain = ex["judge_candidate"] or ""
        scores, n_fwd = llmcheck_scores(tok, model, ex["question"], chain)
        rec = {
            "id":            ex["id"],
            "question":      ex["question"],
            "scores":        scores,           # None if degenerate span
            "is_halluc":     int(not ex["judge_correct"]),
            "error":         None if scores else "degenerate_span",
            "cost":          {"t_seconds": time.perf_counter() - t0,
                              "n_fwd_tokens": int(n_fwd)},
        }
        details.append(rec)
        ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ckpt.flush()
        if (idx + 1) % 10 == 0:
            s = scores or {}
            print(f"  [{idx+1:4d}/{len(items)}]  hidden={s.get('hidden', float('nan')):.3f}"
                  f"  ppl={s.get('ppl', float('nan')):.2f}")
    ckpt.close()

    # ---- AUROC per detector (halluc = positive; raw polarity, no flipping) ----
    valid  = [r for r in details if r["scores"] is not None]
    labels = [r["is_halluc"] for r in valid]
    aurocs = {}
    for det in DETECTORS:
        vals = [r["scores"][det] for r in valid]
        aurocs[det] = float(roc_auc_score(labels, vals)) if len(set(labels)) > 1 else None
        print(f"AUROC {det:15s}: {aurocs[det]}")
    rate = sum(labels) / len(labels) if labels else float("nan")
    print(f"Hallucination rate  : {rate:.2%}   (n_valid={len(valid)}/{len(details)})")

    out = {
        "config": {
            "condition":   "llmcheck_c2_baseline",
            "generator":   GEN_MODEL_NAME,
            "precision":   PRECISION,
            "dataset":     DATASET,
            "source_file": src_path,
            "n_questions": len(details),
            "layer":       LAYER,
            "detectors":   DETECTORS,
            "span":        "response_only (use_toklens)",
            "reference":   "judge_candidate (low-temp C2 chain)",
        },
        "auroc":              aurocs,
        "hallucination_rate": rate,
        "details":            details,
    }
    path = os.path.join(OUT_DIR, RESULT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
