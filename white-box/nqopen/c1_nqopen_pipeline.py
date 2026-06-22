"""
WHITE-BOX C1 on NQ-Open.

Standard prompt (no CoT) -> sample N answers -> read layer-HIDDEN_LAYER hidden
states of each answer -> cosine Gram matrix -> von Neumann entropy (VNE) as the
uncertainty score. This is the hidden-state counterpart of the black-box C1:
SAME prompt, SAME generation, SAME judge; only the kernel/score differs
(hidden states instead of NLI entailment + log-likelihood mixture).

Kernel + VNE are taken verbatim from the white-box C4 pipeline (build_step_kernel
/ step_vne), just renamed without "step" because C1 has no step segmentation.
HPC (Slurm) uyumlu.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from huggingface_hub import login

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, os.path.join(root_dir, "kernel-language-entropy"))

import kle.core                      # vn_entropy only (no heat kernel here)

sys.path.insert(0, root_dir)
from dataset.frozen_utils import load_frozen
import time

# =====================================================================
# Config
# =====================================================================
DATASET    = "nqopen"
FROZEN_DIR = os.path.join(root_dir, "dataset")
SEED         = 42
N_QUESTIONS  = 20
N_SAMPLES    = 10

MODEL_MAP = {
    "8b":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "70b":     "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="8b", help="alias (8b/70b/mistral) veya tam HF id")
_args, _ = _ap.parse_known_args()
GEN_MODEL_NAME = str(MODEL_MAP.get(_args.model, _args.model))
MODEL_TAG      = _args.model if _args.model in MODEL_MAP else "custom"

PRECISION = "bf16"

# Sampling (identical to black-box C1 -> generation is matched)
TEMPERATURE    = 1.0
TOP_P          = 0.9
TOP_K          = 50
MAX_NEW_TOKENS = 64

# Judge (identical to black-box C1)
JUDGE_TEMP     = 0.1

# --- white-box kernel knobs (mirror C4) ---
HIDDEN_LAYER = 16          # output of the 16th transformer block (ablation knob)
POOL         = "mean"      # "last" = last answer token | "mean" = mean over answer tokens
KERNEL       = "cosine"    # "cosine" (normalized inner-product) | "rbf"
RBF_GAMMA    = None        # None -> 1/hidden_dim
FORWARD_BS   = 8           # batch size for the hidden-state forward passes

OUT_DIR     = "./results/"
# "_wb" tag: black-box C1 writes c1_<ds>_<model>_results.json into the SAME
# results/ folder; without this tag the white-box run would overwrite it.
RESULT_FILE = f"c1_{DATASET}_{MODEL_TAG}_WB_results.json"

# =====================================================================
# Hugging Face Login
# =====================================================================
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("No HF Token!")


# =====================================================================
# Reproducibility
# =====================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Model
# =====================================================================
def load_generator():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # right padding: pooling reads the last real token at index (L-1) and
    # mean-pools [prefix:L]; both assume pads are on the RIGHT.
    tok.padding_side = "right"

    if PRECISION == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME, quantization_config=bnb, device_map="auto",
        )
    elif PRECISION == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        )
    else:
        raise ValueError(f"Unknown PRECISION: {PRECISION!r}")

    model.eval()
    return tok, model


# =====================================================================
# Prompt (IDENTICAL to black-box C1 -> same no-CoT generation)
# =====================================================================
def build_prompt(tokenizer, question):
    msgs = [
        {"role": "system", "content": "Answer the following question in a single brief but complete sentence."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# =====================================================================
# Sampling + hidden-state extraction (mirrors C4.sample_step_hidden, but the
# "span" is the full generated answer; no steps, no delimiter truncation)
# =====================================================================
@torch.no_grad()
def sample_answers_hidden(tokenizer, model, question):
    """Returns (texts, H, n_gen_tokens, n_fwd_tokens), H of shape (N, hidden_dim)."""
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
        num_return_sequences=N_SAMPLES,        # batched (faster than the C1 loop)
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=1,                      # never empty -> pooling always has >=1 answer token
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    texts = [tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True).strip()
             for i in range(N_SAMPLES)]

    # Re-tokenize (prompt + answer) without special tokens; the prompt string
    # already carries the chat-template special tokens as text (same as C4).
    base_len   = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    full_texts = [prompt + t for t in texts]

    n_fwd, vecs = 0, []
    for b0 in range(0, N_SAMPLES, FORWARD_BS):
        batch = full_texts[b0:b0 + FORWARD_BS]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        n_fwd += int(enc["attention_mask"].sum().item())
        fout = model(**enc, output_hidden_states=True)
        hs   = fout.hidden_states[HIDDEN_LAYER]        # (b, T, d)
        mask = enc["attention_mask"]                   # (b, T), right-padded
        for j in range(hs.shape[0]):
            L = int(mask[j].sum().item())              # true (unpadded) length
            if POOL == "last":
                v = hs[j, L - 1, :]                    # last answer token
            elif POOL == "mean":
                start = min(base_len, L - 1)
                v = hs[j, start:L, :].mean(dim=0)      # mean over answer tokens
            else:
                raise ValueError(f"Unknown POOL: {POOL!r}")
            vecs.append(v.float().cpu())
        del fout, hs

    n_gen = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts)
    return texts, torch.stack(vecs, dim=0), n_gen, n_fwd


# =====================================================================
# Kernel + VNE  (verbatim from C4: build_step_kernel / step_vne)
# =====================================================================
def build_kernel(H):
    """H: (n, d) hidden vectors -> (n, n) PSD kernel."""
    H = H.numpy().astype(np.float64)
    if KERNEL == "cosine":
        norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
        Hn = H / norms
        K = Hn @ Hn.T                      # Gram matrix -> PSD, cosine sim.
    elif KERNEL == "rbf":
        sq = np.sum(H**2, axis=1, keepdims=True)
        d2 = sq + sq.T - 2.0 * (H @ H.T)
        d2 = np.maximum(d2, 0.0)
        gamma = RBF_GAMMA if RBF_GAMMA is not None else 1.0 / H.shape[1]
        K = np.exp(-gamma * d2)            # PSD
    else:
        raise ValueError(f"Unknown KERNEL: {KERNEL!r}")
    return K


def vne(H):
    """Uncertainty = vn_entropy(K). Normalization follows the repo (normalize=True)."""
    K = build_kernel(H)
    for jitter in [0, 1e-16, 1e-12, 1e-10]:
        try:
            return float(kle.core.vn_entropy(K, normalize=True, scale=False, jitter=jitter))
        except Exception:
            continue
    raise ValueError("VNE did not converge for any jitter")


# =====================================================================
# Judge (IDENTICAL to black-box C1)
# =====================================================================
@torch.no_grad()
def low_temp_sample(tokenizer, model, question):
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, do_sample=True, temperature=JUDGE_TEMP,
        top_p=1.0, top_k=0,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


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
# Main
# =====================================================================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading frozen dataset: {DATASET}")
    ds = load_frozen(DATASET, frozen_dir=FROZEN_DIR)
    if N_QUESTIONS:
        ds = ds[:N_QUESTIONS]

    print(f"Loading generator: {GEN_MODEL_NAME}  ({PRECISION})")
    gen_tok, gen_model = load_generator()

    vne_scores, labels, details = [], [], []

    for idx, ex in enumerate(ds):
        t0 = time.perf_counter()
        question = ex["question"]
        golds    = ex["gold_answers"]

        responses, H, n_gen, n_fwd = sample_answers_hidden(gen_tok, gen_model, question)
        score = vne(H)                         # white-box C1 uncertainty

        candidate  = low_temp_sample(gen_tok, gen_model, question)
        is_correct = llm_judge(gen_tok, gen_model, question, golds, candidate)
        is_halluc  = (not is_correct)

        vne_scores.append(score)
        labels.append(int(is_halluc))
        cost = {
            "t_seconds":       time.perf_counter() - t0,
            "n_gen_tokens":    int(n_gen),
            "n_fwd_tokens":    int(n_fwd),     # extra: hidden-state forward passes
            "n_prompt_tokens": int(len(gen_tok(build_prompt(gen_tok, question))["input_ids"])),
        }
        details.append({
            "id":              ex["id"],
            "question":        question,
            "responses":       responses,
            "vne":             score,
            "judge_candidate": candidate,
            "judge_correct":   is_correct,
            "cost":            cost,
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1:3d}/{len(ds)}]  VNE={score:.3f}  correct={is_correct}")

    auroc = roc_auc_score(labels, vne_scores)
    rate  = sum(labels) / len(labels)
    print(f"\nAUROC               : {auroc:.4f}")
    print(f"Hallucination rate  : {rate:.2%}")

    out = {
        "config": {
            "generator":      GEN_MODEL_NAME,
            "precision":      PRECISION,
            "n_questions":    N_QUESTIONS,
            "n_samples":      N_SAMPLES,
            "temperature":    TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "hidden_layer":   HIDDEN_LAYER,
            "pool":           POOL,
            "kernel":         KERNEL,
            "judge_temp":     JUDGE_TEMP,
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
