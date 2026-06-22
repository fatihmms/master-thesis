"""
WHITE-BOX C3 on TriviaQA  (zero-shot CoT + FINAL-ANSWER hidden-state).

C3 is C2 with ONE difference: instead of pooling the FULL CoT chain, we extract
the final answer and pool layer-HIDDEN_LAYER hidden states over the FINAL-ANSWER
token span only. Counterpart of black-box C3 (which clusters the extracted answer
strings); here we read the hidden states of those same answer tokens.

POOL MUST be "mean": the final answer is the suffix of the generation, so its
LAST token equals the chain's last token -> with "last" pooling C3 would collapse
into C2 (identical vector). "mean" over the answer span keeps C3 distinct (it
averages only the answer tokens, vs C2's average over the whole chain).

Kernel + VNE are the same white-box machinery (layer 16, cosine Gram, VNE).
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

import kle.core                      # vn_entropy only

sys.path.insert(0, root_dir)
from dataset.frozen_utils import load_frozen
import time

# =====================================================================
# Config
# =====================================================================
DATASET    = "nqopen"
FROZEN_DIR = os.path.join(root_dir, "dataset")
SEED         = 42
N_QUESTIONS  = 1000
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

# Sampling (CoT prompt; identical to black-box C3 -> generation matched)
TEMPERATURE    = 1.0
TOP_P          = 0.9
TOP_K          = 50
MAX_NEW_TOKENS = 200

# Zero-shot CoT "magic phrase" (Kojima et al.)
COT_TRIGGER    = "Let's think step by step."

# Judge
JUDGE_TEMP     = 0.1

# --- white-box kernel knobs (mirror C4) ---
HIDDEN_LAYER = 16          # output of the 16th transformer block (ablation knob)
POOL         = "mean"      # MUST be "mean" for C3 (see module docstring); "last" collapses into C2
KERNEL       = "cosine"    # "cosine" (normalized inner-product) | "rbf"
RBF_GAMMA    = None        # None -> 1/hidden_dim
FORWARD_BS   = 8           # batch size for the hidden-state forward passes

OUT_DIR     = "./results/"
RESULT_FILE = f"c3_{DATASET}_{MODEL_TAG}_WB_results.json"

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
    tok.padding_side = "right"     # pooling assumes pads on the RIGHT

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
# Prompt (IDENTICAL to black-box C3 -> enforces the "Final Answer:" marker)
# =====================================================================
def build_prompt(tokenizer, question):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant. Reason step by step, but keep your reasoning concise (under 3 sentences). You MUST conclude your response with the exact phrase 'Final Answer:' followed by your actual answer."},
        {"role": "user", "content": f"{question}\n\n{COT_TRIGGER}"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# =====================================================================
# Final-answer extraction (VERBATIM from black-box C3)
# =====================================================================
def extract_final_answer(text):
    lower_text = text.lower()
    marker = "final answer:"
    if marker in lower_text:
        idx = lower_text.rfind(marker) + len(marker)
        return text[idx:].strip()
    else:
        # Fallback: model forgot the marker -> last full sentence (avoid crash)
        sentences = text.split('.')
        return sentences[-2].strip() if len(sentences) > 1 else text.strip()


# =====================================================================
# Sampling + hidden-state extraction over the FINAL-ANSWER span (mean)
#   - generate N CoT outputs,
#   - extract each final answer text,
#   - find the answer's TOKEN start (tokenize prompt + reasoning-before-answer),
#   - mean-pool layer-HIDDEN_LAYER hidden states over [answer_start : L].
# =====================================================================
@torch.no_grad()
def sample_answers_hidden(tokenizer, model, question):
    """Returns (texts, extracted, H, n_gen_tokens, n_fwd_tokens)."""
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
        num_return_sequences=N_SAMPLES,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    texts = [tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True).strip()
             for i in range(N_SAMPLES)]

    extracted = [extract_final_answer(t) for t in texts]

    # Character offset where each answer begins inside its own generation.
    # The answer is a suffix; rfind locates it. Fallback (-1) -> whole continuation.
    ans_char_start = []
    for t, a in zip(texts, extracted):
        ci = t.rfind(a) if a else -1
        ans_char_start.append(ci if ci >= 0 else 0)

    base_len   = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    full_texts = [prompt + t for t in texts]
    # Token index where each answer span starts = len(prompt + reasoning-before-answer).
    ans_tok_start = [
        len(tokenizer(prompt + t[:ci], add_special_tokens=False)["input_ids"])
        for t, ci in zip(texts, ans_char_start)
    ]

    n_fwd, vecs = 0, []
    for b0 in range(0, N_SAMPLES, FORWARD_BS):
        batch = full_texts[b0:b0 + FORWARD_BS]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        n_fwd += int(enc["attention_mask"].sum().item())
        fout = model(**enc, output_hidden_states=True)
        hs   = fout.hidden_states[HIDDEN_LAYER]        # (b, T, d)
        mask = enc["attention_mask"]                   # (b, T), right-padded
        for jj in range(hs.shape[0]):
            i = b0 + jj
            L = int(mask[jj].sum().item())             # true length
            start = ans_tok_start[i]
            start = max(base_len, min(start, L - 1))   # >=1 answer token, after prompt
            if POOL == "mean":
                v = hs[jj, start:L, :].mean(dim=0)     # mean over ANSWER tokens
            elif POOL == "last":
                # degenerate for C3 (== C2's last token); kept only for ablation
                v = hs[jj, L - 1, :]
            else:
                raise ValueError(f"Unknown POOL: {POOL!r}")
            vecs.append(v.float().cpu())
        del fout, hs

    n_gen = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts)
    return texts, extracted, torch.stack(vecs, dim=0), n_gen, n_fwd


# =====================================================================
# Kernel + VNE  (verbatim from C4: build_step_kernel / step_vne)
# =====================================================================
def build_kernel(H):
    """H: (n, d) hidden vectors -> (n, n) PSD kernel."""
    H = H.numpy().astype(np.float64)
    if KERNEL == "cosine":
        norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
        Hn = H / norms
        K = Hn @ Hn.T
    elif KERNEL == "rbf":
        sq = np.sum(H**2, axis=1, keepdims=True)
        d2 = sq + sq.T - 2.0 * (H @ H.T)
        d2 = np.maximum(d2, 0.0)
        gamma = RBF_GAMMA if RBF_GAMMA is not None else 1.0 / H.shape[1]
        K = np.exp(-gamma * d2)
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
# Judge (extracts the final answer first, like black-box C3)
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

        responses, extracted, H, n_gen, n_fwd = sample_answers_hidden(gen_tok, gen_model, question)
        score = vne(H)                          # white-box C3 uncertainty (answer-span)

        candidate           = low_temp_sample(gen_tok, gen_model, question)
        extracted_candidate = extract_final_answer(candidate)     # judge on the answer, like black-box C3
        is_correct          = llm_judge(gen_tok, gen_model, question, golds, extracted_candidate)
        is_halluc           = (not is_correct)

        vne_scores.append(score)
        labels.append(int(is_halluc))
        cost = {
            "t_seconds":       time.perf_counter() - t0,
            "n_gen_tokens":    int(n_gen),
            "n_fwd_tokens":    int(n_fwd),
            "n_prompt_tokens": int(len(gen_tok(build_prompt(gen_tok, question))["input_ids"])),
        }
        details.append({
            "id":                  ex["id"],
            "question":            question,
            "responses":           responses,
            "extracted_answers":   extracted,
            "vne":                 score,
            "judge_candidate":     candidate,
            "judge_extracted":     extracted_candidate,
            "judge_correct":       is_correct,
            "cost":                cost,
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1:3d}/{len(ds)}]  VNE={score:.3f}  correct={is_correct}")

    auroc = roc_auc_score(labels, vne_scores)
    rate  = sum(labels) / len(labels)
    print(f"\nAUROC               : {auroc:.4f}")
    print(f"Hallucination rate  : {rate:.2%}")

    out = {
        "config": {
            "condition":      "C3_cot_extraction_wb",
            "generator":      GEN_MODEL_NAME,
            "precision":      PRECISION,
            "n_questions":    N_QUESTIONS,
            "n_samples":      N_SAMPLES,
            "temperature":    TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "cot_trigger":    COT_TRIGGER,
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
