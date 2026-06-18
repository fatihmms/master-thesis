"""
Repo-faithful WHITE-BOX STEP-WISE KLE pipeline on TriviaQA.
CONDITION C3: Zero-shot CoT (+ step-format instruction) + step-wise KLE.
HPC (Slurm) compatible version.

SCHEME A (prefix-conditioned re-sampling):
  - Generate one low-temperature reference CoT chain; segment it into steps
    using explicit "Step k:" delimiters.
  - For each step i: prefix = prompt + chain up to and INCLUDING the
    "Step i:" label. Sample N_SAMPLES continuations from this prefix,
    truncate each continuation in TEXT space BEFORE the next delimiter,
    and extract the hidden state of the last CONTENT token (or mean-pool
    over the continuation tokens) at layer HIDDEN_LAYER -> n vectors.
  - Build an n x n PSD kernel K_i from these vectors ->
    vn_entropy(K_i) = uncertainty of step i.
  - Aggregate step scores to a response-level score
    (max / mean / attention-weighted mean).

DIFFERENCES FROM C1/C2:
  - Uncertainty signal comes from the model's hidden states,
    NOT from NLI text similarity.
  - DeBERTa / get_semantic_ids / get_entailment_graph are DISABLED.
  - Instead of ONE kernel per question, there are as many kernels (and VNE
    values) as reasoning steps, followed by an aggregation stage.

PROMPTING NOTE (state explicitly in the thesis):
  - C2 uses pure zero-shot CoT ("Let's think step by step.", Kojima et al.).
  - C3 uses the same trigger PLUS a step-format instruction ("Step 1:", ...,
    "Answer:"). The formatting instruction is required to operationalize
    deterministic step segmentation for Scheme A; it does not alter the
    reasoning paradigm.

LABELING (consistency with C1/C2):
  - PRIMARY: LLM-as-Judge (LABEL_MODE="judge"), identical judge prompt to the
    C1/C2 pipelines (paper Sec. 5). The judged candidate is the final answer
    extracted from the low-temperature reference chain, analogous to the
    low-temperature sample judged in C1/C2.
  - Token-F1 (LABEL_MODE="f1") is kept as an optional robustness check; it can
    also be computed post-hoc from the saved details without re-running.
  - AUROC values are only comparable across conditions if the SAME labeling
    function is used everywhere.

REPO FIDELITY NOTE:
  - vn_entropy comes from the official repo (kle.core); normalize=True is
    repo behavior.
  - The hidden-state -> kernel construction does NOT exist in the repo; it is
    our own contribution (normalized inner-product / RBF). This must be
    stated explicitly in the thesis.

FIXES RELATIVE TO THE FIRST DRAFT:
  1. Hidden states are no longer taken from the delimiter token: with the old
     StoppingCriteria approach, the last generated token was part of the NEXT
     delimiter ("Step 2:" / "Answer:"), so all samples ended on nearly
     identical tokens, the kernel collapsed to a single cluster, and the VNE
     signal vanished. Continuations are now truncated in text space before
     the delimiter and hidden states are read from content tokens.
  2. add_special_tokens=False everywhere after apply_chat_template
     (double-BOS bug fixed).
  3. Batched sampling per step (num_return_sequences) + a separate
     small-batch forward pass for hidden-state extraction (speed +
     correctness). tokenizer.padding_side is forced to "right" so that
     attention-mask-based indexing of the last real token is valid.
  4. Per-question try/except + resumable JSONL checkpointing.
     IMPORTANT: delete the checkpoint file whenever the config changes,
     otherwise records produced under a different config are silently reused.
"""

import os
import sys
import re
import json
import string
from collections import Counter, defaultdict

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

import kle.core
import kle.kernels

sys.path.insert(0, root_dir)
from dataset.frozen_utils import load_frozen
import time
# =====================================================================
# Config
# =====================================================================

DATASET    = "triviaqa"    
FROZEN_DIR = os.path.join(root_dir, "dataset")
SEED        = 42
N_QUESTIONS = 1000          # C3 is ~(n_steps)x more expensive than C1/C2:
                          # validate on a small subset first, then scale up.
N_SAMPLES   = 10          # samples per step (n) -> K_i is (n x n)

GEN_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PRECISION      = "bf16"

# Sampling for continuations (uncertainty arises from this distribution;
# values match the KLE / semantic-uncertainty repo defaults, paper Sec. 5)
TEMPERATURE = 1.0
TOP_P       = 0.9
TOP_K       = 50

# Low temperature for the reference chain (defines the step structure;
# plays the role of the low-temperature sample judged in C1/C2)
REF_TEMP    = 0.1

# Token budget per sampled step continuation (delimiter is cut in text space)
MAX_STEP_TOKENS = 80
# Total budget for the reference chain
MAX_REF_TOKENS  = 384
# Maximum number of steps considered per question (cost cap)
MAX_STEPS       = 3

# Zero-shot CoT trigger (Kojima et al.)
COT_TRIGGER = "Let's think step by step."

# ---- C3-specific knobs ----------------------------------------------------
HIDDEN_LAYER = 16         # hidden_states[0] = embeddings; 16 = output of the
                          # 16th transformer block (mid layer; ablation knob)
STEP_POOL    = "last"     # "last" = last content token before the next
                          #          delimiter
                          # "mean" = mean-pool over continuation tokens
STEP_KERNEL  = "cosine"   # "cosine" (normalized inner-product) | "rbf"
RBF_GAMMA    = None       # None -> 1/hidden_dim
DO_ATTN_AGG  = True       # attention-weighted mean; falls back to uniform
AGG_PRIMARY  = "max"      # aggregation highlighted in the AUROC summary

# ---- Labeling ---------------------------------------------------------------
LABEL_MODE   = "judge"    # "judge" (PRIMARY; same judge as C1/C2)
                          # "f1"    (optional robustness check)
F1_THRESHOLD = 0.3

# ---- Efficiency -------------------------------------------------------------
FORWARD_BS   = 5          # mini-batch size of the hidden-state forward pass
                          # (tune to available VRAM)

OUT_DIR     = "./results/"
RESULT_FILE = f"c4_{DATASET}_results.json"
CKPT_FILE   = f"c4_{DATASET}_checkpoint.jsonl"   # resumable per-question ckpt

# Step / answer delimiters
STEP_RE   = re.compile(r"(?im)^\s*step\s*\d+\s*:")
ANSWER_RE = re.compile(r"(?im)^\s*(final\s+answer|answer)\s*:")

# =====================================================================
# Hugging Face login (token passed via the Slurm script environment)
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

_COST = {"gen_tokens": 0, "fwd_tokens": 0}

def _reset_cost():
    _COST["gen_tokens"] = 0
    _COST["fwd_tokens"] = 0

def _count_generated(out_ids, prompt_len, pad_id):
    gen = out_ids[:, prompt_len:]
    if gen.shape[1] == 0:
        return 0
    return int((gen != pad_id).sum().item())
# =====================================================================
# Model
# =====================================================================
def load_generator():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Right padding is REQUIRED: hidden-state extraction indexes the last
    # real token as attention_mask.sum() - 1, which is only valid when
    # padding is on the right.
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
# Prompts
# =====================================================================
def build_step_prompt(tokenizer, question):
    """Zero-shot CoT trigger + EXPLICIT step-delimiter formatting.

    The format instruction is what makes Scheme A operational: without
    reliable 'Step k:' delimiters, segmentation degenerates to a single
    step and C3 effectively reduces to C2.
    """
    sys_msg = (
        "You are a helpful assistant. Reason step by step. "
        "Use at most 3 concise steps. "
        "Format your reasoning as explicit numbered steps, each on its own line "
        "starting with 'Step 1:', 'Step 2:', and so on. "
        "After the steps, give the final answer on a new line starting with 'Answer:'."
    )
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"{question}\n\n{COT_TRIGGER}"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode(tokenizer, text, **kw):
    """Chat-templated text already contains BOS -> add_special_tokens=False
    is MANDATORY (otherwise a double <|begin_of_text|> is produced)."""
    return tokenizer(text, return_tensors="pt", add_special_tokens=False, **kw)


# =====================================================================
# Reference chain + segmentation
# =====================================================================
@torch.no_grad()
def generate_reference(tokenizer, model, question):
    prompt = build_step_prompt(tokenizer, question)
    inputs = encode(tokenizer, prompt).to(model.device)
    out = model.generate(
        **inputs, do_sample=True, temperature=REF_TEMP,
        top_p=1.0, top_k=0, max_new_tokens=MAX_REF_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    _COST["gen_tokens"] += _count_generated(out, inputs["input_ids"].shape[1], tokenizer.eos_token_id)
    chain = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return prompt, chain


def segment_steps(chain):
    """Segment the reference chain into steps.

    Returns:
      steps = [{'label_end': int, 'text': str, 'span': (s, e)}]
      answer_span = (s, e) | None
    label_end = char index where the 'Step i:' label ends
    (the re-sampling prefix ends exactly here).
    """
    ans_match = ANSWER_RE.search(chain)
    ans_start = ans_match.start() if ans_match else len(chain)

    # Drop spurious 'Step' labels appearing AFTER the answer line
    step_labels = [m for m in STEP_RE.finditer(chain) if m.start() < ans_start]

    steps = []
    for k, m in enumerate(step_labels):
        label_end  = m.end()
        body_start = label_end
        if k + 1 < len(step_labels):
            body_end = step_labels[k + 1].start()
        else:
            body_end = ans_start
        text = chain[body_start:body_end].strip()
        steps.append({
            "label_end": label_end,
            "text": text,
            "span": (m.start(), body_end),
        })

    answer_span = (ans_start, len(chain)) if ans_match else None
    return steps, answer_span


def truncate_at_next_delimiter(continuation):
    """Cut the continuation BEFORE the first Step/Answer delimiter
    (delimiter excluded), so that hidden states never include delimiter
    tokens. If the model emitted a delimiter immediately and the content
    would be empty, fall back to the raw continuation."""
    cut = len(continuation)
    m_step = STEP_RE.search(continuation)
    m_ans  = ANSWER_RE.search(continuation)
    if m_step:
        cut = min(cut, m_step.start())
    if m_ans:
        cut = min(cut, m_ans.start())
    truncated = continuation[:cut].rstrip()
    return truncated if truncated else continuation.strip()


# =====================================================================
# Per-step hidden-state sampling (SCHEME A)
# =====================================================================
@torch.no_grad()
def sample_step_hidden(tokenizer, model, prefix_text, n):
    """Sample n continuations from prefix_text (batched), truncate each in
    text space before the next delimiter, then extract hidden states at
    layer HIDDEN_LAYER via separate forward passes.

      STEP_POOL == "last": last CONTENT token of the truncated continuation
      STEP_POOL == "mean": mean-pool over the continuation tokens

    Returns: (n, hidden_dim) float32 tensor.

    NOTE: with the previous StoppingCriteria approach, the last generated
    token was part of the NEXT delimiter ("Step 2:" / "Answer:"); since all
    samples ended on the same delimiter token, the vectors coincided and the
    kernel collapsed to a single cluster. Here the delimiter is discarded in
    text space and hidden states are read from content tokens only.
    """
    inputs = tokenizer(prefix_text, return_tensors="pt",
                       add_special_tokens=False).to(model.device)

    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
        num_return_sequences=n,
        max_new_tokens=MAX_STEP_TOKENS,
        min_new_tokens=4,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    _COST["gen_tokens"] += _count_generated(out, prompt_len, tokenizer.eos_token_id)
    continuations = [
        tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True)
        for i in range(n)
    ]
    continuations = [truncate_at_next_delimiter(c) for c in continuations]

    # Prefix token length (start of the continuation for mean pooling).
    # Tokenization of the concatenated string may differ by +-1 token at the
    # boundary; this is tolerated via clamping.
    prefix_len = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])

    full_texts = [prefix_text + c for c in continuations]

    vecs = []
    for b0 in range(0, n, FORWARD_BS):
        batch = full_texts[b0:b0 + FORWARD_BS]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        _COST["fwd_tokens"] += int(enc["attention_mask"].sum().item())
        fout = model(**enc, output_hidden_states=True)
        hs = fout.hidden_states[HIDDEN_LAYER]          # (b, T, d)
        mask = enc["attention_mask"]                   # (b, T)
        for j in range(hs.shape[0]):
            L = int(mask[j].sum().item())              # true length (right pad)
            if STEP_POOL == "last":
                v = hs[j, L - 1, :]
            elif STEP_POOL == "mean":
                start = min(prefix_len, L - 1)
                v = hs[j, start:L, :].mean(dim=0)
            else:
                raise ValueError(f"Unknown STEP_POOL: {STEP_POOL!r}")
            vecs.append(v.float().cpu())
        del fout, hs
    return torch.stack(vecs, dim=0)  # (n, hidden_dim)


# =====================================================================
# Step kernel + VNE (kernel construction is OURS, VNE is from the repo)
# =====================================================================
def build_step_kernel(H):
    """H: (n, d) hidden vectors -> (n, n) PSD kernel."""
    H = H.numpy().astype(np.float64)
    if STEP_KERNEL == "cosine":
        norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
        Hn = H / norms
        K = Hn @ Hn.T                      # Gram matrix -> PSD, cosine sim.
    elif STEP_KERNEL == "rbf":
        sq = np.sum(H**2, axis=1, keepdims=True)
        d2 = sq + sq.T - 2.0 * (H @ H.T)
        d2 = np.maximum(d2, 0.0)
        gamma = RBF_GAMMA if RBF_GAMMA is not None else 1.0 / H.shape[1]
        K = np.exp(-gamma * d2)            # PSD
    else:
        raise ValueError(f"Unknown STEP_KERNEL: {STEP_KERNEL!r}")
    return K


def step_vne(H):
    """Step uncertainty = vn_entropy(K_i). Normalization follows the repo."""
    K = build_step_kernel(H)
    for jitter in [0, 1e-16, 1e-12, 1e-10]:
        try:
            return float(kle.core.vn_entropy(K, normalize=True, scale=False, jitter=jitter))
        except Exception:
            continue
    raise ValueError("VNE did not converge for any jitter")


# =====================================================================
# Attention-weighted aggregation weights (optional, failsafe -> uniform)
# =====================================================================
@torch.no_grad()
def attention_step_weights(tokenizer, model, prompt, chain, steps, answer_span):
    """Attention mass flowing from the final-answer tokens to each step's
    tokens (last layer, head-averaged). Returns None on failure -> caller
    falls back to uniform weights.
    (With SDPA, output_attentions=True falls back to eager with a warning;
    this is expected and harmless.)"""
    full = prompt + chain
    enc = tokenizer(full, return_tensors="pt", return_offsets_mapping=True,
                    add_special_tokens=False)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}

    # The chain starts right after the prompt -> char offset shift
    base = len(prompt)

    def char_span_to_tok(cs, ce):
        cs += base; ce += base
        return [i for i, (a, b) in enumerate(offsets) if (b > cs and a < ce)]

    if answer_span is None:
        return None
    ans_tok = char_span_to_tok(*answer_span)
    if not ans_tok:
        return None

    # Temporarily switch to eager attention for this single forward pass:
    # SDPA does not return attention matrices (out.attentions stays None).
    prev = getattr(model.config, "_attn_implementation", None)
    try:
        model.config._attn_implementation = "eager"
        out = model(**enc, output_attentions=True)
    finally:
        if prev is not None:
            model.config._attn_implementation = prev

    if out.attentions is None or out.attentions[-1] is None:
        return None   # graceful fallback -> uniform weights

    att = out.attentions[-1][0].mean(dim=0)   # (q, k), head-averaged, last layer
    ans_idx = torch.tensor(ans_tok, device=att.device)

    weights = []
    for st in steps:
        st_tok = char_span_to_tok(*st["span"])
        if not st_tok:
            weights.append(0.0)
            continue
        k_idx = torch.tensor(st_tok, device=att.device)
        w = att[ans_idx][:, k_idx].sum().item()  # answer -> step attention mass
        weights.append(w)

    w = np.array(weights, dtype=np.float64)
    if w.sum() <= 0:
        return None
    return w / w.sum()


# =====================================================================
# Aggregation
# =====================================================================
def aggregate(step_scores, attn_w=None):
    s = np.array(step_scores, dtype=np.float64)
    agg = {
        "max":  float(np.max(s)),
        "mean": float(np.mean(s)),
    }
    if attn_w is not None and len(attn_w) == len(s):
        agg["attn_mean"] = float(np.sum(attn_w * s))
    else:
        agg["attn_mean"] = float(np.mean(s))  # failsafe -> uniform
    return agg


# =====================================================================
# Labeling
#   PRIMARY: LLM-as-Judge with the SAME judge prompt as C1/C2 (paper Sec. 5).
#   AUROC is only comparable across conditions with an identical labeler.
#   Token-F1 is kept as an optional robustness check (also computable
#   post-hoc from the saved details).
# =====================================================================
def _normalize_text(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(prediction, gold):
    pred_toks = _normalize_text(prediction).split()
    gold_toks = _normalize_text(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall    = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def f1_label(predicted_answer, gold_answers):
    """max-over-aliases F1; F1 < threshold -> hallucination."""
    best = max(token_f1(predicted_answer, g) for g in gold_answers)
    return best, (best < F1_THRESHOLD)


def extract_final_answer(chain, answer_span):
    if answer_span is None:
        return chain.strip()
    return chain[answer_span[0]:answer_span[1]].split(":", 1)[-1].strip()


@torch.no_grad()
def llm_judge(tokenizer, model, question, gold_answers, predicted_answer):
    """Identical judge prompt to the C1/C2 pipelines (paper Sec. 5).
    Note for the thesis: generator and judge are the same 8B model
    (self-judging); validate on a hand-labeled subset or use a stronger
    judge (e.g. Llama-3 70B) as a robustness check."""
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
    inputs = encode(tokenizer, text).to(model.device)
    out = model.generate(
        **inputs, do_sample=False, max_new_tokens=8,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()
    return gen.startswith("yes")


# =====================================================================
# Checkpointing (resumable)
# =====================================================================
def load_checkpoint(path):
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["q_idx"]] = rec
    return done


def append_checkpoint(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =====================================================================
# Main
# =====================================================================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, CKPT_FILE)

    print(f"Loading frozen dataset: {DATASET}")
    ds = load_frozen(DATASET, frozen_dir=FROZEN_DIR)
    if N_QUESTIONS:
        ds = ds[:N_QUESTIONS]

    print(f"Loading generator: {GEN_MODEL_NAME}  ({PRECISION})")
    gen_tok, gen_model = load_generator()

    done = load_checkpoint(ckpt_path)
    if done:
        print(f"Resuming: {len(done)} questions already processed. "
              f"(Delete {ckpt_path} if the config has changed!)")

    for idx, ex in enumerate(ds):
        if idx in done:
            continue
        t0 = time.perf_counter()
        _reset_cost()
        question = ex["question"]
        golds = ex["gold_answers"]
        

        try:
            # 1) Reference chain + step segmentation
            prompt, chain = generate_reference(gen_tok, gen_model, question)
            steps, answer_span = segment_steps(chain)

            # Fallback: if no delimiter is found, treat the whole chain as
            # a single step (C3 degenerates to a holistic score for this item)
            if len(steps) == 0:
                steps = [{"label_end": 0, "text": chain.strip(),
                          "span": (0, len(chain))}]
            steps = steps[:MAX_STEPS]

            # 2) SCHEME A per step: n continuations from the prefix,
            #    hidden states -> K_i -> VNE_i
            step_scores = []
            for st in steps:
                prefix_text = prompt + chain[: st["label_end"]]
                H = sample_step_hidden(gen_tok, gen_model, prefix_text, N_SAMPLES)
                step_scores.append(step_vne(H))

            # 3) Attention weights (optional)
            attn_w = None
            if DO_ATTN_AGG:
                try:
                    attn_w = attention_step_weights(
                        gen_tok, gen_model, prompt, chain, steps, answer_span
                    )
                except Exception as e:
                    print(f"  [attn warn] {e}")
                    attn_w = None

            # 4) Response-level aggregation
            agg = aggregate(step_scores, attn_w=attn_w)

            # 5) Hallucination label (PRIMARY: judge, consistent with C1/C2)
            candidate = extract_final_answer(chain, answer_span)
            if LABEL_MODE == "judge":
                is_correct = llm_judge(gen_tok, gen_model, question, golds, candidate)
                is_halluc  = (not is_correct)
                label_info = {"judge_correct": is_correct}
            elif LABEL_MODE == "f1":
                f1, is_halluc = f1_label(candidate, golds)
                label_info = {"f1": f1}
            else:
                raise ValueError(f"Unknown LABEL_MODE: {LABEL_MODE!r}")

            rec = {
                "q_idx":         idx,
                "qid":           ex["id"],
                "question":      question,
                "chain":         chain,
                "n_steps":       len(steps),
                "step_scores":   step_scores,
                "attn_weights":  None if attn_w is None else attn_w.tolist(),
                "agg":           agg,
                "candidate":     candidate,
                "label_info":    label_info,
                "is_halluc":     int(is_halluc),
                "error":         None,
            }
        except Exception as e:
            print(f"  [error q{idx}] {type(e).__name__}: {e}")
            rec = {"q_idx": idx, "qid": ex["id"], "question": question, "error": str(e)}
            
        rec["cost"] = {
            "t_seconds":  time.perf_counter() - t0,
            "gen_tokens": _COST["gen_tokens"],   # referans zincir + tüm step sample'ları
            "fwd_tokens": _COST["fwd_tokens"],   # hidden-state forward pass'leri (C4'e özel)
        }
        append_checkpoint(ckpt_path, rec)
        done[idx] = rec

        if (idx + 1) % 5 == 0:
            ok = rec.get("error") is None
            msg = (f"steps={rec['n_steps']}  max={rec['agg']['max']:.3f}  "
                   f"halluc={rec['is_halluc']}") if ok else "ERROR"
            print(f"  [{idx+1:3d}/{N_QUESTIONS}]  {msg}")

    # ---- AUROC per aggregation (over error-free records only) --------------
    valid = [r for r in done.values() if r.get("error") is None]
    labels = [r["is_halluc"] for r in valid]
    scores_by_agg = defaultdict(list)
    for r in valid:
        for k, v in r["agg"].items():
            scores_by_agg[k].append(v)

    rate = (sum(labels) / len(labels)) if labels else float("nan")
    aurocs = {}
    for k, sc in scores_by_agg.items():
        try:
            aurocs[k] = roc_auc_score(labels, sc)
        except Exception:
            # e.g. all labels in one class on small N -> AUROC undefined
            aurocs[k] = None

    print(f"\nValid questions    : {len(valid)}/{len(done)}")
    print(f"Hallucination rate : {rate:.2%}")
    for k, a in aurocs.items():
        astr = f"{a:.4f}" if a is not None else "n/a"
        print(f"AUROC [{k:10s}]  : {astr}")
    print(f"PRIMARY ({AGG_PRIMARY}) -> {aurocs.get(AGG_PRIMARY)}")

    out = {
        "config": {
            "condition":       "C4_cot_stepwise_scheme",
            "generator":       GEN_MODEL_NAME,
            "precision":       PRECISION,
            "n_questions":     N_QUESTIONS,
            "n_samples":       N_SAMPLES,
            "temperature":     TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
            "ref_temp":        REF_TEMP,
            "max_step_tokens": MAX_STEP_TOKENS,
            "max_steps":       MAX_STEPS,
            "hidden_layer":    HIDDEN_LAYER,
            "step_pool":       STEP_POOL,
            "step_kernel":     STEP_KERNEL,
            "rbf_gamma":       RBF_GAMMA,
            "cot_trigger":     COT_TRIGGER,
            "label_mode":      LABEL_MODE,
            "f1_threshold":    F1_THRESHOLD,
            "agg_primary":     AGG_PRIMARY,
            "seed":            SEED,
        },
        "auroc":              aurocs,
        "hallucination_rate": rate,
        "n_valid":            len(valid),
        "details":            sorted(done.values(), key=lambda r: r["q_idx"]),
    }
    path = os.path.join(OUT_DIR, RESULT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()