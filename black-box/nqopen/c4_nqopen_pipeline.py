"""
Repo-faithful BLACK-BOX STEP-WISE KLE pipeline on TriviaQA / NQOpen.
CONDITION C4: Zero-shot CoT (+ step-format instruction) + step-wise BLACK-BOX KLE.
HPC (Slurm) compatible version with resumable JSONL checkpointing.

Fully aligned with the latest master-thesis repository layout (dynamic paths, argparse).
"""

import os
import sys
import re
import json
import string
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import time

from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from huggingface_hub import login

# =====================================================================
# Repo-faithful Path & Sys Setup (Exactly matching your repository)
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, os.path.join(root_dir, "kernel-language-entropy"))

import kle.core
import kle.kernels

sys.path.insert(0, root_dir)
from dataset.frozen_utils import load_frozen

# =====================================================================
# Config & Dynamic Argument Parsing (Exactly matching your repository)
# =====================================================================
DATASET    = "nqopen"  # or "nqopen" depending on the folder it sits in
FROZEN_DIR = os.path.join(root_dir, "dataset")
SEED         = 42
N_QUESTIONS  = 1000          
N_SAMPLES    = 10          # samples per step (n)

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

NLI_MODEL_NAME = "microsoft/deberta-v2-xlarge-mnli"
PRECISION      = "bf16"

# Sampling Parameters
TEMPERATURE = 1.0
TOP_P       = 0.9
TOP_K       = 50
REF_TEMP    = 0.1

MAX_STEP_TOKENS = 80
MAX_REF_TOKENS  = 384
MAX_STEPS       = 3

# KLE / KLU Hyperparameters (Matched with black-box C2/C3)
HEAT_T = 0.3
ALPHA  = 0.5
STRICT_ENTAILMENT = True

COT_TRIGGER  = "Let's think step by step."
LABEL_MODE   = "judge"    
AGG_PRIMARY  = "max"      

OUT_DIR     = "./results/"
RESULT_FILE = f"c4_{DATASET}_{MODEL_TAG}_BB_results.json"
CKPT_FILE   = f"c4_{DATASET}_{MODEL_TAG}_BB_checkpoint.jsonl"

STEP_RE   = re.compile(r"(?im)^\s*step\s*\d+\s*:")
ANSWER_RE = re.compile(r"(?im)^\s*(final\s+answer|answer)\s*:")

# =====================================================================
# Hugging Face login
# =====================================================================
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("No HF Token!")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_COST = {"gen_tokens": 0}

def _reset_cost():
    _COST["gen_tokens"] = 0

# =====================================================================
# Models
# =====================================================================
def load_generator():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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


class EntailmentDebertaLite:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_NAME, torch_dtype=torch.float16
        ).to("cuda")
        self.model.eval()
        self.cache = {}

    @torch.no_grad()
    def check_implication(self, text1, text2):
        key = (text1, text2)
        if key in self.cache:
            return self.cache[key]
        inputs = self.tokenizer(
            text1, text2, return_tensors="pt", truncation=True, max_length=512,
        ).to("cuda")
        logits = self.model(**inputs).logits
        probs  = F.softmax(logits, dim=1)
        pred   = int(torch.argmax(probs).item())
        conf   = float(torch.max(probs).item())
        self.cache[key] = (pred, conf)
        return pred, conf


# =====================================================================
# Helpers & Black-Box Core Functions (Verbatim from C2/C3 Repo files)
# =====================================================================
def build_step_prompt(tokenizer, question):
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


def segment_steps(chain):
    ans_match = ANSWER_RE.search(chain)
    ans_start = ans_match.start() if ans_match else len(chain)
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
    cut = len(continuation)
    m_step = STEP_RE.search(continuation)
    m_ans  = ANSWER_RE.search(continuation)
    if m_step: cut = min(cut, m_step.start())
    if m_ans: cut = min(cut, m_ans.start())
    truncated = continuation[:cut].rstrip()
    return truncated if truncated else continuation.strip()


def get_semantic_ids(strings_list, model, strict_entailment=False):
    def are_equivalent(text1, text2):
        impl_1, _ = model.check_implication(text1, text2)
        impl_2, _ = model.check_implication(text2, text1)
        if strict_entailment:
            return (impl_1 == 2) and (impl_2 == 2)
        implications = [impl_1, impl_2]
        return (0 not in implications) and ([1, 1] != implications)

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    for i, s1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(s1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1
    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    unique_ids = sorted(list(set(semantic_ids)))
    log_likelihood_per_semantic_id = []
    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)
    return unique_ids, log_likelihood_per_semantic_id


def get_entailment_graph(strings_list, model):
    def get_edge(t1, t2):
        impl_1, _ = model.check_implication(t1, t2)
        impl_2, _ = model.check_implication(t2, t1)
        weight = (int(impl_1 == 2) + int(impl_2 == 2) + 0.5 * int(impl_1 == 1) + 0.5 * int(impl_2 == 1))
        return weight

    nodes = range(len(strings_list))
    edges = []
    for i, s1 in enumerate(strings_list):
        for j in range(i + 1, len(strings_list)):
            edge = get_edge(s1, strings_list[j])
            if edge > 0:
                edges.append((i, j, edge))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    return G


def get_from_sem_to_sentence_id(ordered_ids):
    out = defaultdict(list)
    for i, el in enumerate(ordered_ids):
        out[el].append(i)
    return out


def reorder_by_semantic_ids(graph, semantic_ids, ordered_sem_ids):
    from_sem_to_sentence_id = get_from_sem_to_sentence_id(semantic_ids)
    new_graph = nx.Graph()
    for sem_id in ordered_sem_ids:
        for sent_id in from_sem_to_sentence_id[sem_id]:
            new_graph.add_node(sent_id)
    new_graph.add_edges_from(graph.edges(data=True))
    return new_graph


def get_block_diagonal_sem_kernel(log_likelihoods_per_sem_id, semantic_ids, ordered_sem_ids):
    from_sem_to_sentence_id = get_from_sem_to_sentence_id(semantic_ids)
    blocks = []
    for sem_id in ordered_sem_ids:
        block_size = len(from_sem_to_sentence_id[sem_id])
        block = (torch.exp(torch.tensor(log_likelihoods_per_sem_id[sem_id])) * torch.ones((block_size, block_size)) / block_size)
        blocks.append(block)
    return torch.block_diag(*blocks)


def full_klu_score(graph, log_lik_per_sem_id, semantic_ids, ordered_sem_ids, t=HEAT_T, alpha=ALPHA):
    graph = reorder_by_semantic_ids(graph, semantic_ids, ordered_sem_ids)
    block_diag = get_block_diagonal_sem_kernel(log_lik_per_sem_id, semantic_ids, ordered_sem_ids)
    K_heat   = kle.kernels.heat_kernel(graph, t=t)
    K_normed = kle.core.normalize_kernel(K_heat) / K_heat.shape[0]
    K_full   = alpha * torch.tensor(K_normed) + (1.0 - alpha) * block_diag
    K_full   = K_full.numpy()

    for jitter in [0, 1e-16, 1e-12]:
        try:
            return kle.core.vn_entropy(K_full, normalize=False, scale=False, jitter=jitter)
        except Exception:
            continue
    raise ValueError("VNE did not converge for any jitter")


# =====================================================================
# Step-wise Text Sampling + Transition Scores
# =====================================================================
@torch.no_grad()
def sample_step_text_with_scores(tokenizer, model, prefix_text, n):
    inputs = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
        num_return_sequences=n,
        max_new_tokens=MAX_STEP_TOKENS,
        min_new_tokens=4,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    ts = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    
    continuations = []
    log_liks_agg = []
    
    for i in range(n):
        gen_ids = out.sequences[i, prompt_len:]
        non_pad_mask = gen_ids != tokenizer.eos_token_id
        true_gen_len = int(non_pad_mask.sum().item())
        _COST["gen_tokens"] += true_gen_len
        
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        truncated_text = truncate_at_next_delimiter(text)
        continuations.append(truncated_text)
        
        seq_scores = ts[i, :true_gen_len].cpu().numpy()
        if len(seq_scores) == 0:
            seq_scores = [0.0]
        log_liks_agg.append(float(np.mean(seq_scores)))
        
    return continuations, log_liks_agg


# =====================================================================
# Reference Generation & Judging
# =====================================================================
@torch.no_grad()
def generate_reference(tokenizer, model, question):
    prompt = build_step_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, do_sample=True, temperature=REF_TEMP,
        top_p=1.0, top_k=0, max_new_tokens=MAX_REF_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:]
    _COST["gen_tokens"] += int((gen_ids != tokenizer.eos_token_id).sum().item())
    chain = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return prompt, chain


def extract_final_answer(chain, answer_span):
    if answer_span is None:
        return chain.strip()
    return chain[answer_span[0]:answer_span[1]].split(":", 1)[-1].strip()


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
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, do_sample=False, max_new_tokens=8,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()
    return gen.startswith("yes")


# =====================================================================
# Checkpointing (Resumable)
# =====================================================================
def load_checkpoint(path):
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                done[rec["q_idx"]] = rec
    return done


def append_checkpoint(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =====================================================================
# Main Loop
# =====================================================================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, CKPT_FILE)

    print(f"Loading frozen dataset from repo layout: {DATASET}")
    ds = load_frozen(DATASET, frozen_dir=FROZEN_DIR)
    if N_QUESTIONS:
        ds = ds[:N_QUESTIONS]

    print(f"Loading generator: {GEN_MODEL_NAME} ({PRECISION})")
    gen_tok, gen_model = load_generator()

    print(f"Loading NLI baseline model: {NLI_MODEL_NAME}")
    nli = EntailmentDebertaLite()

    done = load_checkpoint(ckpt_path)
    if done:
        print(f"Resuming checkpoint: {len(done)} questions processed.")

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

            if len(steps) == 0:
                steps = [{"label_end": 0, "text": chain.strip(), "span": (0, len(chain))}]
            steps = steps[:MAX_STEPS]

            # 2) SCHEME A per step: Prefix-conditioned text sampling + KLU mixture
            step_scores = []
            for st in steps:
                prefix_text = prompt + chain[: st["label_end"]]
                
                # Sample text and collect transition log-likelihoods
                step_continuations, log_liks_agg = sample_step_text_with_scores(
                    gen_tok, gen_model, prefix_text, N_SAMPLES
                )
                
                # DeBERTa Semantics Matrix Package
                semantic_ids = get_semantic_ids(
                    step_continuations, model=nli, strict_entailment=STRICT_ENTAILMENT
                )
                unique_ids, log_lik_per_sem_id = logsumexp_by_id(
                    semantic_ids, log_liks_agg, agg="sum_normalized"
                )
                weighted_graph = get_entailment_graph(step_continuations, model=nli)
                
                # Calculate exact KLU step score matching C2/C3 mix logic
                step_score = full_klu_score(
                    weighted_graph, log_lik_per_sem_id,
                    semantic_ids=semantic_ids, ordered_sem_ids=unique_ids,
                    t=HEAT_T, alpha=ALPHA
                )
                step_scores.append(step_score)

            # 3) Aggregation (Max / Mean only - sticking to pure black-box)
            agg = {
                "max":  float(np.max(step_scores)),
                "mean": float(np.mean(step_scores))
            }

            # 4) Hallucination label
            candidate = extract_final_answer(chain, answer_span)
            is_correct = llm_judge(gen_tok, gen_model, question, golds, candidate)
            is_halluc  = (not is_correct)

            rec = {
                "q_idx":         idx,
                "qid":           ex["id"],
                "question":      question,
                "chain":         chain,
                "n_steps":       len(steps),
                "step_scores":   step_scores,
                "agg":           agg,
                "candidate":     candidate,
                "is_halluc":     int(is_halluc),
                "error":         None,
            }
        except Exception as e:
            print(f"  [error q{idx}] {type(e).__name__}: {e}")
            rec = {"q_idx": idx, "qid": ex["id"], "question": question, "error": str(e)}
            
        rec["cost"] = {
            "t_seconds":  time.perf_counter() - t0,
            "gen_tokens": _COST["gen_tokens"]
        }
        append_checkpoint(ckpt_path, rec)
        done[idx] = rec

        if (idx + 1) % 5 == 0:
            ok = rec.get("error") is None
            msg = f"steps={rec['n_steps']}  max={rec['agg']['max']:.3f}  halluc={rec['is_halluc']}" if ok else "ERROR"
            print(f"  [{idx+1:3d}/{N_QUESTIONS}]  {msg}")

    # ---- AUROC Summary ----
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
            aurocs[k] = None

    print(f"\nValid questions    : {len(valid)}/{len(done)}")
    print(f"Hallucination rate : {rate:.2%}")
    for k, a in aurocs.items():
        astr = f"{a:.4f}" if a is not None else "n/a"
        print(f"AUROC [{k:10s}]  : {astr}")
    print(f"PRIMARY ({AGG_PRIMARY}) -> {aurocs.get(AGG_PRIMARY)}")

    out = {
        "config": {
            "condition":         "C4_cot_stepwise_blackbox_scheme",
            "generator":         GEN_MODEL_NAME,
            "precision":         PRECISION,
            "nli":               NLI_MODEL_NAME,
            "n_questions":       N_QUESTIONS,
            "n_samples":         N_SAMPLES,
            "temperature":       TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
            "ref_temp":          REF_TEMP,
            "max_step_tokens":   MAX_STEP_TOKENS,
            "max_steps":         MAX_STEPS,
            "heat_t":            HEAT_T,
            "alpha":             ALPHA,
            "strict_entailment": STRICT_ENTAILMENT,
            "seed":              SEED,
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