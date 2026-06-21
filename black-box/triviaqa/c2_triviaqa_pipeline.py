"""
Repo-faithful WHITE-BOX KLE pipeline on TriviaQA.
C2 KOŞULU: Zero-shot CoT prompting + HOLISTIC KLE (tüm zincir metni tek birim).
HPC (Slurm) Uyumlu Versiyon

Standart prompting baseline'a (C1) göre TEK fark prompting katmanıdır:
  - build_prompt -> zero-shot CoT ("Let's think step by step")
  - MAX_NEW_TOKENS artırıldı (reasoning zinciri 64 token'a sığmaz)
KLE matematiği, semantic clustering, judge ve metrikler C1 ile AYNIDIR.
KLE girdisi = örneklenen TAM CoT çıktısı (holistic; final answer ayıklanmaz).
"""

import os
import sys
import json
import argparse
from collections import defaultdict
import time
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
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
NLI_MODEL_NAME = "microsoft/deberta-v2-xlarge-mnli"

PRECISION = "bf16"

# Sampling (paper Sec. 5)
TEMPERATURE    = 1.0
TOP_P          = 0.9
TOP_K          = 50
# CoT zincirleri tek cümleden uzun olduğu için artırıldı.
# 64 token'da reasoning kesilir, final answer hiç üretilmez.
MAX_NEW_TOKENS = 200

# Judge (paper Sec. 5)
JUDGE_TEMP     = 0.1

# Zero-shot CoT "magic phrase" (Kojima et al., 2024)
COT_TRIGGER = "Let's think step by step."

# KLE hyperparameters
HEAT_T = 0.3
ALPHA  = 0.5
STRICT_ENTAILMENT = True

# Çıktı Dizini (Colab Drive yerine HPC'de yerel bir klasör kullanıyoruz)
OUT_DIR     = "./results/"
RESULT_FILE = f"c2_{DATASET}_{MODEL_TAG}_results.json"

# =====================================================================
# Hugging Face Login (Slurm script'ten gelen çevresel değişken ile)
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
# Models
# =====================================================================
def load_generator():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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
    def check_implication(self, text1, text2, *args, **kwargs):
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
# Helpers
# =====================================================================
def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    def are_equivalent(text1, text2):
        impl_1, _ = model.check_implication(text1, text2, example=example)
        impl_2, _ = model.check_implication(text2, text1, example=example)
        assert (impl_1 in [0, 1, 2]) and (impl_2 in [0, 1, 2])
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
    assert -1 not in semantic_set_ids
    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
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


def get_entailment_graph(strings_list, model, is_weighted=False, example=None, weight_strategy="manual"):
    def get_edge(t1, t2, is_weighted=False, example=None):
        impl_1, p1 = model.check_implication(t1, t2, example=example)
        impl_2, p2 = model.check_implication(t2, t1, example=example)
        assert impl_1 in [0, 1, 2]
        weight = (int(impl_1 == 2) + int(impl_2 == 2) +
                  0.5 * int(impl_1 == 1) + 0.5 * int(impl_2 == 1))
        if is_weighted:
            if weight_strategy == "manual":
                return weight
            elif weight_strategy == "deberta":
                return p1 + p2
            else:
                raise ValueError
        return weight >= 1.5

    nodes = range(len(strings_list))
    edges = []
    for i, s1 in enumerate(strings_list):
        for j in range(i + 1, len(strings_list)):
            edge = get_edge(s1, strings_list[j], example=example, is_weighted=is_weighted)
            if is_weighted:
                if edge:
                    edges.append((i, j, edge))
            else:
                edges.append((i, j))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    if is_weighted:
        G.add_weighted_edges_from(edges)
    else:
        G.add_edges_from(edges)
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
        block = (torch.exp(torch.tensor(log_likelihoods_per_sem_id[sem_id]))
                 * torch.ones((block_size, block_size)) / block_size)
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
# Generation & Judging
# =====================================================================
def build_prompt(tokenizer, question):
    # Zero-shot CoT: sistem mesajı reasoning'i serbest bırakır,
    # kullanıcı mesajına "magic phrase" eklenir.
    msgs = [
        {"role": "system", "content": "You are a helpful assistant. Reason step by step, but keep your reasoning concise (under 3 sentences). Always conclude with your final answer."},
        {"role": "user", "content": f"{question}\n\n{COT_TRIGGER}"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def sample_with_logliks(tokenizer, model, question):
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[1]

    texts, all_logliks = [], []
    for _ in range(N_SAMPLES):
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
        gen_ids = out.sequences[0, in_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        ts = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        logliks = [s.item() for s in ts[0]]

        if len(logliks) == 0:
            logliks = [0.0]

        texts.append(text)
        all_logliks.append(logliks)

    return texts, all_logliks


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

    print(f"Loading NLI: {NLI_MODEL_NAME}")
    nli = EntailmentDebertaLite()

    kle_scores, labels, details = [], [], []

    for idx, ex in enumerate(ds):
        t0 = time.perf_counter()
        question = ex["question"]
        golds = ex["gold_answers"]

        responses, log_liks = sample_with_logliks(gen_tok, gen_model, question)
        log_liks_agg = [float(np.mean(ll)) for ll in log_liks]

        semantic_ids = get_semantic_ids(
            responses, model=nli,
            strict_entailment=STRICT_ENTAILMENT, example=None,
        )

        unique_ids, log_lik_per_sem_id = logsumexp_by_id(
            semantic_ids, log_liks_agg, agg="sum_normalized",
        )

        weighted_graph = get_entailment_graph(
            responses, model=nli, is_weighted=True, weight_strategy="manual",
        )

        score = full_klu_score(
            weighted_graph, log_lik_per_sem_id,
            semantic_ids=semantic_ids, ordered_sem_ids=unique_ids,
            t=HEAT_T, alpha=ALPHA,
        )

        candidate  = low_temp_sample(gen_tok, gen_model, question)
        is_correct = llm_judge(gen_tok, gen_model, question, golds, candidate)
        is_halluc  = (not is_correct)

        kle_scores.append(score)
        labels.append(int(is_halluc))
        cost = {
            "t_seconds":       time.perf_counter() - t0,
            "n_gen_tokens":    int(sum(len(ll) for ll in log_liks)),   # tüm sample'lar toplamı, exact
            "n_prompt_tokens": int(len(gen_tok(build_prompt(gen_tok, question))["input_ids"])),
        }
        details.append({
            "id":                   ex["id"],
            "question":             question,
            "responses":            responses,
            "log_lik_agg":          log_liks_agg,
            "semantic_ids":         semantic_ids,
            "log_lik_per_sem_id":   log_lik_per_sem_id,
            "judge_candidate":      candidate,
            "judge_correct":        is_correct,
            "kle_full":             score,
            "cost":                 cost,
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1:3d}/{N_QUESTIONS}]  "
                  f"KLE_FULL={score:.3f}  clusters={len(unique_ids)}  "
                  f"correct={is_correct}")

    auroc = roc_auc_score(labels, kle_scores)
    rate  = sum(labels) / len(labels)
    print(f"\nAUROC               : {auroc:.4f}")
    print(f"Hallucination rate  : {rate:.2%}")

    out = {
        "config": {
            "condition":         "C2_cot_holistic",
            "generator":         GEN_MODEL_NAME,
            "precision":         PRECISION,
            "nli":               NLI_MODEL_NAME,
            "n_questions":       N_QUESTIONS,
            "n_samples":         N_SAMPLES,
            "temperature":       TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
            "max_new_tokens":    MAX_NEW_TOKENS,
            "cot_trigger":       COT_TRIGGER,
            "heat_t":            HEAT_T,
            "alpha":             ALPHA,
            "strict_entailment": STRICT_ENTAILMENT,
            "judge_temp":        JUDGE_TEMP,
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