"""
Semantic Entropy baseline (Kuhn 2023 / Farquhar 2024) — pure post-hoc, no GPU.

Recomputes SE from the SAVED black-box C2 result files (all datasets and
models): the stored `semantic_ids` (strict-entailment NLI clusters, identical
to the ones KLE used) and `log_lik_per_sem_id` are re-scored with the two SE
estimators:

  se_discrete : -sum_c (n_c/N) log(n_c/N)      cluster proportions
                -> fully BLACK-BOX (texts + NLI only)
  se_weighted : -sum_c p_c log p_c, p_c = softmax(log_lik_per_sem_id)
                -> GREY-BOX (needs sequence log-probs), original SE

Because clustering is shared with KLE, this is a clean entropy-formula
ablation: same clusters, different entropy. Labels (`judge_correct`) are
reused from the source files -> AUROCs directly comparable to the reported
KLE numbers (also echoed per cell as `kle_reported`).

Runs anywhere (numpy + sklearn; no torch, no GPU):
    python baselines/semantic_entropy_posthoc.py
Writes -> results/se_c2_results.json (missing cells skipped with a note).

Scope is C2 only: the thesis baseline comparison (Sec. 5.7) is defined
against the C2 generations specifically, and C4 has no persisted per-step
sample texts/semantic_ids to re-cluster post-hoc (only the aggregated
step_scores are saved), so C1/C3/C4 are intentionally out of scope here.
"""

import os
import json

import numpy as np
from sklearn.metrics import roc_auc_score

OUT_DIR  = "./results/"
OUT_FILE = "se_c2_results.json"

CONDITIONS = ["c2"]
DATASETS   = ["triviaqa", "nqopen", "morehopqa"]
MODELS     = ["8b", "mistral"]


# =====================================================================
# Input resolution (three naming schemes coexist in results/)
# =====================================================================
def resolve_bb_file(cond, dataset, model_tag):
    cands = [
        os.path.join(OUT_DIR, f"{cond}_{dataset}_{model_tag}_BB_results.json"),
        os.path.join(OUT_DIR, "black-box", dataset, f"{cond}_{dataset}_{model_tag}_BB_results.json"),
    ]
    if model_tag == "8b":  # legacy TriviaQA naming ("llama", lowercase bb)
        cands.append(os.path.join(OUT_DIR, "black-box", dataset, f"{cond}_{dataset}_llama_bb.json"))
        if cond == "c3" and dataset == "triviaqa":
            # known filename typo in the legacy c3 file ("triviaiqa")
            cands.append(os.path.join(OUT_DIR, "black-box", dataset, "c3_triviaiqa_llama_bb.json"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


# =====================================================================
# SE estimators
# =====================================================================
def se_discrete(sem_ids):
    _, counts = np.unique(np.asarray(sem_ids), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def se_weighted(log_lik_per_sem_id):
    ll = np.asarray(log_lik_per_sem_id, dtype=np.float64)
    ll = ll - ll.max()                      # stable softmax
    p = np.exp(ll)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# =====================================================================
# Main
# =====================================================================
def main():
    cells, skipped = {}, []

    for cond in CONDITIONS:
        for dataset in DATASETS:
            for model_tag in MODELS:
                key = f"{cond}_{dataset}_{model_tag}"
                path = resolve_bb_file(cond, dataset, model_tag)
                if path is None:
                    skipped.append(key)
                    continue

                with open(path, encoding="utf-8") as f:
                    src = json.load(f)

                details = []
                for x in src["details"]:
                    details.append({
                        "id":          x["id"],
                        "se_discrete": se_discrete(x["semantic_ids"]),
                        "se_weighted": se_weighted(x["log_lik_per_sem_id"]),
                        "is_halluc":   int(not x["judge_correct"]),
                    })

                y = [d["is_halluc"] for d in details]
                aurocs = {}
                for var in ["se_discrete", "se_weighted"]:
                    vals = [d[var] for d in details]
                    aurocs[var] = float(roc_auc_score(y, vals)) if len(set(y)) > 1 else None

                cells[key] = {
                    "source_file":        path,
                    "n":                  len(details),
                    "hallucination_rate": float(np.mean(y)),
                    "auroc":              aurocs,
                    "kle_reported":       src.get("auroc"),   # same samples, same labels
                    "details":            details,
                }
                print(f"{key:28s} n={len(details):4d}  "
                      f"SE_disc={aurocs['se_discrete']:.4f}  "
                      f"SE_wgt={aurocs['se_weighted']:.4f}  "
                      f"KLE={src.get('auroc'):.4f}  "
                      f"halluc={np.mean(y):.3f}")

    if skipped:
        print(f"\nSkipped (no BB source file yet): {', '.join(skipped)}")

    out = {
        "config": {
            "condition":  "semantic_entropy_posthoc_c1c2c3",
            "variants": {
                "se_discrete": "cluster-proportion entropy (black-box)",
                "se_weighted": "softmax(log_lik_per_sem_id) entropy (grey-box, original SE)",
            },
            "clustering": "reused semantic_ids from saved BB runs (strict entailment)",
            "labels":     "reused judge_correct from saved BB runs",
        },
        "skipped": skipped,
        "cells":   cells,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, OUT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()