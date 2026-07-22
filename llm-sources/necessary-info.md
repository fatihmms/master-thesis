# Chapter 5 – Experimental Setup Information Form

## 0. General Scope

1. Chapter 5’te resmi olarak anlatılacak deney kapsamı:
- [ ] Sadece main experiments
- [ ] Main experiments + robustness/model comparison
- [ ] Main experiments + baselines
- [x] Diğer: Main experiments (C1–C4) + robustness/model comparison (Mistral-7B) + baselines (SelfCheckGPT-NLI, LLM-Check, Semantic Entropy post-hoc)

2. Chapter 5’te kullanılacak datasetler:
- Dataset 1: TriviaQA (`rc.nocontext`)
- Dataset 2: NQ-Open
- Dataset 3: MoreHopQA (human-verified set)


3. Chapter 5’te kullanılacak modeller:
- Main generator model: `meta-llama/Meta-Llama-3.1-8B-Instruct` (argparse alias `8b`)
- Secondary / robustness model: `mistralai/Mistral-7B-Instruct-v0.3` (alias `mistral`). Repo currently contains Mistral results only for MoreHopQA; TriviaQA and NQ-Open Mistral runs completed on the cluster but not yet pulled into the repo.
- Kullanılmadıysa 70B modeli için not: `meta-llama/Meta-Llama-3-70B-Instruct` exists only as the `70b` alias in `MODEL_MAP`; never run (no results, no sbatch script uses it). Possibly to be run for MoreHopQA only — otherwise on hold.

4. White-box ve black-box sonuçlar ikisi de resmi olarak raporlanacak mı?
- White-box: Evet — all 4 conditions × 3 datasets, results in `results/white-box/<dataset>/`.
- Black-box: Evet — all 4 conditions × 3 datasets, results in `results/black-box/<dataset>/`.
- Sadece bazı condition/dataset/model kombinasyonları mı? Açıkla: Llama-3.1-8B covers all 24 cells (4 conditions × 2 box types × 3 datasets), each N=1000, zero error records. Mistral-7B: MoreHopQA results (all 8 cells) are in the repo; TriviaQA/NQ-Open Mistral results exist on the cluster but are not yet synced. Baselines are black-box/C2-style only: SelfCheckGPT-NLI (`selfcheck_c2_*_8b_results.json`: all 3 datasets, 8B), LLM-Check (`results/llm-check/`: 3 datasets × 8B + MoreHopQA × Mistral), Semantic Entropy (post-hoc from saved C1–C3 black-box logs via `baselines/semantic_entropy_posthoc.py`).


---

## 1. Dataset Information

### Dataset 1

Dataset name:
- TriviaQA (closed-book variant, `rc.nocontext`)

Dataset source / HuggingFace name / local source:
- HuggingFace `mandarjoshi/trivia_qa`, config `rc.nocontext`; frozen locally to `dataset/triviaqa.jsonl` (committed, SHA256 `010e0339…` in `dataset/manifest.json`, built 2026-06-14, `datasets` 4.5.0).

Used split:
- `validation` (17,944 source rows)

Closed-book QA olarak mı kullanıldı?
- Evet / Hayır: Evet — `rc.nocontext` config; no context passages are given to the model.

Number of evaluated questions:
- 1000 (`N_QUESTIONS = 1000` in all pipelines; all 8 result files contain exactly 1000 detail records).

Sample selection:
- [ ] First N after shuffle
- [ ] Random sample with seed
- [x] Frozen preprocessed file
- [ ] Other:
- (Freeze procedure = deterministic index shuffle with Python stdlib `random.Random(seed)` (Mersenne Twister, version-stable), then take the first 1000 *unique* questions — `select_unique` in `frozen_utils.py` with `dedupe=True`.)

Selection seed:
- 42 (`SEED = 42` in `build_frozen_datasets.py`; also the runtime seed in every pipeline via `set_seed(42)`).

Reference answer format:
- [ ] Single answer
- [x] Multiple aliases
- [ ] Long-form reference
- [ ] Other:
- (`answer.aliases` + `answer.value` merged by `adapt_triviaqa`; per-question gold count min 1 / max 159 / mean 14.6 in the frozen file.)

Any preprocessing?
- `normalize_golds`: strip whitespace, drop empty, case-insensitive de-duplication of gold strings (order preserved). Rows with empty question or no golds skipped. Question-level de-duplication (whitespace-normalized, lowercased key). Stable IDs `triviaqa-000000` … assigned at freeze time. `load_frozen` verifies the file SHA256 against the manifest at every run.

Any excluded/failed examples?
- None. All 1000 frozen questions were evaluated in every condition; 0 error records across all 8 TriviaQA result files.

Notes for thesis:
- TriviaQA result files use a legacy naming scheme (`c1_triviaqa_llama_bb.json` instead of `c1_triviaqa_8b_BB_results.json`); the black-box C3 file has a filename typo (`c3_triviaiqa_llama_bb.json`) — handled explicitly in `semantic_entropy_posthoc.py::resolve_bb_file`. Content/schema is identical to the other datasets.


### Dataset 2

Dataset name:
- NQ-Open (Natural Questions, open-domain QA variant)

Dataset source / HuggingFace name / local source:
- HuggingFace `google-research-datasets/nq_open` (no config); frozen locally to `dataset/nqopen.jsonl` (SHA256 `9ed4821a…` in manifest, built 2026-06-14, `datasets` 4.5.0).

Used split:
- `validation` (3,610 source rows)

Closed-book QA olarak mı kullanıldı?
- Evet / Hayır: Evet — NQ-Open provides question + short answers only; no context is used.

Number of evaluated questions:
- 1000 (all 8 NQ-Open result files contain exactly 1000 detail records).

Sample selection:
- Frozen preprocessed file (same procedure as Dataset 1: seed-42 deterministic shuffle, first 1000 unique questions, `dedupe=True`).

Selection seed:
- 42

Reference answer format:
- Multiple short answers (list; `adapt_nq_open` takes the `answer` list as-is). Per-question gold count min 1 / max 23 / mean 1.77 in the frozen file.

Any preprocessing?
- Same normalization pipeline as Dataset 1 (`normalize_golds`, empty-row skip, question dedupe, stable IDs, SHA256 manifest check).

Any excluded/failed examples?
- None. 1000/1000 evaluated in every condition; 0 error records.

Notes for thesis:
- ID prefix quirk: frozen IDs carry the build-time registry key `nq_open` (e.g. `nq_open-000000`) while the frozen file itself is `nqopen.jsonl`; the post-hoc tooling maps this via `DATASET_ALIAS = {"nq_open": "nqopen"}` (`analysis/cross_judge.py`). Repo currently has only Llama-3.1-8B results for NQ-Open; Mistral runs done on the cluster, not yet synced.


### Dataset 3

Dataset name:
- MoreHopQA (human-verified set)

Dataset source / HuggingFace name / local source:
- HuggingFace `alabnii/morehopqa`. The HF repo is script-based (unsupported since `datasets>=4`), so the freeze loads the authors' canonical human-verified JSON `data/with_human_verification.json` (1,118 rows) directly via `hf_hub_download`. Frozen locally to `dataset/morehopqa.jsonl` (SHA256 `f387456a…` in manifest, built 2026-07-03, `datasets` 5.0.0).

Used split:
- No named split — the single human-verified file (1,118 questions) is the source pool.

Closed-book QA olarak mı kullanıldı?
- Evet / Hayır: Evet — the `context` field is deliberately NOT stored at freeze time (`adapt_morehopqa` docstring: "Closed-book: `context` deliberately NOT stored").

Number of evaluated questions:
- 1000 (of 1,118 available; all 16 MoreHopQA result files — 8B and Mistral — contain exactly 1000 detail records).

Sample selection:
- Frozen preprocessed file (same seed-42 deterministic shuffle + unique-question selection; 118 source rows simply not selected).

Selection seed:
- 42

Reference answer format:
- Single answer (last-hop answer; every frozen row has exactly 1 gold string).

Any preprocessing?
- Same normalization pipeline as the other datasets. Additionally, multi-hop metadata is preserved in `meta`: `previous_question`, `previous_answer`, `answer_type`, `previous_answer_type`, `no_of_hops`, `reasoning_type`, `question_decomposition`.

Any excluded/failed examples?
- None among the evaluated 1000; 0 error records in all 16 result files. (118 of the 1,118 source rows fall outside the frozen selection.)

Notes for thesis:
- Priority multi-hop setting — the only dataset where C4's step-wise signal is expected to matter. `meta.no_of_hops` and `meta.reasoning_type` (Symbolic/Arithmetic/Commonsense) enable per-hop-count and per-reasoning-type breakdowns. Only dataset with committed results for both generator models.


---

## 2. Model Information

### Generator Models

Main model:
- Full HuggingFace ID: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Short name used in thesis: Llama-3.1-8B-Instruct (file tag: `8b`, legacy TriviaQA tag: `llama`)
- Role: main
- (Loaded in bfloat16, `PRECISION = "bf16"`, `device_map="auto"`; a 4-bit NF4 path exists in code for Colab dev but full runs use bf16.)

Secondary model:
- Full HuggingFace ID: `mistralai/Mistral-7B-Instruct-v0.3` (gated — requires `HF_TOKEN`)
- Short name used in thesis: Mistral-7B-Instruct-v0.3
- Role: robustness
- (Committed results: MoreHopQA only; TriviaQA/NQ-Open Mistral runs completed on cluster, not yet in repo.)

70B model:
- Was it actually run? Hayır — no result files, no sbatch script invokes it; exists only as the `70b` alias in `MODEL_MAP` (`meta-llama/Meta-Llama-3-70B-Instruct`).
- If not, should it be mentioned? Belirsiz — a MoreHopQA-only 70B run is still a possibility; decide when Chapter 5 is finalized.
- Note: If it stays unrun, either omit entirely or mention as future work / compute limitation.


### NLI Model for Black-Box KLE

NLI model:
- Full HuggingFace ID: `microsoft/deberta-v2-xlarge-mnli`
- Used for:
  - [ ] semantic equivalence clustering
  - [ ] entailment graph construction
  - [x] both
- Notes: Loaded in fp16 (`torch.float16`) — v2-xlarge is fp16-safe (unlike DeBERTa-v1). Clustering (`get_semantic_ids`) uses **strict** bidirectional entailment (`STRICT_ENTAILMENT = True`: both directions must predict class 2 = entailment). Graph construction (`get_entailment_graph`) is weighted with `weight_strategy="manual"`: edge weight = 1(fwd entail) + 1(bwd entail) + 0.5·1(fwd neutral) + 0.5·1(bwd neutral). Pairwise results are cached per (text1, text2); inputs truncated at 512 tokens. Same NLI model and settings in black-box AND white-box pipelines (used for clustering/graph in both).


### Judge Model

Judge model:
- Same as generator? Evet — self-judge: every pipeline calls `llm_judge(gen_tok, gen_model, …)`, so the judge is Llama-3.1-8B for 8B runs and Mistral-7B for Mistral runs.
- If separate, model name: — (a fixed judge, Llama-3.1-8B-Instruct for all generators, is planned via `rejudge.py`; see below)
- Judge temperature: The judge call itself is **greedy** (`do_sample=False`, `max_new_tokens=8`; answer counted correct iff output starts with "yes"). `JUDGE_TEMP = 0.1` is the temperature of the **low-temperature candidate generation** that produces the answer being judged (`low_temp_sample`, `top_p=1.0`, `top_k=0`).
- Was rejudge.py used? Planned but not final — spec exists in `todo/judge-task.md` (fixed judge = Llama-3.1-8B-Instruct, identical prompt, labels renamed `judge_correct_selfjudge` → new `judge_correct`), but `rejudge.py` is not written and no rejudged output files exist. `analysis/cross_judge.py` (cross-model re-labeling + Cohen's κ) exists but has produced no outputs in `results/` yet.
- Final labels come from which files/results? Current final labels are the self-judge labels stored inside the pipeline outputs: `details[].judge_correct` (C1–C3; hallucination = `not judge_correct`) and `details[].is_halluc` (C4) in `results/{black-box,white-box}/<dataset>/c{1-4}_<dataset>_<tag>_{BB,WB}_results.json` (TriviaQA: legacy `*_llama_{bb,wb}.json` names).
- Notes: Judge prompt follows Nikitin et al. Sec. 5 verbatim ("We are assessing the quality of answers… Respond only with yes or no."), with gold answers joined by "; ". What is judged per condition: C1/C2 = full low-temp candidate; C3 = `extract_final_answer(candidate)` (text after the last "Final Answer:" marker, else second-to-last sentence); C4 = answer span extracted from the reference chain. Self-judging does not affect within-model comparisons (C1–C4, BB vs WB — same judge throughout) but does affect cross-model comparisons; this is the motivation for the planned fixed-judge re-labeling.


---

## 3. Experimental Conditions

### C1 – Standard Prompting

Included in Chapter 5?
- Evet (canonical KLE baseline)

Used for:
- [ ] White-box
- [ ] Black-box
- [x] Both

Prompt text:
```text
System: Answer the following question in a single brief but complete sentence.
User:   {question}
```
(Applied via `tokenizer.apply_chat_template(..., add_generation_prompt=True)`; identical string in all 6 C1 pipelines — black-box and white-box × 3 datasets. Sampling for uncertainty estimation: `N_SAMPLES = 10`, `temperature = 1.0`, `top_p = 0.9`, `top_k = 50`, `max_new_tokens = 64`; plus one low-temperature candidate at T=0.1 for judging.)

Uncertainty score:
- Black-box: full KLE (`kle_full`) on the 10 sampled answers — strict-entailment semantic clustering + weighted entailment graph → `heat_kernel(t=0.3)`, normalized, mixed with the block-diagonal semantic-likelihood kernel at `alpha=0.5` → `kle.core.vn_entropy(normalize=False)`. Log-likelihoods = mean transition score per sample, cluster-aggregated with `sum_normalized`.
- White-box: hidden-state kernel-VNE (`vne`) — layer-16 hidden states of each sampled answer, `POOL="mean"` over answer tokens, cosine Gram kernel → `kle.core.vn_entropy(normalize=True)`. Same generation settings as black-box C1.

Judged answer:
- Full low-temperature candidate (T=0.1), no extraction.

Notes:
- Output files: `c1_<dataset>_<tag>_{BB,WB}_results.json` (TriviaQA: legacy `c1_triviaqa_llama_{bb,wb}.json`).


### C2 – Holistic CoT

Included in Chapter 5?
- Evet (holistic CoT condition; expected-weaker by design — reasoning-text diversity inflates cluster counts and dilutes the KLE signal)

Used for:
- [ ] White-box
- [ ] Black-box
- [x] Both

Prompt text:
```text
System: You are a helpful assistant. Reason step by step, but keep your
        reasoning concise (under 3 sentences). Always conclude with your
        final answer.
User:   {question}

        Let's think step by step.
```
(`COT_TRIGGER = "Let's think step by step."` (Kojima et al.) appended to the user turn. Dataset difference: MoreHopQA uses "under 6 sentences" instead of "under 3 sentences". Same prompt in black-box and white-box.)

Sampling:
- `N_SAMPLES = 10`, `temperature = 1.0`, `top_p = 0.9`, `top_k = 50`; `max_new_tokens = 200` (TriviaQA, NQ-Open) / `350` (MoreHopQA) — raised from C1's 64 so the reasoning chain is not truncated. Low-temp candidate at T=0.1 with the same token budget.

Uncertainty score:
- Input = the **entire generated CoT chain** as one unit (holistic; final answer NOT extracted).
- Black-box: identical KLE machinery to C1 (strict-entailment clustering + weighted graph + heat kernel + likelihood mixture), but NLI compares full chains.
- White-box: identical hidden-state VNE machinery to white-box C1 (layer 16, `POOL="mean"`, cosine kernel), pooled over the **full chain span**. Generation is batched (`num_return_sequences=10`); hidden states read via separate forward passes (`FORWARD_BS=8`), `n_fwd_tokens` logged.

Judged answer:
- Full low-temperature CoT chain (no extraction) — judged as-is.

Notes:
- C2 is C1 with exactly one change: the prompting layer (CoT prompt + larger token budget). KLE math, clustering, judge, and metrics are unchanged — stated explicitly in the module docstrings.
- Output files: `c2_<dataset>_<tag>_{BB,WB}_results.json`.


### C3 – CoT with Final-Answer Extraction

Included in Chapter 5?
- Evet (best black-box detector across datasets)

Used for:
- [ ] White-box
- [ ] Black-box
- [x] Both

Prompt text:
```text
System: You are a helpful assistant. Reason step by step, but keep your
        reasoning concise (under 3 sentences). You MUST conclude your
        response with the exact phrase 'Final Answer:' followed by your
        actual answer.
User:   {question}

        Let's think step by step.
```
(Dataset difference: MoreHopQA uses "under 6 sentences". Same prompt in black-box and white-box.)

Sampling:
- Identical to C2: `N_SAMPLES = 10`, T=1.0, top_p=0.9, top_k=50, `max_new_tokens = 200` (TriviaQA, NQ-Open) / `350` (MoreHopQA); low-temp candidate at T=0.1.

Answer-span extraction (`extract_final_answer`):
- Take the text after the **last** occurrence of the marker `"final answer:"` (case-insensitive). Fallback if the model omits the marker: second-to-last sentence (split on "."), else the full text.

Uncertainty score:
- Input = the **extracted final answers** of the 10 sampled chains (reasoning text discarded).
- Black-box: identical KLE machinery to C1/C2 over the extracted answers; log-likelihoods are still those of the full sampled chains (mean transition score per sample).
- White-box: layer-16 hidden states **mean-pooled over the final-answer token span only**, cosine kernel → VNE. `POOL` MUST be `"mean"` here — `"last"` would read the same last token as C2 and collapse C3 into C2 (documented in the module docstring; `judge_extracted` also stored in WB details).

Judged answer:
- `extract_final_answer(candidate)` — the extracted short answer from the low-temp chain, not the full chain.

Notes:
- Sampling/generation is identical to C2; C3 differs only in the "Final Answer:" format instruction and in what enters the uncertainty estimator (and the judge).
- Output files: `c3_<dataset>_<tag>_{BB,WB}_results.json` (TriviaQA black-box legacy file has the `triviaiqa` filename typo).


### C4 – Step-wise Kernel-VNE (own contribution)

Included in Chapter 5?
- Evet — own contribution, not in the official KLE repo. Terminology rule: always "hidden-state kernel-VNE variant of KLE" (white-box) / step-wise KLE (black-box), never plain "KLE".

Used for:
- [ ] White-box
- [ ] Black-box
- [x] Both (white-box = main variant with attention aggregation; black-box = RQ2 variant with attention aggregation removed)

Prompt text:
```text
System: You are a helpful assistant. Reason step by step. Use at most
        3 concise steps. Format your reasoning as explicit numbered steps,
        each on its own line starting with 'Step 1:', 'Step 2:', and so on.
        After the steps, give the final answer on a new line starting
        with 'Answer:'.
User:   {question}

        Let's think step by step.
```
(Dataset difference: MoreHopQA uses "at most 5 concise steps". The explicit step-delimiter format is what makes step segmentation operational; without reliable 'Step k:' markers C4 degenerates to a single step ≈ C2. Same prompt in black-box and white-box.)

Procedure (both box types):
1. **Reference chain**: one low-temperature generation (`REF_TEMP = 0.1`, `max_ref_tokens = 384`; MoreHopQA: 512). Plays the role of the judged low-temp candidate of C1–C3.
2. **Step segmentation**: regex on `Step k:` labels up to the `Answer:`/`Final Answer:` line (`STEP_RE`, `ANSWER_RE`); if no step labels found, the whole chain is one step; capped at `MAX_STEPS = 3` (MoreHopQA: 5).
3. **Per-step resampling**: for each step, condition on `prompt + chain[:label_end]` and sample `N_SAMPLES = 10` continuations (T=1.0, top_p=0.9, top_k=50, `max_step_tokens = 80`, `min_new_tokens = 4`), each truncated in text space before the next `Step k:`/`Answer:` delimiter (prevents the shared-delimiter-token single-cluster collapse).
4. **Per-step score**:
   - Black-box: full KLU score per step — strict-entailment clustering + weighted entailment graph + heat-kernel/likelihood mixture (same `HEAT_T=0.3`, `ALPHA=0.5` as C1–C3) over the 10 step continuations.
   - White-box: layer-16 hidden states of each continuation (`STEP_POOL = "last"` content token), cosine Gram kernel → `vn_entropy(normalize=True)` per step.
5. **Aggregation across steps**:
   - Black-box: `max` and `mean` only — attention aggregation deliberately REMOVED (white-box signal leakage guard for RQ2). `AGG_PRIMARY = "max"`.
   - White-box: `max`, `mean`, and `attn_mean` (`DO_ATTN_AGG = True`): attention mass flowing from the final-answer tokens to each step's tokens (last layer, head-averaged, eager attention) as step weights; failsafe → uniform mean. `AGG_PRIMARY = "max"`.

Judged answer:
- `extract_final_answer(chain, answer_span)` — text after ":" in the `Answer:` span of the reference chain (fallback: whole chain). Label stored as `details[].is_halluc`.

Notes:
- Only condition with resumable JSONL checkpointing (`c4_<dataset>_<tag>_{BB,WB}_checkpoint.jsonl`, keyed on `q_idx`; checkpoint must be deleted when config changes). Per-question error handling exists (`error` field), but all committed runs have 0 error records.
- Cost instrumentation: exact `gen_tokens` (and `fwd_tokens` in WB) counters — the basis of the token-based cross-condition compute comparison.
- Single-hop caveat: on TriviaQA ~995/1000 questions cap at ~3 steps, so the step-wise signal is weak on single-hop; MoreHopQA (up to 5 steps) is the setting where C4 is meaningful.
- Output files: `c4_<dataset>_<tag>_{BB,WB}_results.json`; AUROC reported per aggregation key (BB: max/mean; WB: max/mean/attn_mean).