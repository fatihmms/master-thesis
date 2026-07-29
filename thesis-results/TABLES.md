# `thesis-results/tables/` — data dictionary

Every CSV in this folder is produced by `thesis-results/result-analysis.ipynb`, which reads the saved experiment outputs under `results/` and computes everything post-hoc. No file here involves new generation, and none of them should be edited by hand — re-run the notebook section instead.

---

## Experiment context (needed to read any of these files)

**Task.** Hallucination detection for LLM question answering. For each question the generator produces an answer, an LLM judge labels it correct or hallucinated, and an *uncertainty score* is computed. A good score is high exactly when the answer is a hallucination; detection quality is measured by ranking metrics (AUROC etc.). The scoring method under study is **Kernel Language Entropy (KLE)** (Nikitin et al., NeurIPS 2024) applied under Chain-of-Thought (CoT) prompting.

**Conditions** — four ways of applying the detector:

| Condition | What is scored |
|---|---|
| **C1** | Direct answer, no CoT. Standard KLE. The canonical baseline. |
| **C2** | Full CoT chain treated as one unit ("holistic"). |
| **C3** | CoT chain with the final answer span extracted first, then scored. |
| **C4** | Step-wise: each reasoning step is scored separately, then aggregated into one number per question. A contribution of this thesis, not part of the original KLE paper. In white-box form it is a *hidden-state kernel-VNE variant of KLE*, not plain KLE. |

**Signal / access method** (column `signal`) — where the uncertainty score comes from:
- `blackbox` — sampled output text only: 10 samples are clustered by semantic equivalence (NLI entailment) and the entropy of that structure is the score. No model internals.
- `whitebox` — model internals: hidden states (layer 16) turned into a kernel, scored by von Neumann entropy.

**C4 aggregation variants.** C4 produces one score per reasoning step, aggregated per question by `max`, `mean`, or (white-box only) `attn_mean` (attention-weighted). **`max` is the thesis-stated primary strategy** and is what unqualified "C4" means in these files; the other variants are always carried explicitly.

**Datasets** (all `N = 1000` questions, seed 42): `triviaqa` and `nqopen` are single-hop short-answer QA; `morehopqa` is multi-hop and is the priority setting for C4, since only there do chains actually have several steps (`max_steps = 5`).

**Models** (column `model`): `Llama-3.1-8B` = Llama-3.1-8B-Instruct, `Mistral-7B` = Mistral-7B-Instruct-v0.3. Files written before 2026-07-26 carry the older tag spelling (`llama31_8b`, `mistral7b_v03`) for the same two models. A Llama-3-70B run exists only as a queued MoreHopQA job and appears in no file here.

**Labels.** All correctness labels come from a single fixed LLM judge (Llama-3.1-8B-Instruct) applied uniformly to every cell. Positive class = **hallucination**, so `hallucination_rate` is the positive rate (prevalence).

**External baselines** (established detectors this work is compared against). All of them score the **C2** generations, so they share C2's samples and labels exactly:
- `SelfCheckGPT-NLI` — contradiction rate between the answer and the other samples. Black-box.
- `LLM-Check` — single-forward-pass detectors; five variants stored (`logit_entropy`, `ppl`, `window_entropy`, `hidden`, `attn`), `logit_entropy` is the strongest and the one plotted. White-box.
- `SemanticEntropy` — entropy over semantic clusters; `se_discrete` uses cluster proportions only (black-box), `se_weighted` is the original likelihood-weighted estimator (needs token likelihoods → white-box).

**Uncertainty conventions shared by all files.** Confidence intervals are 95% nonparametric bootstrap intervals, questions resampled with replacement, `n_boot = 2000`, `bootstrap_seed = 0`, identical settings everywhere so intervals are mutually comparable. In files 1–8 they are **unpaired per-run** intervals: overlapping CIs are not a significance test for a difference between two methods. Files 9 and 10 exist precisely for that question and should be used whenever a claim is about A being better than B.

**Columns that mean the same thing everywhere:** `model`, `dataset`, `signal`, `condition`, `n_questions` (questions the value is computed on, normally 1000), `n_boot`, `bootstrap_seed`, `hallucination_rate` (share of hallucinated answers = prevalence), `source_file` (repo-relative path of the raw result file the row was derived from).

---

## 1. `auroc_scores.csv` — headline AUROC table

**What it shows.** Detection quality of every condition, as AUROC: the probability that a randomly chosen hallucinated answer receives a higher uncertainty score than a randomly chosen correct one. 1.0 = perfect, 0.5 = chance, **below 0.5 = the score is inverted** (more confident when wrong).

**Grain.** One row per model × dataset × condition × signal = 2 × 3 × 4 × 2 = **48 rows**. Produced by notebook §1.

| Column | Meaning |
|---|---|
| `auroc` | The comparable value across conditions. For C1–C3 the single stored AUROC; for C4 the `max` aggregation. |
| `auroc_max`, `auroc_mean`, `auroc_attn_mean` | C4 only (empty for C1–C3): AUROC of each aggregation variant. `attn_mean` exists in white-box only. |

**Reading notes.** AUROC is insensitive to class balance, so cells with very different hallucination rates can still be compared. Comparing `blackbox` against `whitebox` is methodologically valid — the metric does not care where the score came from.

---

## 2. `auroc_scores_ci.csv` — the same AUROCs with bootstrap CIs

**What it shows.** The values of file 1 recomputed from the per-question scores together with a 95% bootstrap interval, so differences between conditions can be judged against sampling noise.

**Grain.** One row per plotted series: model × dataset × signal × {C1, C2, C3, C4:max, C4:mean, (+ C4:attn_mean in white-box)} = **66 rows**. Produced by notebook §2.

| Column | Meaning |
|---|---|
| `condition` | `C1`…`C4`. |
| `aggregation` | C4 only: `max` / `mean` / `attn_mean`; empty for C1–C3. |
| `auroc`, `ci_lo`, `ci_hi` | Point estimate and 95% bootstrap interval. |

**Reading notes.** These intervals are **unpaired**: each series is bootstrapped on its own, so two overlapping intervals do *not* establish that the two conditions are indistinguishable. For any claim of the form "A detects better than B", use `paired_delta_auroc.csv` (file 9), where both series are scored on the same resample and the shared noise cancels.

---

## 3. `baseline_comparison_auroc.csv` — CoT conditions vs. external baselines

**What it shows.** AUROC of the CoT conditions (C2, C3, C4) side by side with the external baselines, under the access method each baseline actually requires. C1 is excluded because every baseline scores the C2 generations, so the CoT conditions are the like-for-like comparison.

**Grain.** One row per model × dataset × signal × series = **102 rows**. Produced by notebook §3.

| Column | Meaning |
|---|---|
| `family` | `kle` = a condition from this thesis, `baseline` = an external detector. |
| `method` | `C2`/`C3`/`C4` or the baseline name. |
| `variant` | C4 aggregation (`max`/`mean`/`attn_mean`), LLM-Check detector, or Semantic Entropy variant; empty where the method has none. |
| `auroc`, `ci_lo`, `ci_hi` | Point estimate and 95% bootstrap interval. |
| `hallucination_rate` | Prevalence of the generations this row scored. Baselines and C2 share generations, so their rates are identical; C3 and C4 generate their own (differ by ~0.02). |
| `in_figure` | `True` for the series drawn in the figures; `False` for table-only rows (the weaker LLM-Check detectors). |

**Reading notes.** Baselines appear only in the panel matching their signal requirement (black-box: SelfCheckGPT-NLI, `se_discrete`; white-box: LLM-Check, `se_weighted`). `hidden` and `attn` from LLM-Check fall below 0.5 in most cells — their raw scores are inverted under the "higher = more likely hallucination" convention and are reported as stored, **without sign flipping**. Semantic Entropy vs. C2 is the cleanest ablation in the file: same samples, same clustering, only the scoring function differs.

---

## 4. `c4_step_diagnostics_morehopqa.csv` — what C4's chains look like

**What it shows.** Structural description of the C4 reasoning chains on MoreHopQA — how long they are, how often the model failed to produce step structure at all, and (black-box only) how diverse the semantic clusters per step are.

**Grain.** One row per model × signal = **4 rows** (MoreHopQA only, since it is the only dataset with real multi-step chains). Produced by notebook §4.

| Column | Meaning |
|---|---|
| `n_total`, `n_valid` | Questions in the run; questions left after dropping format failures. |
| `mean/median/min/max_n_steps` | Chain length in reasoning steps. |
| `cap_hit_rate` | Share of chains that hit the `max_steps = 5` cap, i.e. were cut off. |
| `samples_per_step` | Samples drawn per step to build the step score (10). |
| `format_failure_rate` | Share of chains with no usable step structure (`n_steps <= 1`). |
| `format_failure_halluc_rate` | How often those malformed chains were hallucinations — very high, i.e. a generation failure mode rather than a detection problem. |
| `clean_flagged_rate` | Share of *correctly answered* chains in which the judge still found a broken reasoning hop ("right answer, wrong reasoning"). A ceiling on what answer-level labels can say about reasoning quality. |
| `mean_step_clusters` | Mean number of distinct semantic clusters per step. **Black-box only** (empty in white-box, which stores attention weights instead of clusters). |
| `single_cluster_step_rate` | Share of steps whose samples all collapsed into one cluster, i.e. zero measured uncertainty. Black-box only. |
| `n_seg_mismatch` | Chains where step segmentation disagreed with the pipeline's own; 0 everywhere (sanity check). |

---

## 5. `c4_step_profile_morehopqa.csv` — step uncertainty along the chain

**What it shows.** Mean C4 step score at each position in the chain, split by whether the final answer was hallucinated or correct. This is the evidence for the central C4 finding: hallucinated chains sit **above** correct ones at *every* step with the same profile shape — the signal is a chain-level level shift, not a spike at the step that goes wrong.

**Grain.** One row per model × signal × label × step index (1–5) = **40 rows** (some late-step cells are absent where no chain is that long). Produced by notebook §4. Format failures are excluded.

| Column | Meaning |
|---|---|
| `label` | `hallucinated` or `correct` — the final-answer judgement of the chains averaged in this row. |
| `step_index` | 1-based position in the reasoning chain. |
| `n_chains` | Chains contributing to this cell (falls with step index, since not all chains are 5 steps long). |
| `mean_score`, `sem_score` | Mean step score and its standard error. Black-box scores are semantic-cluster entropies, white-box scores are hidden-state VNE — **different scales, do not compare across `signal`**. |
| `mean_clusters` | Mean distinct semantic clusters at that step. Black-box only. |

---

## 6. `c4_error_localization_morehopqa.csv` — does the step score find the broken step?

**What it shows.** Whether C4's per-step uncertainty points at the reasoning step where the chain *first* goes wrong. The ground truth comes from a separate LLM-judge pass (`analysis/c4_error_localization.py`) that checks each gold reasoning hop against the chain and records the first unsupported one.

**Grain.** One row per model × signal = **4 rows**. Produced by notebook §4.

Because the C4 reference chain is decoded at low temperature with a fixed seed, the black-box and white-box runs contain **identical chains and identical gold error labels** (verified 1000/1000). The two rows per model therefore differ only in the uncertainty signal — a properly paired comparison.

| Column | Meaning |
|---|---|
| `n_valid`, `n_halluc`, `n_located` | Non-format-failure chains; of those, hallucinated ones; of those, the ones where the judge could pin down a first broken hop (the cohort all metrics below are computed on). |
| `loc_rate_in_halluc` | `n_located / n_halluc`. |
| `mean_norm_error_position` | Mean position of the first error as a fraction of chain length (≈0.39: errors are front-loaded, most chains break at step 1). |
| `top1`, `pm1` | Share of chains where the highest-scoring step is exactly / within ±1 of the true first-error step. |
| `auc` | **Within-chain AUC**: probability that the true error step scores higher than a randomly drawn other step of the same chain. Chance = 0.5. The most informative column here — unlike `top1` it is unaffected by chain length. |
| `mrr` | Mean reciprocal rank of the true error step when steps are sorted by descending score. |
| `gap` | Mean score at the error step minus the mean of the other steps of the same chain. |
| `*_chance` | What a uniformly random step pick achieves on the same chains. **Essential**: chains are capped at 5 steps, so `top1_chance` ≈ 0.22 and `pm1_chance` ≈ 0.48 — raw agreement is meaningless without it. |
| `*_lift` | `value − chance`. The honest quantity to quote. |
| `*_ci_lo`, `*_ci_hi` | 95% bootstrap interval of the value, resampling chains. |

**Reading notes.** Black-box step entropy localizes weakly but consistently above chance (`auc` ≈ 0.52–0.56); white-box hidden-state VNE is at or below chance (≈0.48). So step-level uncertainty helps rank *responses* far more than it helps find *which step* broke.

---

## 7. `extra_metrics.csv` — AUPRC, PRR, FPR@95TPR, AUARC

**What it shows.** Four complementary metrics for every series in files 1–3, because AUROC hides both precision at the top of the ranking and the cost of a fixed operating point. Long format: one row per series **per metric**.

**Grain.** 114 series (model × dataset × signal × {C1–C4 variants + all baseline variants}) × 4 metrics = **456 rows**. Produced by notebook §5.

| Metric (`metric`) | Direction | Chance level (`chance`) | Meaning |
|---|---|---|---|
| `AUPRC` | higher better | prevalence | Area under the precision–recall curve for the hallucination class. Depends on the positive rate — read as lift over `chance`, never compare raw values across cells with different rates. |
| `PRR` | higher better | 0 | Prediction rejection ratio: share of an oracle's rejection benefit the score achieves, normalized between random rejection (0) and oracle (1). Negative = worse than random. **The one metric here comparable across every cell.** |
| `FPR@95TPR` | **lower** better | 0.95 | Share of correct answers falsely flagged when the threshold is set to catch 95% of hallucinations. |
| `AUARC` | higher better | base accuracy | Area under the accuracy–rejection curve. Tracks the cell's base accuracy more than the score's quality; `PRR` is its normalized, comparable form. |

| Column | Meaning |
|---|---|
| `family`, `method`, `variant` | Same scheme as file 3; `family = kle` also covers C1 here. |
| `value`, `ci_lo`, `ci_hi` | Point estimate and 95% bootstrap interval. All four metrics of one series share the same bootstrap resamples. |
| `chance`, `higher_is_better` | Metric-specific reference level and direction, carried per row so the file is self-describing. |

**Reading notes.** The metric implementations are hand-written numpy for speed and are cross-checked against scikit-learn at every notebook run (agreement to ~1e-16). Where a `PRR` interval includes 0, the score is no better than rejecting answers at random.

---

## 8. `token_cost.csv` — compute cost per condition

**What it shows.** What each condition costs, counted in tokens per question, with the forward-pass component kept separate from generation. Tokens — not seconds — are the fair cross-condition cost metric here.

**Grain.** One row per model × dataset × signal × condition = **48 rows**. Produced by notebook §6.

| Column | Meaning |
|---|---|
| `mean/median/sd_gen_tokens` | **Generated** tokens: autoregressive sampling (10 samples + the low-temperature candidate; for C4 also every step continuation). The expensive kind. |
| `mean/median_fwd_tokens` | **Forward-pass** tokens: teacher-forced re-encoding for hidden-state/attention extraction. White-box only; 0 in black-box by construction. Parallel over the sequence, so far cheaper per token than generation. |
| `mean_total_tokens` | `gen + fwd`. Prompt tokens are deliberately **not** included. |
| `fwd_share_of_total` | Forward-pass share of the total — 0 for black-box, 0.62–0.78 for white-box. |
| `mean_prompt_tokens` | Prompt length, reported for completeness. Empty for C4, whose logs do not record it — which is why it is excluded from the total. |
| `mean_t_seconds` | Wall-clock per question. **Do not compare across conditions**: C4 batches its samples via `num_return_sequences` while C1–C3 loop sequentially, so this compares pipeline implementations, not methods. |
| `auroc` | The condition's AUROC (C4 = `max`), repeated here so cost and quality can be read from one file. |

**Reading notes.** The cost ladder is C1 ≪ C2 ≈ C3 ≪ C4 on every dataset, and everything roughly doubles on MoreHopQA (longer chains). C3 costs the same as or less than C2 while detecting better — there is no cost argument for preferring C2. The black-box/white-box difference is almost entirely forward-pass load: a white-box bar four times taller is nowhere near four times slower.

---

## 9. `paired_delta_auroc.csv` — is A actually better than B?

**What it shows.** Head-to-head differences in AUROC with a **paired** bootstrap, which is the file to cite for any "A beats B" statement. Every other table reports each series bootstrapped on its own; here both series are recomputed on the *same* resample of question ids in each of the 2000 iterations, so the sampling noise the two share cancels out of the difference. Intervals are therefore typically much narrower than the gap between two unpaired intervals, and contrasts that look inconclusive in files 1–3 can be decisive here.

**Grain.** One row per model × dataset × comparison = **192 rows**. Produced by notebook §7.

**How pairing works.** Two runs are aligned on the question ids they share (`n_paired`), and **each series keeps its own labels**. This is deliberate: C1–C4 each generate their own answers, so the same question can be correct under one condition and hallucinated under another. A row therefore answers "would switching from B to A give me a better detector overall", which bundles a different scoring method with a different answer distribution — the honest framing for the research question, with both regimes visible in `halluc_rate_a` / `halluc_rate_b`.

| Column | Meaning |
|---|---|
| `signal` | The access method both series were taken from, or `cross_signal` for the white-box − black-box rows. |
| `comparison` | Human-readable label, e.g. `C3 - C2` or `C1 (whitebox) - C1 (blackbox)`. A is always the first term. |
| `method_a`, `variant_a`, `method_b`, `variant_b` | The two series, split into method and variant (C4 aggregation, LLM-Check detector, Semantic Entropy variant). |
| `auroc_a`, `auroc_b` | AUROC of each series on the paired subset. |
| `delta` | `auroc_a − auroc_b`. Positive = A is the better detector. |
| `ci_lo`, `ci_hi` | 95% percentile interval of Δ from the paired bootstrap. |
| `p_two_sided` | `2 · min(P(Δ ≤ 0), P(Δ ≥ 0))`, clipped at 1.0. A bootstrap percentile p-value: with B=2000 it cannot resolve below ≈0.001, so 0.000 means "below what this bootstrap can measure". |
| `n_paired` | Questions present in both runs. |
| `halluc_rate_a`, `halluc_rate_b` | Prevalence in each series on the paired subset. Different values mean the two conditions produced differently-hard answer sets. |
| `same_generations` | `True` when both series read one and the same generation run, so their labels are identical question by question (asserted in code, not assumed). See below. |

**Which comparisons are included** (per model × dataset): condition contrasts C3−C2, C4(max)−C2, C4(mean)−C2, C4(max)−C3, C2−C1, C3−C1 in each signal, plus C4(attn_mean)−C2 in white-box; C4 aggregation contrasts max−mean (both signals) and max−attn_mean (white-box); white-box − black-box for each of C1, C2, C3, C4(max); and each of C2, C3, C4(max) against Semantic Entropy, SelfCheckGPT-NLI (black-box) and LLM-Check `logit_entropy` (white-box).

**Reading notes.**
- `same_generations = True` marks the strictly like-for-like rows — same answers, same labels, only the scoring function differs. That covers the C4 aggregation pairs and each baseline against the C2 run it was computed from (SelfCheckGPT-NLI and Semantic Entropy from the **black-box** C2 logs, LLM-Check from the **white-box** one; label agreement is exactly 1.000 in all six cells).
- **The black-box and white-box C2 runs are not the same run**: their labels agree on only 76–92% of questions (lowest on MoreHopQA). So every `cross_signal` row compares two pipelines end to end, not two scores over identical answers. C4 is the exception — both boxes share a deterministically decoded reference chain.
- Semantic Entropy `se_weighted` is listed under white-box because it needs token likelihoods, but it scores the black-box C2 generations; its comparison against white-box C2 is consequently a cross-run comparison (`same_generations = False`), not a same-generation ablation.
- No multiple-comparison correction is applied. With ~29 comparisons per cell, prefer the effect size and its interval over a bare p-value just under 0.05.

---

## 10. `paired_delta_prr.csv` — the same comparisons in PRR

**What it shows.** Identical procedure, comparison set, and columns as file 9, with PRR instead of AUROC (`prr_a`, `prr_b` replace `auroc_a`, `auroc_b`). PRR is included as a second view because it is the only metric in this collection that is comparable across cells with different base accuracies: it normalizes the rejection benefit between random rejection (0) and an oracle (1), so a Δ here is a difference in *usable* rejection quality rather than in pure ranking.

**Grain.** **192 rows**, same keys as file 9. Produced by notebook §7.

**Reading notes.** Where AUROC and PRR disagree in sign or significance for the same comparison, the two metrics are telling different stories: AUROC weighs the whole ranking, PRR weighs what a deployed rejection rule actually recovers on the given base accuracy. Both are worth reporting when they diverge, especially on MoreHopQA where the base accuracy is low.

---

## 11. `hallucination_rate_vs_auroc.csv` — generation quality next to detection quality

**What it shows.** The two properties of a cell side by side: how often the condition hallucinates, and how well its score detects those hallucinations. They are independent axes — a condition can produce the best answers and still be the worst detector — and this file is what lets that be stated with numbers.

**Grain.** One row per model × dataset × signal × condition = **48 rows**. Produced by notebook §8.

| Column | Meaning |
|---|---|
| `condition` | `C1`…`C4`; `aggregation` is `max` for C4 (the primary aggregation) and empty otherwise. |
| `hallucination_rate` | Share of hallucinated answers in **this condition's own generations**. C1–C4 each generate separately, so this differs between conditions in the same cell — that is the point of the file, not an inconsistency. |
| `auroc`, `auroc_ci_lo`, `auroc_ci_hi` | Detection quality on those same generations, with the unpaired 95% bootstrap interval (identical values to file 2). |
| `n_questions` | Questions the pair was computed on. |

**Reading notes.** Black-box and white-box rates for the same condition are close but not identical, because they are separate runs — except C4, where both boxes score the same deterministically decoded chain. For "does condition A detect better than condition B" always use file 9; this file answers the different question of what regime each condition operates in.

---

## 12. `hallucination_rate_vs_auroc_correlations.csv` — the relationship, quantified

**What it shows.** Spearman rank correlations between the two columns of file 11, computed separately at two levels because they point in different directions.

**Grain.** One row per model × signal × scope, with scope ∈ {`overall`, `triviaqa`, `nqopen`, `morehopqa`} = **16 rows**. Produced by notebook §8.

| Column | Meaning |
|---|---|
| `scope` | `overall` = all 12 points of the panel (3 datasets × 4 conditions); a dataset name = the 4 conditions of that dataset only. |
| `n_points` | 12 for `overall`, 4 for a dataset. |
| `spearman_rho` | Rank correlation between hallucination rate and AUROC. |
| `p_value` | Meaningful only for the `overall` rows. **With 4 points a within-dataset p-value carries no information** — read those ρ values as a direction, never as a test. |

**Reading notes.** The `overall` rows are strongly negative and mainly restate task difficulty (MoreHopQA is both hard to answer and hard to audit); they should not be quoted as "models that hallucinate less are easier to audit". The within-dataset rows are the ones relevant to the research question, and their sign is *not* stable across cells: on MoreHopQA the condition with the lowest hallucination rate (C2, holistic CoT) is among the weakest detectors, while C4 has the highest rate and the strongest black-box signal.
