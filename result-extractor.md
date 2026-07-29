# Chapter 6 (Results & Analysis) — Sonuç Üretim Spesifikasyonu

Bu dosya, master tezin Bölüm 6'sı için gereken tüm tablo ve şekillerin üretim
talimatıdır. Amaç: mevcut deney çıktılarından (yeni generation **yapmadan**)
tez tablolarına doğrudan dönüştürülebilir tidy CSV'ler ve figürler üretmek.

---

## 0. Genel kurallar

- **Yeni generation yapılmayacak.** Tüm sayılar kayıtlı deney çıktılarından
  post-hoc hesaplanacak. Eğer bir metrik için gerekli ham veri kayıtlı değilse,
  hesaplamayı uydurma — o satırı `NA` bırak ve `MISSING_DATA.md` dosyasına
  hangi artefaktın eksik olduğunu yaz.
- **Long-format (tidy) CSV.** Her satır tek bir hücre. Pivotlama sonradan
  yapılacak, geniş formatlı tablo üretme.
- **Yuvarlama yok.** Tüm metrikler 6 ondalık basamakla yazılsın; yuvarlama
  tez yazımı aşamasında yapılacak.
- **Sabit seed.** Tüm bootstrap işlemleri için sabit seed kullan ve
  `bootstrap_seed` kolonunda raporla.


### Ortak kolonlar (her CSV'de bulunacak)

| Kolon | Değerler |
|---|---|
| `model` | `llama31_8b` \| `mistral7b_v03` |
| `dataset` | `triviaqa` \| `nqopen` \| `morehopqa` |
| `condition` | `C1` \| `C2` \| `C3` \| `C4` |
| `signal` | `blackbox` \| `whitebox` |
| `n_questions` | int |
| `seed` | generation seed |
| `judge_model` | kullanılan judge modelin tam adı |
| `run_id` | sonuç dosyasının kimliği / yolu |

### Kritik doğrulamalar (rapor edilmesi zorunlu)

1. **Judge tutarlılığı:** Mistral sonuçları **re-judge pipeline'ından geçmiş**
   hâli kullanılacak, eski judge çıktıları değil. Her satırda `judge_model`
   kolonu bunu doğrulasın. Tüm satırlarda aynı judge yoksa hata ver.
2. **N tutarlılığı:** Aynı (model, dataset) için tüm koşullarda `n_questions`
   eşit mi? Eşit değilse paired bootstrap geçersizdir. Bu durumda koşullar
   arası **kesişim kümesi** kullan ve bunu `n_paired` kolonunda ayrıca belirt;
   ayrıca `SANITY_CHECKS.md` dosyasına hangi soruların düştüğünü yaz.
3. **Polarite:** Hiçbir skorun yönü, değerlendirme kümesindeki performansa
   bakılarak çevrilmeyecek. Özellikle LLM-Check hidden-state ve attention
   tabanlı detektörlerinde AUROC 0.5'in altına düşse bile **olduğu gibi**
   raporlanacak. Otomatik sign-flip **yasak**.

### Metrik tanımları

- Pozitif sınıf = **hallucination**.
- `auroc`, `auprc`, `prr`, `fpr95` (FPR@95TPR), `auarc`.
- **Güven aralıkları:** question-level nonparametric bootstrap, **B = 2000**,
  %95 percentile CI.
- **Eşleştirilmiş testler:** paired bootstrap (aynı soru kümesi üzerinde
  yeniden örnekleme), B = 2000, iki yönlü p değeri.

---

## 1. T1 — Ana performans tablosu (master)

**Dosya:** `tables/T1_main_results.csv`
**Kullanıldığı yer:** §6.1 Main Results
**Satır sayısı:** 2 model × 3 dataset × 4 koşul × 2 sinyal = **48**

Ortak kolonlar +:

```
auroc, auroc_ci_lo, auroc_ci_hi,
auprc, auprc_ci_lo, auprc_ci_hi,
prr,   prr_ci_lo,   prr_ci_hi,
fpr95, fpr95_ci_lo, fpr95_ci_hi,
auarc, auarc_ci_lo, auarc_ci_hi,
c4_agg, bootstrap_seed
```

- `c4_agg`: C4 satırları için kullanılan aggregation stratejisi
  (`max` \| `mean` \| `attn_mean`). C1–C3 satırlarında `NA`.
- T1'de C4 için **tek bir birincil aggregation** kullanılacak; diğer ikisi
  T9'da (ablation) raporlanacak. Birincil olarak hangisini seçtiğini
  `README_chapter6.md` içinde açıkça belirt.

---

## 2. T2 — Etiket ve veri betimleyici tablosu

**Dosya:** `tables/T2_label_descriptives.csv`
**Kullanıldığı yer:** §6.1 Main Results
**Satır sayısı:** 2 model × 3 dataset × 4 koşul = **24**
(Etiketler sinyal kaynağından bağımsızdır; `signal` kolonu bu dosyada yok.)

```
model, dataset, condition, n_questions, n_hallucinated, halluc_rate,
mean_response_tokens, median_response_tokens, judge_model, run_id
```

Gerekçe: §5.5'te AUPRC'nin pozitif sınıf prevalence'ına bağlı olduğu
yazılmıştır. Koşullar arası AUPRC farklarının yorumlanabilmesi için
hallucination rate'in tabloda görünmesi zorunludur. Ayrıca C1→C2 geçişinde
correctness dağılımının değiştiği burada gösterilecek.

---

## 3. T3 — Eşleştirilmiş koşul karşılaştırmaları (RQ1)

**Dosya:** `tables/T3_paired_conditions.csv`
**Kullanıldığı yer:** §6.2 Effect of CoT on KLE-Based Detection
**Satır sayısı:** 2 model × 3 dataset × 2 sinyal × 6 çift = **72**

Karşılaştırılacak çiftler (her (model, dataset, signal) için):
`C1–C2`, `C1–C3`, `C1–C4`, `C2–C3`, `C2–C4`, `C3–C4`

```
model, dataset, signal,
cond_a, cond_b,
auroc_a, auroc_b, delta_auroc,
delta_ci_lo, delta_ci_hi,
p_two_sided, p_holm,
n_paired, bootstrap_seed
```

- `delta_auroc = auroc_a - auroc_b` (işaret yönünü README'de sabitle).
- Paired bootstrap, aynı sorular üzerinden, B = 2000.
- `p_holm`: 72 test için Holm–Bonferroni düzeltmesi. Düzeltme, (model, dataset,
  signal) grubu içinde değil, **tüm 72 test üzerinde** uygulansın; ayrıca
  aile bazında düzeltme istenirse diye `p_holm_within_cell` kolonunu da ekle.
- Ham `p_two_sided` her hâlükârda korunacak.

---

## 4. T4 — C4 step-level tanısal istatistikler

**Dosya:** `tables/T4_step_diagnostics.csv`
**Kullanıldığı yer:** §6.2
**Koşul:** Yalnızca C4. Bu veriler pipeline çıktısında kayıtlı değilse üretme,
`MISSING_DATA.md`'ye yaz.

```
model, dataset, signal,
mean_n_steps, median_n_steps, min_n_steps, max_n_steps,
samples_per_step,
empty_step_rate, unparsable_step_rate
```

---

## 5. T5 — Semantik küme tanısı

**Dosya:** `tables/T5_cluster_diagnostics.csv`
**Kullanıldığı yer:** §6.2
**Satır sayısı:** 2 model × 3 dataset × 4 koşul (black-box; küme yapısı orada
tanımlıdır). White-box için küme kavramı yoksa o satırları üretme.

```
model, dataset, condition,
mean_n_clusters, median_n_clusters,
single_cluster_rate,
extraction_fail_rate,
entailment_rule
```

- `single_cluster_rate`: 10 örneklemin tek semantik kümeye düştüğü soruların
  oranı.
- `extraction_fail_rate`: yalnızca C3 için — answer-span çıkarımının başarısız
  olduğu soru oranı. Diğer koşullarda `NA`.
- `entailment_rule`: kullanılan eşdeğerlik kriteri (`strict_bidirectional`
  bekleniyor). Koşullar arasında farklıysa mutlaka belirt.

---

## 6. T6 — Black-box vs White-box eşleştirmeli karşılaştırma (RQ2)

**Dosya:** `tables/T6_blackbox_vs_whitebox.csv`
**Kullanıldığı yer:** §6.3 Black-Box vs White-Box Comparison
**Satır sayısı:** 2 model × 3 dataset × 4 koşul = **24**

```
model, dataset, condition,
auroc_blackbox, auroc_whitebox,
delta_auroc, delta_ci_lo, delta_ci_hi,
p_two_sided,
auprc_blackbox, auprc_whitebox, delta_auprc, delta_auprc_ci_lo, delta_auprc_ci_hi,
n_paired, bootstrap_seed
```

- `delta_auroc = auroc_blackbox - auroc_whitebox`.
- RQ2 "karşılaştırılabilir performans" iddiasını test ettiği için burada
  **CI genişliği p değerinden daha önemlidir**. Delta CI'sını mutlaka üret;
  yalnızca p değeri yeterli değil.

---

## 7. T7 — Baseline karşılaştırma tablosu

**Dosya:** `tables/T7_baselines.csv`
**Kullanıldığı yer:** §6.3
**Satır sayısı:** (8 baseline detektörü + 2 KLE referansı) × 2 model × 3 dataset

`baseline` kolonu değerleri:

| Değer | Sinyal | Kaynak |
|---|---|---|
| `llmcheck_perplexity` | whitebox | LLM-Check |
| `llmcheck_logit_entropy` | whitebox | LLM-Check |
| `llmcheck_window_logit_entropy` | whitebox | LLM-Check |
| `llmcheck_hidden_score` | whitebox | LLM-Check |
| `llmcheck_attention_score` | whitebox | LLM-Check |
| `se_likelihood` | whitebox | likelihood-weighted Semantic Entropy |
| `selfcheckgpt_nli` | blackbox | response-level SelfCheckGPT-NLI |
| `discrete_se` | blackbox | Discrete Semantic Entropy |
| `kle_c2` | her ikisi | referans (KLE, C2) |
| `kle_c4` | her ikisi | referans (KLE, C4) |

Kolonlar: T1 ile aynı metrik seti (AUROC/AUPRC/PRR/FPR@95/AUARC + CI'lar) +
`baseline`, `signal`, `layer` (LLM-Check hidden/attention için `16`, diğerleri
`NA`).

**Uyarılar:**
- Tüm baseline'lar kayıtlı **C2 generation'larından** post-hoc hesaplanacak;
  yeni örnekleme yapılmayacak.
- LLM-Check hidden/attention skorlarında **sign-flip yasak** (bkz. §0.3).
- `se_likelihood` ve `discrete_se`, C2 pipeline'larının kaydettiği semantik
  kümeleri **yeniden hesaplamadan** kullanacak.

---

## 8. T8 — Cross-dataset özet matrisi

**Dosya:** `tables/T8_cross_dataset.csv`
**Kullanıldığı yer:** §6.4 Cross-Dataset Analysis
**Not:** Yeni hesaplama gerekmez, T1'in türevidir.

```
model, signal, condition,
auroc_triviaqa, auroc_nqopen, auroc_morehopqa,
delta_nqopen_vs_triviaqa, delta_morehopqa_vs_triviaqa
```

Amaç: multi-hop (MoreHopQA) ile single-hop (TriviaQA, NQ-Open) arasındaki
farkı tek bakışta göstermek.

---

## 9. T9 — C4 aggregation ablasyonu

**Dosya:** `tables/T9_c4_aggregation_ablation.csv`
**Kullanıldığı yer:** §6.5 Ablation Results
**Satır sayısı:** 2 model × 3 dataset × 2 sinyal × 3 strateji = **36**

```
model, dataset, signal, agg_strategy,
auroc, auroc_ci_lo, auroc_ci_hi,
auprc, auprc_ci_lo, auprc_ci_hi,
prr, prr_ci_lo, prr_ci_hi,
fpr95, fpr95_ci_lo, fpr95_ci_hi,
auarc, auarc_ci_lo, auarc_ci_hi,
delta_vs_primary, delta_ci_lo, delta_ci_hi, p_two_sided,
bootstrap_seed
```

- `agg_strategy` ∈ {`max`, `mean`, `attn_mean`}.
- `delta_vs_primary`: T1'de birincil seçilen stratejiye göre fark
  (paired bootstrap).

> **Layer ablation ÜRETİLMEYECEK.** Katman seçimi bu tezin katkısı değildir
> ve tezde iddia edilmemektedir. Layer-wise tarama yapma.

---

## 10. Şekiller

Tüm figürler hem **PDF** (vektörel, teze gömülecek) hem de figürü üreten
**CSV** ile birlikte kaydedilecek. Dosya adlandırması: `fig6_1_*.pdf` vb.

| ID | İçerik | Bölüm |
|---|---|---|
| **F1** | Gruplu bar chart. x = C1–C4, y = AUROC, hata çubukları = %95 CI, renk = black/white-box, facet = dataset, ayrı figür = model. | §6.1 |
| **F2** | Accuracy–rejection (risk–coverage) eğrileri. C1–C4 aynı panelde, dataset başına facet, model başına ayrı figür. AUARC'ın görselleştirmesi. | §6.2 |
| **F3** | Belirsizlik skoru dağılımı, doğru vs. halüsinasyon ayrımıyla (violin veya üst üste histogram). C1–C4 panelleri. | §6.2 |
| **F4** | Heatmap. Satır = condition × signal, sütun = dataset × model, hücre = AUROC. | §6.4 |

Stil kuralları: grayscale-safe renk paleti, minimum 8pt font, figür genişliği
tez metin genişliğine uygun (tek sütun), başlık figürün içine gömülmesin
(caption tezde yazılacak).

---

## 11. Teslim edilecek çıktılar

```
results/chapter6/
├── README_chapter6.md          # birincil c4_agg seçimi, delta işaret yönü,
│                               # üretim tarihi, commit hash
├── SANITY_CHECKS.md            # N tutarlılığı, judge tutarlılığı sonuçları
├── MISSING_DATA.md             # üretilemeyen tablolar ve nedenleri
├── tables/
│   ├── T1_main_results.csv
│   ├── T2_label_descriptives.csv
│   ├── T3_paired_conditions.csv
│   ├── T4_step_diagnostics.csv
│   ├── T5_cluster_diagnostics.csv
│   ├── T6_blackbox_vs_whitebox.csv
│   ├── T7_baselines.csv
│   ├── T8_cross_dataset.csv
│   └── T9_c4_aggregation_ablation.csv
└── figures/
    ├── fig6_1_auroc_by_condition_{model}.pdf (+ .csv)
    ├── fig6_2_risk_coverage_{model}.pdf (+ .csv)
    ├── fig6_3_score_distributions_{model}.pdf (+ .csv)
    └── fig6_4_auroc_heatmap.pdf (+ .csv)
```

## 12. Öncelik sırası

Hepsini tek seferde bekleme. Şu sırayla üret:

1. **T1 + T2 + F1** → §6.1 yazılabilir hâle gelir
2. **T3 + T5 (+ T4) + F2 + F3** → §6.2
3. **T6 + T7** → §6.3
4. **T8 + F4** → §6.4
5. **T9** → §6.5