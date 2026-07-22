# TASK: Fixed-Judge Re-labeling (`rejudge.py`)

## Sorun

Mevcut tüm pipeline'larda judge modeli = generator modeli (self-judging):

```python
is_correct = llm_judge(gen_tok, gen_model, question, golds, candidate)
```

Yani judge, generator değiştikçe değişiyor:

| Generator | Kullanılan judge |
|---|---|
| Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct |
| Mistral-7B-Instruct-v0.3 | Mistral-7B-Instruct-v0.3 |
| Llama-3-70B-Instruct | Llama-3-70B-Instruct |

Nikitin et al. (2024) ise judge'ı **sabit** tutuyor (Llama-3-8B-Instruct), generator ne olursa olsun.

## Etki analizi (panik yok)

Judge sadece `z(x)` etiketini üretiyor; `u(x)` belirsizlik skoruna dokunmuyor.

| Karşılaştırma | Etkileniyor mu? |
|---|---|
| C1 vs C2 vs C3 vs C4 (tek model içinde) | HAYIR — hepsi aynı judge |
| White-box vs black-box (tek model içinde) | HAYIR — aynı judge |
| Modeller arası (8B vs Mistral vs 70B) | EVET — judge modelle birlikte değişiyor |

**RQ1 ve RQ2 tek-model-içi sorular → ana katkı etkilenmiyor.**
Etkilenen: modeller arası hallucination rate / AUROC karşılaştırmaları.

## Çözüm

**Generation'ı yeniden çalıştırmaya GEREK YOK.** Sonuç JSON'larında re-judging için gereken her şey zaten var:

- `judge_candidate` — low-temp aday cevap (saklı)
- `id` — frozen dataset'ten `gold_answers` çekmek için
- `question`
- `vne` / `kle` skorları — judge'dan tamamen bağımsız, dokunulmayacak

## Yapılacaklar

### 1. `rejudge.py` yaz

Girdi: `results/` altındaki tüm sonuç JSON'ları (tüm koşullar, tüm modeller, hem BB hem WB).

Adımlar:
1. Sabit judge modelini yükle: **Llama-3.1-8B-Instruct** (bir kez, tüm dosyalar için)
2. Her JSON için, her `details` kaydında:
   - `id` → `load_frozen(dataset)` üzerinden `gold_answers` al
   - `judge_candidate` + `gold_answers` → sabit judge ile yeniden etiketle
   - Judge prompt: mevcut `llm_judge` fonksiyonundakiyle **birebir aynı** (Nikitin et al. Sec. 5 formatı)
   - Greedy: `do_sample=False`, `max_new_tokens=8`, `yes` ile başlıyorsa correct
3. Alan yönetimi:
   - Eski `judge_correct` → `judge_correct_selfjudge` olarak sakla (silme!)
   - Yeni etiket → `judge_correct`
4. Yeni etiketlerle **saklanan skorlardan** AUROC'u yeniden hesapla (`vne` / `kle` alanları değişmiyor)
5. `hallucination_rate`'i güncelle
6. `config`'e ekle: `"judge_model": "meta-llama/Meta-Llama-3.1-8B-Instruct"`, `"judge_mode": "fixed"`

### 2. Robustness analizi (bedava çıktı)

İki etiket setini karşılaştır:
- **Cohen's κ** (self-judge vs fixed-judge), model başına
- AUROC değişimi (Δ), koşul × model başına
- Hallucination rate değişimi

Bu, kusuru titizlik göstergesine çeviriyor.

### 3. Tez metnine ekle (§4.5)

```
The judge model is fixed to Llama-3.1-8B-Instruct across all conditions and
all generator models, so that the correctness labels do not depend on which
model produced the response. The judge prompt follows Nikitin et al. (2024).
Since the labels are produced automatically, judge reliability is verified on
a manually annotated subset and reported as Cohen's kappa, following the
validation practice of prior work (Kuhn et al., 2023; Cheng et al., 2025).
```

Ayrıca (Ablation / Threats of Validity):

```
Labels were originally produced with the generating model as judge. To remove
the dependence of the labels on the generator, all conditions were re-labeled
with a fixed judge model. Agreement between the two labelings is kappa = [X],
and the AUROC values change by at most [Y]. The results reported in this
thesis use the fixed judge.
```

### 4. El etiketli doğrulama (ayrı, sonraki adım)

150–200 soruluk rastgele alt küme, manuel etiketle, fixed-judge ile Cohen's κ raporla.
Emsal: Cheng et al. (2025) → κ = 0.863; Kuhn et al. (2023) → TriviaQA'da otomatik/insan uyumu 0.96.

## Maliyet

Soru başına 8 token'lık greedy judge çağrısı. 1000 soru x 8 koşul → tek GPU'da < 30 dk.
`gpu_h100_short` yeterli. Generation/hidden-state/NLI YOK.

## Dikkat

- Mevcut JSON'ları **üzerine yazma** — `_rejudged.json` suffix'i ile ayrı yaz veya backup al
- 8B çalıştırmalarını da yeniden etiketle: judge'ın "sabit judge olarak bilinçli seçilmiş" olması gerekiyor, tesadüfen aynı model olması değil
- C4'ün JSONL checkpoint'i bu işlemden etkilenmiyor (o generation checkpoint'i, judge değil)