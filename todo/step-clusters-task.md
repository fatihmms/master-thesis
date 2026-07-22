# TASK: Black-box C4 `step_clusters` Diagnostic (single-cluster collapse ölçümü)

## Sorun

CLAUDE.md, RQ2 için black-box C4'te `step_clusters` diyagnostiğini (adım başına unique
semantic cluster sayısı) öngörüyor: 10 adım devamı tek kümeye çöküyorsa o adımın KLE
skoru bilgi taşımaz ("single-cluster collapse"). Black-box C4'ün gerçekten sinyal
üretip üretmediği bu sayıyla gösterilecek.

Mevcut kodda (`black-box/*/c4_*_pipeline.py`) durum:

- `get_semantic_ids(step_continuations, ...)` döngü içinde **hesaplanıyor** ama
  checkpoint/sonuç kaydına **yazılmıyor**. Kayıt alanları: `q_idx, qid, question,
  chain, n_steps, step_scores, agg, candidate, is_halluc, error, cost` — küme bilgisi yok.
- Örneklenen adım devamları (`step_continuations`) da kaydedilmiyor (sadece referans
  `chain` var) → eldeki sonuç JSON'larından **post-hoc hesaplamak mümkün değil**.

## Yapılacaklar

1. BB C4 kaydına adım başına diagnostic alanları ekle:
   - `step_clusters`: `[len(set(semantic_ids))  per step]`
   - (opsiyonel, tartışılacak) `step_continuations` — post-hoc analiz kapısını açık tutar,
     dosya boyutunu büyütür
2. Koşu kararı: tam N=1000 re-run mı, küçük N'lik ayrı diagnostic run mı? (tasarım kararı)
   - Dikkat: config değişince C4 checkpoint'i silinmeli (mixed-label resumption riski)
3. Analiz: single-cluster collapse oranı (adımların yüzde kaçı 1 kümeye çöküyor),
   RQ2 metnine BB-C4 vs WB-C4 karşılaştırmasının yanına ekle.
   - Referans: strict entailment single-cluster collapse'ı 57% → ~11%'e düşürmüştü (C1 bağlamında)

## Durum

- [ ] Tasarım onayı (kayıt alanları + run boyutu)
- [ ] Kod değişikliği
- [ ] Run + analiz

## Not

Bu diyagnostik `llm-sources/necessary-info.md` formuna bilerek YAZILMADI — form
tamamen kod-dependent ve mevcut kodda/sonuçlarda bu alan yok. Eklendikten sonra
formun C4 bölümü güncellenmeli.