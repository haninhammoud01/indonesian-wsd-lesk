# Data documentation

## What is published in this repository

| File | Content | Privacy |
|---|---|---|
| `data/gloss_kb_sample.csv` | 54 KBBI sense definitions for 10 ambiguous words (`kata_token, sense_id, gloss_kbbi`) | Safe — dictionary content only |
| `results/predictions_lesk_vs_indosbert.csv` | Per-token sense predictions of the final Lesk config and the IndoSBERT baseline, with the annotated gold sense (646 rows) | Safe — word tokens and sense numbers only |
| `results/method_comparison.csv` | Correct counts of all 6 evaluated configurations | Safe |
| `results/per_word_accuracy.csv` | Token accuracy for all 273 ambiguous word types | Safe |
| `results/mcnemar_results.json` | McNemar statistics + 2×2 contingency table | Safe |
| `results/teacher_student_consistency_top10.csv` | Sense-consistency between teacher and student contexts, top-10 words | Safe |
| `results/idf_examples.csv` | Document frequencies / IDF values in the two corpora | Safe |

## What is NOT published (and why)

The full research dataset contains personally identifiable information and
school-internal material:

- **1,258 student essay answers with student names** — real student data
  from an Indonesian senior high school; publishing is not permitted.
- **Teacher model answers (answer keys)** — school-internal assessment
  material.
- **The full gloss KB (1,083 definitions)** — can be rebuilt from the
  public online KBBI; a 10-word sample (`data/gloss_kb_sample.csv`) is
  included so the pipeline can run offline.

Aggregated, non-identifying statistics derived from the full data are fully
reproducible from `results/` via `experiments/reproduce_results.py`.

## Annotation protocol (gold senses)

1. Start from the teacher model answer of each of the 37 biology question
   points (materi: *sel*, cell biology).
2. Tokenise the answer; keep occurrences of words that have **more than one
   sense in the online KBBI** (KBBI Daring, the official Indonesian
   dictionary published by Badan Pengembangan dan Pembinaan Bahasa).
3. For every token occurrence, read the sentence and manually select the
   KBBI sense number that fits the context → 646 annotated tokens covering
   273 word types.
4. Sense numbers are 1-based and follow the KBBI gloss order stored in the
   gloss KB.

## Requesting the full data

For research purposes, aggregated or de-identified versions of the dataset
may be shared — open a GitHub issue describing your use case.
