# -*- coding: utf-8 -*-
"""Reproduce every headline statistic of the README from published artifacts.

Needs only pandas + scipy (no PySastrawi / no heavy models):

    python experiments/reproduce_results.py

Artifacts used (all in results/):
- predictions_lesk_vs_indosbert.csv : per-token predictions of the final
  Lesk config and the IndoSBERT baseline on the same 646 annotated tokens.
- method_comparison.csv             : correct counts of all 6 configurations.
- per_word_accuracy.csv             : token accuracy for all 273 words.
- mcnemar_results.json              : reference values (recomputed here).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402

from indonesian_wsd.evaluation import accuracy, contingency_table, mcnemar  # noqa: E402

pred = pd.read_csv(os.path.join(RES, "predictions_lesk_vs_indosbert.csv"))
gold = pred["sense_true"].tolist()
lesk_pred = pred["pred_lesk_final"].tolist()
bert_pred = pred["pred_indosbert"].tolist()

print("=" * 72)
print("1. TOKEN ACCURACY ON THE SAME 646 ANNOTATED TOKENS")
print("=" * 72)
print("final Lesk (no stemming + stopword filter) :", accuracy(lesk_pred, gold))
print("IndoSBERT gloss-similarity baseline        :", accuracy(bert_pred, gold))

print()
print("=" * 72)
print("2. CONTINGENCY TABLE (rows = Lesk, columns = IndoSBERT)")
print("=" * 72)
cont = contingency_table(lesk_pred, bert_pred, gold)
print("                    IndoSBERT correct   IndoSBERT wrong   total")
print(f"Lesk correct     :  {cont['both_correct']:<19} {cont['a_only_correct']:<17} "
      f"{cont['both_correct'] + cont['a_only_correct']}")
print(f"Lesk wrong       :  {cont['b_only_correct']:<19} {cont['both_wrong']:<17} "
      f"{cont['b_only_correct'] + cont['both_wrong']}")

print()
print("=" * 72)
print("3. McNEMAR TEST - RECOMPUTED LIVE")
print("=" * 72)
res = mcnemar(lesk_pred, bert_pred, gold)
ref = json.load(open(os.path.join(RES, "mcnemar_results.json")))
print("final Lesk vs IndoSBERT :", res)
assert res == ref["mcnemar"]["final_vs_indosbert"], "does not match stored reference!"
print("-> matches results/mcnemar_results.json (p < 0.001: the gap is not luck)")

print()
print("=" * 72)
print("4. METHOD COMPARISON (results/method_comparison.csv)")
print("=" * 72)
print(pd.read_csv(os.path.join(RES, "method_comparison.csv")).to_string(index=False))

print()
print("=" * 72)
print("5. PER-WORD ACCURACY - TOP 6 AND BOTTOM 4")
print("=" * 72)
per = pd.read_csv(os.path.join(RES, "per_word_accuracy.csv"))
print(per.head(6).to_string(index=False))
print("...")
print(per.tail(4).to_string(index=False))
