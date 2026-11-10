# -*- coding: utf-8 -*-
"""Evaluation utilities: token accuracy, per-word breakdown, McNemar's test.

The whole pipeline is evaluated per annotated token: every occurrence of an
ambiguous word in a teacher answer key has exactly one correct KBBI sense
(annotated manually against the online KBBI), and a prediction is either
right or wrong.
"""
import pandas as pd
from scipy.stats import binomtest


def accuracy(pred, gold):
    """Return ``(correct, n, accuracy_pct)`` for two aligned lists."""
    assert len(pred) == len(gold), "prediction/gold length mismatch"
    n = len(gold)
    correct = sum(1 for p, g in zip(pred, gold) if p == g)
    return correct, n, round(correct / n * 100, 1)


def per_word_accuracy(df, pred_col, gold_col="sense_true", word_col="kata_token"):
    """Token accuracy grouped by ambiguous word, best-coverage words first."""
    rows = []
    for word, sub in df.groupby(word_col):
        correct = int((sub[pred_col] == sub[gold_col]).sum())
        rows.append({
            word_col: word,
            "tokens": len(sub),
            "correct": correct,
            "accuracy_pct": round(correct / len(sub) * 100, 1),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["tokens", "accuracy_pct"], ascending=[False, False]).reset_index(drop=True)


def contingency_table(pred_a, pred_b, gold):
    """2x2 contingency table of two systems on the same tokens.

    Returns ``{"both_correct", "a_only_correct", "b_only_correct", "both_wrong"}``.
    """
    both_c = a_only = b_only = both_w = 0
    for pa, pb, g in zip(pred_a, pred_b, gold):
        pa_ok, pb_ok = pa == g, pb == g
        if pa_ok and pb_ok:
            both_c += 1
        elif pa_ok:
            a_only += 1
        elif pb_ok:
            b_only += 1
        else:
            both_w += 1
    return {"both_correct": both_c, "a_only_correct": a_only,
            "b_only_correct": b_only, "both_wrong": both_w}


def mcnemar(pred_a, pred_b, gold):
    """McNemar's test on paired binary outcomes (correct/incorrect).

    Only the *discordant* pairs matter: b = tokens system A got right but
    system B got wrong, c = the reverse. Exact binomial test on b vs c
    (two-sided, p = 0.5) is the small-sample-safe statistic; the chi-square
    approximation ``(|b - c| - 1)^2 / (b + c)`` with 1 degree of freedom is
    reported alongside.

    Returns ``{"b", "c", "n_discordant", "chi2", "p_exact"}``.
    """
    b = sum(1 for pa, pb, g in zip(pred_a, pred_b, gold) if pa == g and pb != g)
    c = sum(1 for pa, pb, g in zip(pred_a, pred_b, gold) if pa != g and pb == g)
    p_exact = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if b + c else 0.0
    return {"b": int(b), "c": int(c), "n_discordant": int(b + c),
            "chi2": round(chi2, 2), "p_exact": float(f"{p_exact:.3e}")}
