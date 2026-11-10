# -*- coding: utf-8 -*-
"""Unit tests for the WSD core: preprocessing, Lesk decisions, McNemar.

The Lesk tests reuse the demo sentences: the same word "sel" must resolve
to different KBBI senses depending on its context.
"""
from indonesian_wsd.evaluation import accuracy, mcnemar
from indonesian_wsd.lesk import simplified_lesk
from indonesian_wsd.preprocessing import parse_glosses, tokenize

BIO_SENTENCE = ("Sel merupakan bagian terkecil penyusun organisme yang terdiri "
                "atas inti, protoplasma, dan dikelilingi membran.")
BATTERY_SENTENCE = ("Sel surya adalah alat penghasil tenaga listrik melalui "
                    "reaksi kimia, contohnya pada baterai kendaraan listrik.")


# ---------- preprocessing ----------

def test_tokenize_keeps_lowercase_alphabetic_tokens():
    assert tokenize("Membran Sel (semi-permeabel) 12 kg.") == \
        ["membran", "sel", "semi", "permeabel", "kg"]


def test_parse_glosses_splits_kbbi_senses():
    raw = "1) arah; haluan (jurusan) | 2) yang dituju; maksud"
    assert parse_glosses(raw) == ["arah; haluan (jurusan)", "yang dituju; maksud"]


# ---------- Lesk decisions ----------

def test_biology_context_selects_sense_1(gloss_kb, stopwords):
    assert simplified_lesk(BIO_SENTENCE, "sel", gloss_kb, stopwords) == 1


def test_battery_context_selects_sense_3(gloss_kb, stopwords):
    assert simplified_lesk(BATTERY_SENTENCE, "sel", gloss_kb, stopwords) == 3


def test_zero_overlap_falls_back_to_sense_1(gloss_kb, stopwords):
    # no content word overlaps any gloss -> default to the first KBBI sense
    assert simplified_lesk("Sel itu mahal sekali harganya.", "sel",
                           gloss_kb, stopwords) == 1


def test_unknown_word_returns_none(gloss_kb, stopwords):
    assert simplified_lesk("Kalimat biasa saja.", "zzztidakada", gloss_kb, stopwords) is None


# ---------- evaluation ----------

def test_accuracy_simple_case():
    gold = [1, 2, 2, 1]
    pred = [1, 1, 2, 1]
    assert accuracy(pred, gold) == (3, 4, 75.0)


def test_mcnemar_known_values():
    # 6 tokens: A correct on all, B wrong on the last two -> b=2, c=0
    gold = [1] * 6
    pred_a = [1] * 6
    pred_b = [1, 1, 1, 1, 0, 0]
    res = mcnemar(pred_a, pred_b, gold)
    assert res["b"] == 2 and res["c"] == 0 and res["n_discordant"] == 2
    assert res["chi2"] == 0.5 and res["p_exact"] == 0.5
