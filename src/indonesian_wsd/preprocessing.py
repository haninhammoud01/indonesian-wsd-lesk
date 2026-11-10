# -*- coding: utf-8 -*-
"""Preprocessing utilities for the Indonesian WSD pipeline.

Faithful to the final notebook configuration:

- no stemming (Sastrawi stemming hurt accuracy, see README "Results"),
- lowercase tokenization that keeps only ``[a-z]+`` tokens,
- stopword filtering with the PySastrawi stopword list (809 words),
- gloss parsing for the KBBI sense dictionary ("Daftar_Gloss" column).
"""
import re

import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


def load_stopwords():
    """Return the PySastrawi stopword set (809 Indonesian function words)."""
    return set(StopWordRemoverFactory().get_stop_words())


def tokenize(text):
    """Lowercase a text and keep alphabetic tokens only.

    >>> tokenize("Membran sel (semipermeabel) mengatur keluar-masuknya zat.")
    ['membran', 'sel', 'semipermeabel', 'mengatur', 'keluar', 'masuknya', 'zat']
    """
    return re.findall(r"[a-z]+", str(text).lower())


def context_words(text, stopwords, target_word):
    """Unique content words of ``text`` (order preserved, stopwords removed).

    The target word itself is NOT removed here - the Lesk scorer skips it
    during scoring, which mirrors the notebook implementation.
    """
    return [w for w in dict.fromkeys(tokenize(text)) if w not in stopwords]


def parse_glosses(raw_gloss):
    """Split one "Daftar_Gloss" cell into a list of sense definitions.

    Raw format example::

        "1) arah; haluan | 2) yang dituju; maksud"

    Returns ``["arah; haluan", "yang dituju; maksud"]``.
    """
    parts = re.split(r"\s*\|\s*(?=\d+\)\s)", str(raw_gloss))
    return [re.sub(r"^\d+\)\s*", "", p).strip() for p in parts if p.strip()]


def build_gloss_kb(df, word_col="kata_token", gloss_col="Daftar_Gloss"):
    """Build the KBBI gloss knowledge base from a ground-truth dataframe.

    Returns ``dict[str, list[str]]``: word -> list of sense definitions
    (sense 1, 2, ... following the KBBI numbering).
    """
    kb = {}
    for _, row in df.iterrows():
        kb.setdefault(row[word_col], parse_glosses(row[gloss_col]))
    return kb


def build_gloss_kb_from_csv(path):
    """Build a gloss KB from the published sample CSV (one row per sense).

    Expected columns: ``kata_token, sense_id, gloss_kbbi``.
    """
    df = pd.read_csv(path)
    kb = {}
    for word, group in df.sort_values("sense_id").groupby("kata_token"):
        kb[word] = group["gloss_kbbi"].tolist()
    return kb
