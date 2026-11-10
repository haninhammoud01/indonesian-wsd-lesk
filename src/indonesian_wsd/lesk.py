# -*- coding: utf-8 -*-
"""Simplified Lesk algorithm over KBBI glosses.

The idea (Lesk, 1986, simplified variant): a word's correct sense is the
gloss (dictionary definition) that shares the most words with the context
the word appears in.

    score(sense) = | context_words ∩ words(gloss_sense) |
    predicted sense = argmax score   (ties / zero scores -> sense 1)

Config used for the reported results:

- context  = unique content words of the sentence (PySastrawi stopwords removed),
- gloss    = KBBI definitions, tokenized by plain whitespace split,
- no stemming, no IDF weighting (both were tested and rejected, see README).
"""
from .preprocessing import context_words


def sense_scores(sentence, target_word, gloss_kb, stopwords):
    """Return ``(scores, context)`` for every sense of ``target_word``.

    ``scores[i]`` is the context-gloss overlap count for sense ``i+1``.
    Exposed separately so demos can inspect how the decision was made.
    """
    context = context_words(sentence, stopwords, target_word)
    glosses = gloss_kb.get(target_word)
    if not glosses:
        return None, context
    scores = [
        sum(1 for w in context if w != target_word and w in gloss.split())
        for gloss in glosses
    ]
    return scores, context


def simplified_lesk(sentence, target_word, gloss_kb, stopwords):
    """Predict the KBBI sense number (1-based) for ``target_word``.

    Returns ``None`` when the word is not in the gloss KB. Ties and
    all-zero scores resolve to sense 1 (the first KBBI meaning), which
    acts as a sensible default for biology essays.
    """
    scores, _ = sense_scores(sentence, target_word, gloss_kb, stopwords)
    if scores is None:
        return None
    return scores.index(max(scores)) + 1


def simplified_lesk_idf(sentence, target_word, gloss_kb, stopwords, idf):
    """IDF-weighted variant of the Lesk score (tested, then rejected).

    ``idf`` maps a context word to its Inverse Document Frequency, computed
    over the gloss corpus (each sense definition = one document). Overlaps
    are summed as IDF weights instead of raw counts; unknown words get 0.
    On the evaluation data this variant scored 63.6% vs 63.8% for the plain
    count version - McNemar p = 1.0 - so IDF was not kept (see README).
    """
    context = context_words(sentence, stopwords, target_word)
    glosses = gloss_kb.get(target_word)
    if not glosses:
        return None
    scores = [
        sum(idf.get(w, 0.0) for w in context if w != target_word and w in set(gloss.split()))
        for gloss in glosses
    ]
    return scores.index(max(scores)) + 1


def build_idf(gloss_kb):
    """Compute IDF over the gloss corpus (document = one sense definition).

    Returns ``dict[str, float]`` with ``idf(w) = ln(N / df(w))`` where N is
    the total number of gloss documents and df(w) the number of glosses
    containing word w.
    """
    import math

    total_docs = sum(len(glosses) for glosses in gloss_kb.values())
    df_counts = {}
    for glosses in gloss_kb.values():
        for gloss in glosses:
            for w in set(gloss.split()):
                df_counts[w] = df_counts.get(w, 0) + 1
    return {w: math.log(total_docs / d) for w, d in df_counts.items()}
