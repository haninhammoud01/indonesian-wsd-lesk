# -*- coding: utf-8 -*-
"""Interactive demo of the simplified Lesk WSD pipeline.

Runs fully offline on the published sample gloss KB (data/gloss_kb_sample.csv,
KBBI definitions only - no student data). For every example it prints the
context words, the overlap score of each KBBI sense, and the chosen sense.

Usage:
    python demo/demo.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indonesian_wsd.lesk import sense_scores, simplified_lesk  # noqa: E402
from indonesian_wsd.preprocessing import build_gloss_kb_from_csv, load_stopwords  # noqa: E402

KB_PATH = os.path.join(ROOT, "data", "gloss_kb_sample.csv")

# (target word, sentence) - sentences written for this demo in the same
# register as the research data. The second "sel" example deliberately uses
# a non-biology context to show the same word resolving to a different
# KBBI sense purely because of its context.
EXAMPLES = [
    ("sel",
     "Sel merupakan bagian terkecil penyusun organisme yang terdiri atas "
     "inti, protoplasma, dan dikelilingi membran."),
    ("sel",
     "Sel surya adalah alat penghasil tenaga listrik melalui reaksi kimia, "
     "contohnya pada baterai kendaraan listrik."),
    ("air",
     "Air adalah cairan jernih tanpa warna dan tanpa rasa yang sangat "
     "diperlukan dalam kehidupan manusia, hewan, dan tumbuhan."),
]


def main():
    gloss_kb = build_gloss_kb_from_csv(KB_PATH)
    stopwords = load_stopwords()

    print(f"KBBI gloss KB : {len(gloss_kb)} words, "
          f"{sum(len(g) for g in gloss_kb.values())} sense definitions (sample)")
    print(f"stopwords     : {len(stopwords)} PySastrawi function words")
    print("=" * 72)

    for target, sentence in EXAMPLES:
        scores, context = sense_scores(sentence, target, gloss_kb, stopwords)
        chosen = simplified_lesk(sentence, target, gloss_kb, stopwords)

        print(f"\nTarget word : '{target}'")
        print(f"Sentence    : {sentence}")
        print(f"Context     : {context}")
        for i, (score, gloss) in enumerate(zip(scores, gloss_kb[target]), 1):
            marker = " <-- chosen" if i == chosen else ""
            print(f"  sense {i} | overlap = {score}{marker}")
            print(f"          gloss: {gloss[:80]}{'...' if len(gloss) > 80 else ''}")
        print(f"  -> simplified Lesk picks sense {chosen}")

    print("=" * 72)
    print("On the full 646-token annotated evaluation this pipeline reached")
    print("63.8% token accuracy and beat the IndoSBERT baseline (53.7%).")
    print("Reproduce the statistics:  python experiments/reproduce_results.py")


if __name__ == "__main__":
    main()
