"""Make the src-layout package importable without installation and share
session fixtures (sample gloss KB + stopwords) across the test suite."""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

import pytest  # noqa: E402

from indonesian_wsd.preprocessing import build_gloss_kb_from_csv, load_stopwords  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def gloss_kb():
    return build_gloss_kb_from_csv(os.path.join(ROOT, "data", "gloss_kb_sample.csv"))


@pytest.fixture(scope="session")
def stopwords():
    return load_stopwords()
