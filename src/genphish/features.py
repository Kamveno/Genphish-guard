"""S2 - stylometric feature extraction.

Turns a raw email body into the 17-dimensional stylometric vector defined in
config.FEATURES. The Zenodo corpus already ships these columns; this module
lets the same pipeline also ingest *new* raw emails (e.g. phishing minted
locally from Ollama models for extra generators), so the transfer matrix can
be widened without changing anything downstream.
"""
from __future__ import annotations
import re
import math
import pandas as pd
from .config import FEATURES

_WORD = re.compile(r"[A-Za-z']+")
_SENT = re.compile(r"[.!?]+")
_URL = re.compile(r"https?://|www\.|\b[\w.-]+\.(?:com|net|org|io|ru|cn|info|xyz|link)\b", re.I)

# small, dependency-free stopword / function-word / pronoun lists
_STOP = set("the a an and or but if then of to in on at for with by from as is are was were be been being this that these those it its".split())
_FUNC = set("the a an and or but if then of to in on at for with by from as is are was were be been being not no do does did will would can could should may might must have has had".split())
_PRON = set("i me my mine we us our ours you your yours he him his she her hers they them their theirs".split())
_URGENCY = set("urgent immediately now verify suspended account password confirm click security alert expire expires limited action required warning locked unauthorized".split())


def _syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def extract(text: str) -> dict:
    """Return the 17 stylometric features for one email body."""
    text = text or ""
    chars = len(text)
    words = _WORD.findall(text)
    wl = [len(w) for w in words]
    nwords = len(words) or 1
    sents = [s for s in _SENT.split(text) if s.strip()]
    nsent = len(sents) or 1
    lower = [w.lower() for w in words]
    uniq = len(set(lower))
    syll = sum(_syllables(w) for w in words)

    f = {
        "char_count": float(chars),
        "word_count": float(len(words)),
        "sentence_count": float(len(sents)),
        "avg_word_len": sum(wl) / nwords,
        "avg_sentence_len": len(words) / nsent,
        "type_token_ratio": uniq / nwords,
        "uppercase_ratio": sum(c.isupper() for c in text) / (chars or 1),
        "digit_ratio": sum(c.isdigit() for c in text) / (chars or 1),
        "punct_ratio": sum(not c.isalnum() and not c.isspace() for c in text) / (chars or 1),
        "exclamation_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "url_count": float(len(_URL.findall(text))),
        "stopword_ratio": sum(w in _STOP for w in lower) / nwords,
        "function_word_ratio": sum(w in _FUNC for w in lower) / nwords,
        "personal_pronoun_ratio": sum(w in _PRON for w in lower) / nwords,
        "urgency_term_count": float(sum(w in _URGENCY for w in lower)),
        # Flesch reading ease
        "flesch_reading_ease": 206.835 - 1.015 * (len(words) / nsent) - 84.6 * (syll / nwords),
    }
    return {k: f[k] for k in FEATURES}


def extract_frame(texts: list[str]) -> pd.DataFrame:
    """Vectorise a list of raw email bodies into a feature DataFrame."""
    return pd.DataFrame([extract(t) for t in texts], columns=FEATURES)
