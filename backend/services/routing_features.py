"""Versioned training/inference features; never alters the urgency input."""
import re
from pathlib import Path
from threading import RLock

PREPROCESSING_VERSION = "wordnet-lesk-v1"
AMBIGUOUS_WORDS = frozenset({"bank", "drain", "main", "line"})
_WORDNET_LOCK = RLock()


def tokens(text):
    return re.findall(r"[a-z]+", text.lower())[:512]


def extract_features(text):
    """Original tokens plus contextual noun synsets and their lemma features.

    Lesk selects WordNet senses, never department labels. Without gloss overlap
    abstain from WSD rather than adding an arbitrary zero-overlap tied sense.
    """
    import nltk
    from nltk.corpus import wordnet
    from nltk.wsd import lesk
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = tokens(text)
    senses = {}
    with _WORDNET_LOCK:
        local_corpus = str(Path(__file__).resolve().parents[2] / ".nltk_data")
        if local_corpus not in nltk.data.path:
            nltk.data.path.insert(0, local_corpus)
        wordnet.ensure_loaded()  # Provision explicitly; never download in requests.
        for word in sorted(set(words) & AMBIGUOUS_WORDS):
            context = set(words) - ENGLISH_STOP_WORDS - {word}
            candidates = wordnet.synsets(word, pos=wordnet.NOUN)
            candidates = [s for s in candidates if context & set(tokens(s.definition()))]
            if candidates:
                sense = lesk(sorted(context), word, pos="n", synsets=candidates)
                if sense is not None:
                    senses[word] = sense.name()
        augmented = list(words)
        for name in senses.values():
            sense = wordnet.synset(name)
            augmented.append("sense_" + name.replace(".", "_"))
            augmented.extend("meaning_" + lemma.lower() for lemma in sense.lemma_names()[:4])
    return " ".join(augmented), senses
