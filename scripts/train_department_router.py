"""Train only the fixed synthetic TRAIN split; save vectorizer and NB together."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from backend.services.routing_features import extract_features, PREPROCESSING_VERSION
from scripts.routing_dataset import rows


def train(output):
    data = rows("train")  # Test split is never fitted or used to select parameters.
    features = [extract_features(r["title"] + " " + r["description"])[0] for r in data]
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", lowercase=True)
    matrix = vectorizer.fit_transform(features)
    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(matrix, [r["department"] for r in data])
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    artifact = {"classifier": classifier, "vectorizer": vectorizer,
                "model_version": "synthetic-nb-v1-" + digest[:12],
                "preprocessing_version": PREPROCESSING_VERSION,
                "training_sha256": digest, "training_count": len(data),
                "sklearn_version": sklearn.__version__, "nltk_version": nltk.__version__,
                "synthetic_only": True}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    metadata = {k: v for k, v in artifact.items() if k not in ("classifier", "vectorizer")}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "models/department_router/router.joblib"))
    args = parser.parse_args()
    train(args.output)
