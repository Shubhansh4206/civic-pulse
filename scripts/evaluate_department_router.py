"""Evaluate held-out synthetic scenarios; results do not establish real-world accuracy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from backend.services.department_service import load_router
from backend.services.routing_features import extract_features
from scripts.routing_dataset import rows


def evaluate(model, threshold=0.60):
    train, test = rows("train"), rows("test")
    texts = lambda data: [r["title"] + " " + r["description"] for r in data]
    # Report lexical near-duplicate screening independently from model features.
    analyzer = CountVectorizer(stop_words="english").build_analyzer()
    train_sets = [set(analyzer(t)) for t in texts(train)]
    test_sets = [set(analyzer(t)) for t in texts(test)]
    max_overlap = max(len(a & b) / len(a | b) for a in train_sets for b in test_sets)
    if max_overlap >= 0.65:
        raise ValueError("Potential near-duplicate across splits: review dataset")
    artifact = load_router(model)
    features = [extract_features(t)[0] for t in texts(test)]
    probabilities = artifact["classifier"].predict_proba(artifact["vectorizer"].transform(features))
    labels = list(artifact["classifier"].classes_)
    predicted = [labels[int(p.argmax())] for p in probabilities]
    actual = [r["department"] for r in test]
    accepted = [float(p.max()) >= threshold for p in probabilities]
    report = {
        "warning": "Synthetic held-out data only; does not establish real-world accuracy.",
        "model_version": artifact["model_version"], "train_count": len(train), "test_count": len(test),
        "max_cross_split_token_jaccard": round(max_overlap, 4), "threshold": threshold,
        "coverage": sum(accepted) / len(test),
        "accepted_accuracy": (sum(a == p for a, p, ok in zip(actual, predicted, accepted) if ok)
                              / sum(accepted)) if any(accepted) else None,
        "classification_report": classification_report(actual, predicted, output_dict=True, zero_division=0),
        "labels": labels, "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "examples": [{"id": r["id"], "expected": a,
                      "predicted": p if ok else "MANUAL_REVIEW", "confidence": round(float(prob.max()), 6)}
                     for r, a, p, ok, prob in zip(test, actual, predicted, accepted, probabilities)]}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models/department_router/router.joblib"))
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--output", default=str(ROOT / "models/department_router/evaluation.json"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between zero and one")
    report = evaluate(args.model, args.threshold)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
