"""One-time download of the pretrained CPU model; never run inside a request."""
from pathlib import Path
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu",
                                cache_folder=str(destination.parent / ".semantic-cache"))
    model.save(str(destination))
    print(f"Saved pretrained semantic model to {destination}")
