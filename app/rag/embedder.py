import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from app.core.config import settings


class Embedder:
    def __init__(self):
        self.model = None

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model {settings.embedding_model}...")
            self.model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device
            )
            print("Model loaded.")

    def embed_text(self, text: str) -> list[float]:
        self._load_model()
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


embedder = Embedder()
