"""Cross-Encoder Reranker: 对初召结果做语义精排

使用 BAAI/bge-reranker-v2-m3 对 query-doc 对做 joint encoding，
比双向量（embedding 相似度）排序精度更高。
"""

import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder 精排模型，懒加载 + 自动降级"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._available = True

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            logger.info(f"Loading reranker model {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, local_files_only=True
            )
            self._model.eval()
            logger.info("Reranker model loaded.")
        except OSError:
            logger.warning(
                f"Reranker model {self.model_name} not found locally. "
                "Falling back to no reranking. "
                "Run the following to download:\n"
                f"  python -c \"from transformers import AutoModelForSequenceClassification, AutoTokenizer; "
                f"AutoTokenizer.from_pretrained('{self.model_name}'); "
                f"AutoModelForSequenceClassification.from_pretrained('{self.model_name}')\""
            )
            self._available = False
        except Exception as e:
            logger.warning(f"Failed to load reranker model: {e}. Falling back to no reranking.")
            self._available = False

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """对文档列表做 cross-encoder 重排序"""
        self._load_model()
        if not self._available:
            return documents[:top_k]

        if not documents:
            return []

        try:
            import torch

            pairs = [(query, doc["text"]) for doc in documents]
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                scores = outputs.logits.squeeze(-1).tolist()

            if isinstance(scores, float):
                scores = [scores]

            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return documents[:top_k]

        except Exception as e:
            logger.warning(f"Rerank inference failed: {e}. Returning non-reranked results.")
            return documents[:top_k]


# 单例
reranker = Reranker()
