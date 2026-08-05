"""共享 Embedding 提供者 — BGE-M3，懒加载单例"""
import os
import logging

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Milvus Lite 与 sentence-transformers 都内置 libomp，避免 OMP 冲突报错
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """BGE-M3 懒加载单例。首次加载 ~6s，之后 <1ms"""

    def __init__(self):
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载 Embedding 模型: {settings.embedding_model} "
                        f"(device={settings.embedding_device}, 首次较慢)")
            self._model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device,
                local_files_only=True,
            )
        return self._model

    def encode(self, texts, normalize=True) -> list:
        model = self._get()
        if isinstance(texts, str):
            texts = [texts]
        embs = model.encode(texts, normalize_embeddings=normalize,
                            show_progress_bar=False, batch_size=8)
        return embs.tolist()


embedding_provider = EmbeddingProvider()
