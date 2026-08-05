"""共享 Embedding 提供者 — BGE-M3，懒加载单例 + 进程内文本→向量缓存"""
import os
import logging
from collections import OrderedDict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Milvus Lite 与 sentence-transformers 都内置 libomp，避免 OMP 冲突报错
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import settings

logger = logging.getLogger(__name__)

# 进程内文本→向量缓存上限。文本→向量是确定性映射（无过期），只防内存无限增长
EMB_CACHE_MAX = 2048


class EmbeddingProvider:
    """BGE-M3 懒加载单例。首次加载 ~6s，之后 <1ms。

    文本→向量缓存：同一段文本（query/文档 chunk）只 encode 一次，
    后续直接命中缓存，避免重复向量化。OrderedDict FIFO 淘汰。
    """

    def __init__(self):
        self._model = None
        self._text_cache: "OrderedDict[str, list]" = OrderedDict()

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
        # 只 encode 缓存未命中的文本，命中直接复用向量
        missing = [t for t in texts if t not in self._text_cache]
        if missing:
            new_embs = model.encode(missing, normalize_embeddings=normalize,
                                    show_progress_bar=False, batch_size=8)
            for t, e in zip(missing, new_embs):
                self._text_cache[t] = e.tolist()
                self._text_cache.move_to_end(t)
                if len(self._text_cache) > EMB_CACHE_MAX:
                    self._text_cache.popitem(last=False)
        return [self._text_cache[t] for t in texts]

    def cache_size(self) -> int:
        return len(self._text_cache)


embedding_provider = EmbeddingProvider()
