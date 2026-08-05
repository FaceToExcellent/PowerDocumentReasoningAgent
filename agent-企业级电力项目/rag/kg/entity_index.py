"""知识图谱实体关系索引 — 基于 LlamaIndex SimpleGraphStore

轻量 KG：用 SimpleGraphStore（JSON 落盘）存三元组 (subject, relation, object)，
配合内置电力实体别名词典做实体归一化，一跳关系查询走 SimpleGraphStore.query。

设计动机：
  - 不引入 Neo4j：电力场景实体 ~500 / 关系 ~2000，SimpleGraphStore 够用，零运维
  - 复用 RAG 基础设施：三元组从入库文档抽取（KG 与向量库同源），
    实体语义匹配复用 embedding_provider（BGE-M3）
  - 面试点：这是"向量检索补不了的精确关系"——问"主变和母线保护什么关系"，
    向量召回不到关系，KG 一跳就能命中
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

# 电力实体别名词典：name → 归一化实体名（"主变" == "主变压器"）
_ENTITY_ALIASES: Dict[str, str] = {
    "主变": "主变压器",
    "主变压器": "主变压器",
    "变压器": "主变压器",
    "断路器": "断路器",
    "开关": "断路器",
    "隔离开关": "隔离开关",
    "母线": "母线",
    "母排": "母线",
    "母线保护": "母线保护",
    "主变保护": "主变保护",
    "熔断器": "熔断器",
    "避雷器": "避雷器",
    "互感器": "互感器",
    "电容器": "电容器",
    "电抗器": "电抗器",
    "dl/t 572": "DL/T 572",
    "dl/t 574": "DL/T 574",
    "gb 50150": "GB 50150",
    "继电保护": "继电保护",
    "保护装置": "继电保护",
    "调度规程": "调度规程",
    "运行规程": "运行规程",
    "检修规程": "检修规程",
}


class EntityIndex:
    """KG 实体关系索引：三元组抽取 + 别名归一化 + 一跳关系查询"""

    def __init__(self, persist_path: str = None):
        self.persist_path = persist_path or str(
            Path(settings.chroma_persist_dir).parent / "kg" / "graph_store.json"
        )
        self._graph_store = None
        self._load()

    def _load(self):
        try:
            from llama_index.core.graph_stores import SimpleGraphStore
            p = Path(self.persist_path)
            if p.exists():
                self._graph_store = SimpleGraphStore.from_persist_path(str(p))
            else:
                self._graph_store = SimpleGraphStore()
        except ImportError:
            logger.warning("llama-index 未安装，KG 不可用（仅入库/查询被禁用）")
            self._graph_store = None

    # ── 实体归一化 ──
    def normalize_entity(self, name: str) -> str:
        """别名 → 归一化实体名（小写匹配，命中词典返回规范名）"""
        key = name.strip().lower()
        if key in _ENTITY_ALIASES:
            return _ENTITY_ALIASES[key]
        return name.strip()

    def extract_entities(self, text: str) -> List[str]:
        """从文本中抽取已知实体（词典子串匹配 + 别名归一化，去重）"""
        found = set()
        low = text.lower()
        for alias, canonical in _ENTITY_ALIASES.items():
            if alias.lower() in low:
                found.add(canonical)
        return sorted(found)

    # ── 三元组入库 ──
    def add_triplets(self, triplets: List[Tuple[str, str, str]], source_chunk_id: str = "") -> int:
        """批量写入三元组 (subject, relation, object)，实体先归一化"""
        if not self._graph_store:
            return 0
        added = 0
        for subj, rel, obj in triplets:
            subj, obj = self.normalize_entity(subj), self.normalize_entity(obj)
            if not subj or not obj or subj == obj:
                continue
            self._graph_store.upsert_triplet(subj, rel, obj)
            added += 1
        self.persist()
        return added

    # ── 一跳关系查询 ──
    def get_relations(self, entity: str, max_depth: int = 1) -> List[Dict]:
        """查实体的关系（默认一跳）。返回 [{subject, relation, object}]"""
        if not self._graph_store:
            return []
        ent = self.normalize_entity(entity)
        rels = []
        try:
            # get_rel_map: {entity: [[subject, relation, object], ...]}
            rel_map = self._graph_store.get_rel_map([ent], depth=max_depth)
            for _subj, triplets in rel_map.items():
                for triplet in triplets:
                    if len(triplet) == 3:
                        rels.append({
                            "subject": triplet[0],
                            "relation": triplet[1],
                            "object": triplet[2],
                        })
        except Exception as e:
            logger.warning(f"KG 关系查询失败: {e}")
        return rels

    def query(self, entity: str) -> List[Dict]:
        """别名感知的一跳查询（上层统一入口）"""
        return self.get_relations(entity, max_depth=1)

    # ── 持久化 ──
    def persist(self):
        if self._graph_store:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            self._graph_store.persist(self.persist_path)

    def count(self) -> int:
        """当前三元组数量"""
        try:
            data = getattr(self._graph_store, "_data", None)
            graph_dict = getattr(data, "graph_dict", None) or {}
            return sum(len(triplets) for triplets in graph_dict.values())
        except Exception:
            return 0


entity_index = EntityIndex()
