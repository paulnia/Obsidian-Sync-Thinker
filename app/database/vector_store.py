from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_config

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class VectorStoreResult:
    """向量检索结果（用于类型提示与调试）。"""

    content: str
    metadata: dict[str, Any]
    distance: float | None = None
    id: str | None = None


class VectorStoreManager:
    """
    本地知识向量库管理器（ChromaDB）：
    - 持久化目录固定为项目根目录 `./.chroma_db`
    - 存储 chunk 的 content 与 metadata（至少包含 file、header）

    设计要点（符合 `.cursorrules`）：
    - 初始化/读写/检索均做异常捕捉，避免阻断文件监听/队列消费者等链路
    - metadata 必须携带 filepath 与 header，便于定位原文与后续安全回写
    """

    def __init__(self, persist_dir: str | None = None, collection_name: str = "ost_chunks") -> None:
        """
        初始化 ChromaDB 客户端与集合。

        参数：
        - persist_dir: 持久化目录（默认 `./.chroma_db`）
        - collection_name: collection 名称（默认 `ost_chunks`）
        """

        self._enabled: bool = False
        self._collection = None

        # persist_dir 若不传，则从配置读取（避免硬编码）
        if persist_dir is None:
            persist_dir = get_config().data.db_path
        base_dir = Path(persist_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            # ChromaDB 持久化客户端
            client = chromadb.PersistentClient(path=base_dir.as_posix())

            # 默认轻量 embedding（由 Chroma 提供的默认实现；通常为本地轻量模型）
            # 若后续需要替换为云端/本地自定义 embedding，可在此处替换 embedding_function。
            embedding_fn = embedding_functions.DefaultEmbeddingFunction()

            self._collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self._enabled = True
        except Exception:
            # 优雅降级：初始化失败则禁用向量库，避免影响主流程
            self._enabled = False
            self._collection = None

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """
        写入/更新 chunks 到向量库。

        约定 chunks 格式（来自 `MarkdownHandler.parse_and_chunk`）：
        - chunk["content"]: str
        - chunk["metadata"]["file"]: str
        - chunk["metadata"]["header"]: str
        - chunk["metadata"]["yaml"]: dict（可选但建议保留）

        注意：
        - 为了幂等（Idempotency），这里为每个 chunk 生成稳定 ID，并使用 upsert。
        """

        if not self._enabled or self._collection is None:
            return

        try:
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []

            for ch in chunks:
                content = str(ch.get("content") or "")
                md = ch.get("metadata") or {}
                if not isinstance(md, dict):
                    md = {}

                filepath = str(md.get("file") or "")
                header = str(md.get("header") or "")

                # 关键 metadata：file/header 必须写入
                metadata: dict[str, Any] = dict(md)
                metadata["file"] = filepath
                metadata["header"] = header

                # 生成稳定 ID：file + header + content 的 md5
                # 这样同一文件同一标题同一内容不会重复入库，满足“不要重复向量化”理念。
                stable_key = f"{filepath}\n{header}\n{content}".encode("utf-8", errors="replace")
                chunk_id = hashlib.md5(stable_key).hexdigest()

                ids.append(chunk_id)
                documents.append(content)
                metadatas.append(metadata)

            if not ids:
                return

            # upsert：存在则更新，不存在则插入
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        except Exception:
            # 优雅降级：写入失败不应阻断整体处理流水线
            return

    def search_similar(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        语义检索：返回最相似的 chunks（content + metadata）。

        返回格式：
        - list[dict]，每个元素：
          {
            "content": "...",
            "metadata": {...},
            "distance": 0.123,   # 可能为 None（取决于底层返回）
            "id": "..."
          }
        """

        if not self._enabled or self._collection is None:
            return []

        query_text = (query_text or "").strip()
        if not query_text:
            return []

        # top_k 最小为 1
        k = max(1, int(top_k))

        try:
            res = self._collection.query(
                query_texts=[query_text],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        # Chroma 的返回结构通常是按 query 维度嵌套 list（这里 query_texts 只有一个）
        docs = (res or {}).get("documents") or [[]]
        metas = (res or {}).get("metadatas") or [[]]
        dists = (res or {}).get("distances") or [[]]
        ids = (res or {}).get("ids") or [[]]

        out: list[dict[str, Any]] = []

        # 防御式处理：避免结构不符合预期导致异常
        doc_list = docs[0] if isinstance(docs, list) and docs else []
        meta_list = metas[0] if isinstance(metas, list) and metas else []
        dist_list = dists[0] if isinstance(dists, list) and dists else []
        id_list = ids[0] if isinstance(ids, list) and ids else []

        n = max(len(doc_list), len(meta_list), len(dist_list), len(id_list))
        for i in range(n):
            content = doc_list[i] if i < len(doc_list) else ""
            metadata = meta_list[i] if i < len(meta_list) and isinstance(meta_list[i], dict) else {}
            distance = dist_list[i] if i < len(dist_list) else None
            _id = id_list[i] if i < len(id_list) else None

            out.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "distance": distance,
                    "id": _id,
                }
            )

        return out
