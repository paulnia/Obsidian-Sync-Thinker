from __future__ import annotations

import hashlib
import logging
import json
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
            logger.info("Chroma 向量库已启用：dir=%s, collection=%s", base_dir.as_posix(), collection_name)
        except Exception as e:
            # 优雅降级：初始化失败则禁用向量库，避免影响主流程，但记录详细原因便于诊断
            self._enabled = False
            self._collection = None
            logger.error("初始化 Chroma 向量库失败，将禁用向量检索: %s", e)

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
            logger.warning("尝试写入向量库但当前处于禁用状态（_enabled=False 或 collection=None），chunks 将被丢弃。")
            return

        # 说明：若你希望“每次处理一个文件时彻底替换该文件在向量库中的所有版本”，
        # 建议在调用层先执行 clear_file(filepath)，然后再调用 add_chunks。

        try:
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []

            # 为避免同一 file+header 的旧版本残留，这里先按 (file, header) 做一次软删除，
            # 然后再 upsert 当前批次的最新内容。
            seen_keys: set[tuple[str, str]] = set()

            for ch in chunks:
                content = str(ch.get("content") or "")
                md = ch.get("metadata") or {}
                if not isinstance(md, dict):
                    md = {}

                filepath = str(md.get("file") or "")
                header = str(md.get("header") or "")

                # 首次遇到某个 (file, header) 时，先删除该组合的历史向量，避免旧版本污染检索结果
                key = (filepath, header)
                if key not in seen_keys:
                    seen_keys.add(key)
                    try:
                        # Chroma 1.5+ 要求 where 只包含一个顶层操作符，这里使用 $and 组合 file 与 header
                        self._collection.delete(
                            where={
                                "$and": [
                                    {"file": filepath},
                                    {"header": header},
                                ]
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            "删除历史向量失败 (file=%s, header=%s)，可能残留旧版本结果：%s",
                            filepath,
                            header,
                            e,
                        )

                # 关键 metadata：file/header 必须写入
                metadata: dict[str, Any] = dict(md)
                metadata["file"] = filepath
                metadata["header"] = header

                # Chroma 的 metadata value 不能是 dict，这里统一对 dict 做 JSON 序列化
                for k, v in list(metadata.items()):
                    if isinstance(v, dict):
                        metadata[k] = json.dumps(v, ensure_ascii=False)

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
            logger.info("已向 Chroma 写入/更新 %s 个 chunks", len(ids))
        except Exception as e:
            # 优雅降级：写入失败不应阻断整体处理流水线，但记录错误便于排查
            logger.error("向 Chroma 写入 chunks 失败，将跳过本批次: %s", e)
            return

    def search_similar(
        self,
        query_text: str,
        top_k: int = 5,
        *,
        exclude_file: str | None = None,
        prefer_same_dir: bool = False,
    ) -> list[dict[str, Any]]:
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
            logger.warning("尝试检索向量库但当前处于禁用状态（_enabled=False 或 collection=None），返回空结果。")
            return []

        query_text = (query_text or "").strip()
        if not query_text:
            return []

        # 为了在 Python 侧做一定的重排序，这里多取一些候选，再根据目录等进行优先级调整
        # 实际返回仍然限制为 top_k。
        k = max(1, int(top_k))
        raw_k = max(k * 3, k)

        # 如果希望直接在 Chroma 侧排除当前文件，可使用 where 过滤，
        # 但部分版本对 $ne 支持有限，因此这里先不在 where 中写死，统一在 Python 侧过滤。
        try:
            res = self._collection.query(
                query_texts=[query_text],
                n_results=raw_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Chroma 检索失败，将返回空结果: %s", e)
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

        # --------------------------
        # 元数据过滤与优先级调整
        # --------------------------

        exclude_file_norm = str(exclude_file or "").replace("\\", "/").lower()
        same_dir_parent: str | None = None
        if exclude_file_norm:
            same_dir_parent = str(Path(exclude_file_norm).parent)

        filtered: list[dict[str, Any]] = []
        for item in out:
            md = item.get("metadata") or {}
            if not isinstance(md, dict):
                md = {}
            f = str(md.get("file") or "").replace("\\", "/").lower()

            # 1) 排除当前 chunk 所在文件
            if exclude_file_norm and f == exclude_file_norm:
                continue

            filtered.append(item)

        # 对同一 (file, header) 的多条候选，只保留距离最小的一条，避免“同一目标的多个版本”淹没结果
        best_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for item in filtered:
            md = item.get("metadata") or {}
            if not isinstance(md, dict):
                md = {}
            f = str(md.get("file") or "").replace("\\", "/").lower()
            h = str(md.get("header") or "")
            key = (f, h)

            dist = item.get("distance")
            try:
                d_val = float(dist) if dist is not None else 1.0
            except Exception:
                d_val = 1.0

            if key not in best_by_key:
                best_by_key[key] = item
            else:
                old = best_by_key[key]
                old_dist = old.get("distance")
                try:
                    old_val = float(old_dist) if old_dist is not None else 1.0
                except Exception:
                    old_val = 1.0
                if d_val < old_val:
                    best_by_key[key] = item

        deduped = list(best_by_key.values())

        # 2) 若启用 prefer_same_dir，则优先同一父目录下的笔记
        if prefer_same_dir and same_dir_parent:
            def _score(it: dict[str, Any]) -> tuple[int, float]:
                md = it.get("metadata") or {}
                if not isinstance(md, dict):
                    md = {}
                f = str(md.get("file") or "").replace("\\", "/").lower()
                parent = str(Path(f).parent) if f else ""
                same_dir_flag = 0 if parent == same_dir_parent else 1  # 0 表示“更优先”
                # distance 可能为 None，统一转换为 float
                dist = it.get("distance")
                try:
                    d_val = float(dist) if dist is not None else 1.0
                except Exception:
                    d_val = 1.0
                return (same_dir_flag, d_val)

            deduped.sort(key=_score)

        # 最终只返回 top_k 条
        return deduped[:k]

    def clear_file(self, filepath: str) -> None:
        """
        删除指定文件在向量库中的所有向量（任意 header）。

        用途：
        - 在对某个 Markdown 文件重新切块/写入前，先调用本方法，
          可以确保该文件在向量库中只保留“当前最新版本”的 chunks。
        """

        if not self._enabled or self._collection is None:
            return

        f_norm = str(filepath)
        try:
            # Chroma 1.5+ 要求 where 顶层为单一操作符或单字段，这里按 file 精确删除
            self._collection.delete(where={"file": f_norm})
            logger.info("已从向量库中清空文件的历史向量: %s", f_norm)
        except Exception as e:
            logger.warning("清空文件向量失败 (file=%s)，可能残留旧版本结果：%s", f_norm, e)
