from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CacheRecord:
    """缓存记录（用于类型提示与调试展示）。"""

    filepath: str
    md5: str
    updated_at: float


class CacheManager:
    """
    文件 MD5 缓存管理器（持久化）：
    - 使用本地 SQLite 保存“文件路径 -> MD5”映射
    - 用于实现幂等：未变化的文件不重复 embed / infer

    设计说明：
    - 使用 sqlite3（Python 标准库）避免引入额外依赖。
    - SQLite 天然支持并发读写的基本一致性，适合本地工具型项目。
    """

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        """
        初始化缓存数据库。

        参数：
        - db_path: SQLite 文件路径；若不传，默认使用项目内 `./.ost_cache/cache.sqlite3`
        """

        if db_path is None:
            db_path = Path(".ost_cache") / "cache.sqlite3"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库表结构（幂等）
        self._init_schema()

    def is_modified(self, filepath: str, current_hash: str) -> bool:
        """
        判断文件是否“相对缓存发生变化”。

        约定：
        - 若缓存中不存在该 filepath，则视为“已修改/需要处理”，返回 True。
        - 若缓存中 md5 与 current_hash 不一致，则返回 True。
        - 若一致，则返回 False（不需要重复处理）。

        注意：
        - 该方法只负责“比较”，不负责计算 MD5（由调用方根据业务决定如何计算/何时计算）。
        """

        try:
            cached = self._get_md5(filepath)
        except Exception:
            # 优雅降级：读取缓存失败时，宁愿当作“已修改”以保证数据最终一致性
            return True

        if cached is None:
            return True

        if cached == current_hash:
            logger.info("MD5 校验未变，跳过处理 (截断死循环): %s", filepath)
            return False

        return True

    def update_cache(self, filepath: str, new_hash: str) -> None:
        """
        更新缓存（upsert）。

        注意：
        - 该方法只写入缓存，不做额外校验；上层可在写入前保证 new_hash 的正确性。
        """

        now = time.time()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO file_md5_cache (filepath, md5, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(filepath) DO UPDATE SET
                        md5=excluded.md5,
                        updated_at=excluded.updated_at
                    """,
                    (filepath, new_hash, now),
                )
        except Exception:
            # 优雅降级：写缓存失败不应阻断主流程（例如 Watchdog 事件队列消费者）
            # 上层可以通过日志/监控发现该问题并处理；这里保持静默失败。
            return

    # -----------------------------
    # 内部实现（不暴露给业务层）
    # -----------------------------

    def _connect(self) -> sqlite3.Connection:
        """
        获取 SQLite 连接。

        关键参数：
        - timeout: 避免短时间并发写导致“database is locked”立即抛错
        - isolation_level=None: 使用 autocommit 模式，减少锁持有时间（更适合工具型场景）
        """

        conn = sqlite3.connect(self._db_path.as_posix(), timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        """初始化数据库表结构（幂等）。"""

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_md5_cache (
                    filepath TEXT PRIMARY KEY,
                    md5 TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _get_md5(self, filepath: str) -> str | None:
        """读取某个文件路径对应的 md5；不存在则返回 None。"""

        with self._connect() as conn:
            cur = conn.execute(
                "SELECT md5 FROM file_md5_cache WHERE filepath = ? LIMIT 1",
                (filepath,),
            )
            row = cur.fetchone()
            return None if row is None else str(row[0])
