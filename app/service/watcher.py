from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from app.config import get_config

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DebounceConfig:
    """防抖配置。"""

    seconds: float = 10.0


class ObsidianWatcher:
    """
    Obsidian 文件夹监听器（Watchdog + 防抖 + asyncio.Queue 解耦）。

    核心约束（来自 `.cursorrules`）：
    - 文件 I/O（watchdog 事件回调线程）与后续异步处理（asyncio worker）必须绝对解耦
    - 任何文件事件都必须走 10 秒防抖，然后把 filepath 放入 asyncio.Queue

    设计说明：
    - Watchdog 的 Observer/回调通常运行在独立线程中。
    - `asyncio.Queue` 只能在其所属事件循环线程安全地操作，因此这里通过
      `loop.call_soon_threadsafe(...)` 把“入队动作”切回事件循环线程执行。
    - 防抖采用“按文件路径独立计时器”的策略：同一路径在 10 秒内多次变动只会入队一次。
    """

    def __init__(
        self,
        vault_dir: str | Path | None,
        queue: asyncio.Queue[str],
        loop: asyncio.AbstractEventLoop,
        debounce_seconds: float = 10.0,
    ) -> None:
        # vault_dir 若不传，则从配置读取（避免硬编码）
        if vault_dir is None:
            vault_dir = get_config().vault_dir
        self._vault_dir = Path(vault_dir).expanduser().resolve()
        self._queue = queue
        self._loop = loop
        self._debounce = DebounceConfig(seconds=float(debounce_seconds))

        self._observer = None
        self._running = False

        # 防抖状态：filepath -> Timer
        self._lock = threading.Lock()
        self._timers: dict[str, threading.Timer] = {}

    def start(self) -> None:
        """
        启动监听（非阻塞）。

        注意：
        - 该方法适合在后台线程或主线程中调用（内部会启动 watchdog 的线程）。
        """

        if self._running:
            return

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except Exception:
            # 优雅降级：watchdog 未安装或导入失败时，直接不启动
            return

        if not self._vault_dir.exists() or not self._vault_dir.is_dir():
            # 目录无效：不启动
            return

        watcher = self

        class _Handler(FileSystemEventHandler):
            """watchdog 回调处理器（运行在 watchdog 线程）。"""

            def on_modified(self, event) -> None:  # type: ignore[override]
                watcher._on_fs_event(event)

            def on_created(self, event) -> None:  # type: ignore[override]
                watcher._on_fs_event(event)

            def on_moved(self, event) -> None:  # type: ignore[override]
                watcher._on_fs_event(event)

        try:
            handler = _Handler()
            observer = Observer()
            # recursive=True：监听整个 vault
            observer.schedule(handler, self._vault_dir.as_posix(), recursive=True)
            observer.start()
        except Exception:
            # 优雅降级：启动失败则不阻断主程序
            return

        self._observer = observer
        self._running = True

    def stop(self) -> None:
        """停止监听并清理定时器。"""

        if not self._running:
            return

        # 停止 observer
        try:
            if self._observer is not None:
                self._observer.stop()
                self._observer.join(timeout=5)
        except Exception:
            pass
        finally:
            self._observer = None
            self._running = False

        # 取消所有防抖定时器
        with self._lock:
            for t in self._timers.values():
                try:
                    t.cancel()
                except Exception:
                    pass
            self._timers.clear()

    # -------------------------
    # 内部逻辑：事件过滤 + 防抖
    # -------------------------

    def _on_fs_event(self, event) -> None:
        """
        watchdog 事件入口（发生在 watchdog 线程）。

        关键点：
        - 只负责“过滤 + 启动/重置防抖计时器”
        - 绝不直接做耗时处理，避免阻塞 watchdog 线程
        """

        try:
            if getattr(event, "is_directory", False):
                return

            # moved 事件可能同时带 src_path / dest_path，这里优先 dest_path
            path = getattr(event, "dest_path", None) or getattr(event, "src_path", None)
            if not path:
                return

            filepath = str(Path(path).resolve())
            if not self._should_include(filepath):
                return

            self._debounce_emit(filepath)
        except Exception:
            # 优雅降级：事件处理异常不应影响 observer 线程
            return

    def _should_include(self, filepath: str) -> bool:
        """
        过滤规则：
        - 忽略 `.obsidian/` 目录下任何文件
        - 仅处理 `.md`（可按需扩展）
        """

        p = Path(filepath)

        # 忽略 .obsidian
        if ".obsidian" in {part.lower() for part in p.parts}:
            return False

        # Obsidian 核心内容：Markdown
        if p.suffix.lower() != ".md":
            return False

        return True

    def _debounce_emit(self, filepath: str) -> None:
        """
        按 filepath 防抖：10 秒内重复事件只保留最后一次。

        实现方式：
        - 为每个 filepath 维护一个 Timer
        - 新事件到来时取消旧 Timer 并重新启动
        - Timer 到期后把 filepath 放入 asyncio.Queue（切回事件循环线程执行）
        """

        def fire() -> None:
            # Timer 回调发生在 Timer 线程，必须线程安全地把任务切回 asyncio loop
            logger.info("防抖结束，推入任务队列: %s", filepath)
            self._loop.call_soon_threadsafe(self._enqueue_filepath, filepath)

        with self._lock:
            old = self._timers.get(filepath)
            if old is not None:
                try:
                    old.cancel()
                except Exception:
                    pass

            t = threading.Timer(self._debounce.seconds, fire)
            t.daemon = True
            self._timers[filepath] = t
            t.start()

    def _enqueue_filepath(self, filepath: str) -> None:
        """
        在 asyncio 事件循环线程中执行的“入队动作”。

        说明：
        - 优先使用 put_nowait，避免阻塞事件循环
        - 若队列满，则降级为 create_task(queue.put(...))（可等待直到有空位）
        """

        # 清理 timer 记录（避免字典无限增长）
        with self._lock:
            self._timers.pop(filepath, None)

        try:
            self._queue.put_nowait(filepath)
        except asyncio.QueueFull:
            # 队列满：降级为异步等待入队
            try:
                self._loop.create_task(self._queue.put(filepath))
            except Exception:
                return
