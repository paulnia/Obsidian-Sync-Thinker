from __future__ import annotations

import logging
import argparse
import asyncio
import contextlib
import signal
from pathlib import Path

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ost.log", encoding="utf-8"),
    ],
)

from app.config import get_config
from app.core.graph import graph
from app.core.state import KnowledgeState
from app.core.writer import InsightsWriter
from app.database.cache_manager import CacheManager
from app.database.vector_store import VectorStoreManager
from app.utils.md_handler import MarkdownHandler
from app.utils.observability import setup_langsmith
from app.service.watcher import ObsidianWatcher

logger = logging.getLogger(__name__)

async def worker_loop(queue: asyncio.Queue[str]) -> None:
    """
    队列消费者：持续从队列取出 filepath 并处理。

    目前仅打印，后续可在此接入：
    - CacheManager（MD5 幂等检查）
    - MarkdownHandler（解析与切块）
    - VectorStoreManager（向量化入库）
    - LangGraph 推理流程（严格与 I/O 解耦）
    """

    cfg = get_config().data
    md_handler = MarkdownHandler()
    cache_manager = CacheManager()
    # VectorStoreManager 的持久化路径已支持从配置读取，这里显式传入也可读性更强
    vector_store = VectorStoreManager(persist_dir=cfg.db_path)
    writer = InsightsWriter()

    while True:
        filepath = await queue.get()
        try:
            logger.info("开始处理任务: %s", filepath)

            # 1) 计算 MD5 并做幂等检查（未变化则跳过）
            try:
                # 关键修复：使用“核心内容 MD5”（已剔除 AI 回写区块），避免自我触发死循环
                current_md5 = await md_handler.compute_core_md5_async(filepath)
            except Exception as e:
                logger.error("计算核心内容 MD5 失败，跳过: %s，原因: %s", filepath, e)
                continue

            if not cache_manager.is_modified(filepath, current_md5):
                # 未变化：不重复向量化/推理
                continue

            # 2) 解析 + 分层切块（异步读取避免阻塞）
            chunks = await md_handler.parse_and_chunk_async(filepath)
            if not chunks:
                cache_manager.update_cache(filepath, current_md5)
                continue

            # 3) 清空该文件在向量库中的历史版本，然后 upsert 当前文件 chunks
            #    这样可以保证向量库中只保留“当前版本”的内容，避免旧版本长期堆积。
            vector_store.clear_file(filepath)
            vector_store.add_chunks(chunks)

            # 4) 对每个 chunk 检索 candidates（排除当前文件，优先同目录），跑 LangGraph 状态机，收集 links
            final_links: list[dict] = []

            for idx, chunk in enumerate(chunks):
                query_text = str(chunk.get("content") or "").strip()
                candidates = (
                    vector_store.search_similar(
                        query_text,
                        top_k=5,
                        exclude_file=filepath,
                        prefer_same_dir=True,
                    )
                    if query_text
                    else []
                )

                # 保险起见，再在调用侧过滤一次“当前文件”的候选，防止路径大小写/分隔符差异导致漏网
                if candidates:
                    cur_file_norm = str(filepath).replace("\\", "/").lower()
                    filtered: list[dict] = []
                    for c in candidates:
                        md = c.get("metadata") or {}
                        if not isinstance(md, dict):
                            filtered.append(c)
                            continue
                        f = str(md.get("file") or "").replace("\\", "/").lower()
                        if f == cur_file_norm:
                            continue
                        filtered.append(c)
                    candidates = filtered

                # KnowledgeState 初始化（只放必要字段）
                init_state: KnowledgeState = {
                    "task_id": f"{current_md5}:{idx}",
                    "source_file": filepath,
                    "current_chunk": chunk,
                    "candidates": candidates,
                    "proposed_links": [],
                    "critique_log": [],
                    "retry_count": 0,
                    # 把配置注入 state，供 conditional edge / critic 节点读取
                    "max_retries": int(cfg.max_retries),
                }

                try:
                    # 节点已切换为异步 LLM 调用，必须使用 ainvoke 以避免阻塞事件循环
                    out_state = await graph.ainvoke(init_state)
                except Exception as e:
                    logger.error("LangGraph 运行失败，跳过该 chunk: %s #%s，原因: %s", filepath, idx, e)
                    continue

                links = out_state.get("proposed_links") if isinstance(out_state, dict) else None
                if isinstance(links, list):
                    final_links.extend([l for l in links if isinstance(l, dict)])

            # 5) 回写 AI-Insights 区块（非侵入式 upsert）
            wr = await writer.write_links_async(filepath, final_links)
            if not wr.ok:
                logger.error("回写失败: %s，原因: %s", filepath, wr.error)

            # 6) 更新缓存（成功处理后写入新 hash）
            cache_manager.update_cache(filepath, current_md5)
        finally:
            queue.task_done()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Obsidian-Sync-Thinker (OST) skeleton runner")
    parser.add_argument("--queue-size", type=int, default=1000, help="任务队列容量（默认 1000）")
    args = parser.parse_args()

    # 读取配置：找不到配置文件会抛出明确异常（按需求）
    cfg = get_config().data
    vault_dir = get_config().vault_dir

    # 在使用任何 LLM 之前初始化 LangSmith 观测（基于环境变量）
    # - 若本机设置了 LANGSMITH_API_KEY 或 LANGCHAIN_API_KEY，则会自动开启 Tracing v2
    # - 未设置时不会有任何副作用
    setup_langsmith(project_name="Obsidian-Sync-Thinker")
    logger.info("OST 启动，vault=%s db=%s debounce=%ss max_retries=%s provider=%s",
                vault_dir.as_posix(), cfg.db_path, cfg.debounce_timeout, cfg.max_retries, cfg.llm.provider)

    # 1) 初始化 asyncio.Queue
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max(1, int(args.queue_size)))

    # 2) 获取当前事件循环（用于跨线程安全入队）
    loop = asyncio.get_running_loop()

    # 3) 启动 ObsidianWatcher（watchdog 自带线程，不阻塞 asyncio）
    watcher = ObsidianWatcher(
        vault_dir=vault_dir,
        queue=queue,
        loop=loop,
        debounce_seconds=float(cfg.debounce_timeout),
    )
    watcher.start()

    # 4) 启动 worker 消费循环
    worker_task = asyncio.create_task(worker_loop(queue))

    # 5) 优雅退出：捕获 Ctrl+C / SIGTERM，停止 watcher
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _request_stop)
        loop.add_signal_handler(signal.SIGTERM, _request_stop)
    except NotImplementedError:
        # Windows 事件循环对 signal handler 支持有限，保留 stop_event，依赖 KeyboardInterrupt
        pass

    try:
        await stop_event.wait()
    finally:
        watcher.stop()
        worker_task.cancel()
        with contextlib.suppress(Exception):
            await worker_task


if __name__ == "__main__":
    # asyncio 入口
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
