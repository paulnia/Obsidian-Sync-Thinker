from __future__ import annotations
# # 在 main.py 最顶部加上：
# import os
# # 请把 7890 换成你实际使用的代理软件端口（如 Clash 通常是 7890，v2ray 是 10809）
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
import logging
import argparse
import asyncio
import contextlib
import re
import signal
import time
import uuid
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
from app.utils.llm_factory import LLMFactory
from app.utils.observability import setup_langsmith
from app.service.watcher import ObsidianWatcher

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _RerankResponse(BaseModel):
    """Rerank 的结构化输出：最多返回 3 个候选 ID（按相关性从高到低）。"""

    ranked_ids: list[str] = Field(default_factory=list, max_length=3)


def _transform_query_for_retrieval(chunk: dict) -> str:
    """
    Query transformation（轻量版）：
    - 统一空白，降低向量检索的噪声
    - 保留 header 语义锚点（若 header 尚未出现在前部）
    - 限制长度控制 embedding 与后续 rerank prompt 成本
    """

    content = str(chunk.get("content") or "").strip()
    md = chunk.get("metadata") or {}
    header = str(md.get("header") or "").strip() if isinstance(md, dict) else ""

    content = " ".join(content.split())

    if header and header not in content[:200]:
        content = f"{header}\n{content}"

    if len(content) > 1200:
        content = content[:1200]
    return content


def _hard_filter_candidates(
    candidates: list[dict[str, object]],
    *,
    exclude_file_norm: str,
    min_chars: int = 120,
    max_distinct_headers: int = 8,
) -> list[dict[str, object]]:
    """
    Hard Filtering（硬过滤）：
    1) 去 self（exclude_file_norm）
    2) 去短文本（content 长度阈值）
    3) header 优先去重（尽量保留不同章节，避免局部冗余）
    """

    def _dist(it: dict[str, object]) -> float:
        d = it.get("distance")
        try:
            return float(d) if d is not None else 1.0
        except Exception:
            return 1.0

    sorted_items = sorted(candidates, key=_dist)

    cleaned: list[dict[str, object]] = []
    for c in sorted_items:
        content = str(c.get("content") or "").strip()
        if len(content) < min_chars:
            continue

        md = c.get("metadata") or {}
        if not isinstance(md, dict):
            md = {}
        f = str(md.get("file") or "").replace("\\", "/").lower()
        if f and f == exclude_file_norm:
            continue

        cleaned.append(c)

    distinct: list[dict[str, object]] = []
    seen_headers: set[str] = set()
    for c in cleaned:
        md = c.get("metadata") or {}
        if not isinstance(md, dict):
            md = {}
        header = str(md.get("header") or "").strip()

        # header 为空则退化到 id，避免所有空 header 被去重成同一项
        key = header if header else str(c.get("id") or "")
        if key in seen_headers:
            continue
        seen_headers.add(key)
        distinct.append(c)
        if len(distinct) >= max_distinct_headers:
            break

    return distinct


def _info_density_score(text: str) -> float:
    """
    信息密度启发式评分（近似指标）：
    - 去掉空白后，取“独特字符数 / 总字符数”的比例，再乘以总长度权重
    - 目标：偏向“内容更丰富且不太重复”的片段
    """

    t = re.sub(r"\s+", "", str(text or ""))
    if not t:
        return 0.0
    uniq = len(set(t))
    total = max(1, len(t))
    # 加一个长度权重，避免纯短文本在密度上占优
    return (uniq / total) * total


def _heuristic_prefilter_candidates(
    candidates: list[dict[str, object]],
    *,
    top_n: int = 5,
) -> list[dict[str, object]]:
    """
    Heuristic Pre-Filter：
    - 按长度 / 信息密度排序
    - 保留 Top-N
    """

    def _content(it: dict[str, object]) -> str:
        return str(it.get("content") or "")

    def _score(it: dict[str, object]) -> tuple[int, float]:
        c = _content(it).strip()
        length = len(c)
        density = _info_density_score(c)
        return (length, density)

    sorted_items = sorted(candidates, key=_score, reverse=True)
    return sorted_items[: max(1, int(top_n))]


async def _conditional_rerank(
    *,
    llm: object,
    current_text: str,
    candidates: list[dict[str, object]],
    trace_id: str,
    max_pick: int = 3,
) -> tuple[list[dict[str, object]], int]:
    """
    Conditional Rerank（关键）：
    - 当候选数量 > 4：调用轻量 LLM rerank（只返回候选 ID）
    - 按 ranked_ids 选择 Top-N
    - rerank 失败则回退到原始 candidates[:max_pick]
    """

    if len(candidates) <= 4:
        # 未触发 rerank：不增加 LLM 调用次数
        return candidates[:max_pick], 0

    logger.info("Rerank triggered trace_id=%s candidates=%d", trace_id, len(candidates))

    MAX_CAND_CHARS_FOR_RERANK = 450
    cand_lines: list[str] = []
    for c in candidates:
        cand_id = str(c.get("id") or "")
        md = c.get("metadata") or {}
        if not isinstance(md, dict):
            md = {}
        header = str(md.get("header") or "").strip()
        content = str(c.get("content") or "").strip()
        if len(content) > MAX_CAND_CHARS_FOR_RERANK:
            content = content[:MAX_CAND_CHARS_FOR_RERANK] + "..."
        cand_lines.append(f"- ID={cand_id} HEADER={header}\n  TEXT={content}")

    candidates_blob = "\n".join(cand_lines)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是候选片段的逻辑重排器（reranker）。\n"
                "目标：从候选中选择与“当前文本”存在真实逻辑关联的片段。\n\n"
                "输出硬性要求：\n"
                "1. 只能输出 JSON（必须匹配 schema）\n"
                "2. 字段 ranked_ids：按相关性从高到低排序，最多返回 3 个候选 ID\n"
                "3. 不要解释、不允许多余字段。",
            ),
            (
                "human",
                "当前文本：\n{current_text}\n\n候选片段：\n{candidates_blob}\n",
            ),
        ]
    )

    try:
        structured_llm = llm.with_structured_output(_RerankResponse)  # type: ignore[attr-defined]
        resp = await (prompt | structured_llm).ainvoke(
            {"current_text": current_text, "candidates_blob": candidates_blob}
        )
        ranked_ids = list(getattr(resp, "ranked_ids", []) or [])
    except Exception as e:
        logger.warning("Rerank 失败，回退到原始 candidates。trace_id=%s error=%s", trace_id, e)
        # 发生 rerank 调用失败：本次计入一次 LLM 调用预算消耗
        return candidates[:max_pick], 1

    logger.info("Rerank result trace_id=%s ranked_ids=%s", trace_id, ranked_ids)

    by_id: dict[str, dict[str, object]] = {}
    for c in candidates:
        cid = str(c.get("id") or "")
        if cid:
            by_id[cid] = c

    picked: list[dict[str, object]] = []
    for cid in ranked_ids:
        if cid in by_id:
            picked.append(by_id[cid])
        if len(picked) >= max_pick:
            break

    if not picked:
        return candidates[:max_pick], 1

    if len(picked) < max_pick:
        remaining = [c for c in candidates if c not in picked]
        picked.extend(remaining[: max_pick - len(picked)])

    return picked, 1


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
    # 进程级简单统计：用于观测总体健康度（错误累计、处理任务数）
    worker_task_count = 0
    worker_error_count = 0
    # 并发控制与背压
    semaphore = asyncio.Semaphore(int(cfg.max_concurrent_tasks))
    max_pending_tasks = int(cfg.max_pending_tasks)
    pending_tasks: set[asyncio.Task[None]] = set()
    last_backpressure_log_at = 0.0

    async def task_wrapper(filepath: str) -> None:
        nonlocal worker_task_count, worker_error_count

        try:
            # 限制“同时处理的文件任务数”，避免批量改动时资源耗尽/限流
            async with semaphore:
                task_started_at = time.perf_counter()
                embed_started_at = 0.0
                embed_seconds = 0.0
                llm_seconds = 0.0
                total_llm_calls = 0
                task_error: str | None = None
                status = "SUCCESS"
                # 任务级 trace_id：贯穿 worker 与 LangGraph 节点日志
                trace_id = uuid.uuid4().hex[:12]
                worker_task_count += 1

            try:
                logger.info("开始处理任务: %s trace_id=%s", filepath, trace_id)

                # Context Refinement Layer & Dynamic Control（文件级策略）
                # 1) 动态 chunk：避免无效调用，只处理信息量更大的 Top-N chunks
                DYNAMIC_CHUNK_TOP_N = 5
                # 2) 调用熔断：限制每个文件最多的 LLM 调用次数（包含 rerank + linker/critic）
                MAX_LLM_CALLS_PER_FILE = 5
                # 3) 上下文压缩：current_chunk / candidate 内容截断
                CURRENT_CHUNK_MAX_CHARS = 300
                CANDIDATE_MAX_CHARS = 200
                # 4) 候选启发式筛选：Hard Filter 后保留 Top-5，再进入 LLM rerank
                HEURISTIC_CANDIDATE_TOP_N = 5

                # 1) 计算 MD5 并做幂等检查（未变化则跳过）
                try:
                    # 关键修复：使用“核心内容 MD5”（已剔除 AI 回写区块），避免自我触发死循环
                    current_md5 = await md_handler.compute_core_md5_async(filepath)
                except Exception as e:
                    logger.error("计算核心内容 MD5 失败，跳过: %s，原因: %s", filepath, e)
                    status = "FAILED"
                    task_error = f"core_md5_failed: {e}"
                    worker_error_count += 1
                    return

                if not cache_manager.is_modified(filepath, current_md5):
                    # 未变化：不重复向量化/推理
                    status = "SKIPPED_UNCHANGED"
                    return

                # 2) 解析 + 分层切块（异步读取避免阻塞）
                embed_started_at = time.perf_counter()
                chunks = await md_handler.parse_and_chunk_async(filepath)
                if not chunks:
                    cache_manager.update_cache(filepath, current_md5)
                    status = "SKIPPED_EMPTY_CHUNKS"
                    embed_seconds = time.perf_counter() - embed_started_at
                    return

                # 3) 清空该文件在向量库中的历史版本，然后 upsert 当前文件 chunks
                #    这样可以保证向量库中只保留“当前版本”的内容，避免旧版本长期堆积。
                vector_store.clear_file(filepath)
                vector_store.add_chunks(chunks)
                embed_seconds = time.perf_counter() - embed_started_at

                # 4) 动态 chunk 选择（Dynamic Chunk Selection）
                #    只处理内容长度更长的 Top-N，避免大量无意义 chunk 触发 LLM 调用。
                chunks_sorted = sorted(
                    chunks,
                    key=lambda c: len(str(c.get("content") or "")),
                    reverse=True,
                )
                chunks_for_infer = chunks_sorted[: max(1, int(DYNAMIC_CHUNK_TOP_N))]
                logger.info(
                    "Dynamic Chunk Selection trace_id=%s file=%s original_chunks=%d selected_chunks=%d",
                    trace_id,
                    filepath,
                    len(chunks),
                    len(chunks_for_infer),
                )

                # 5) 对每个 chunk 检索 candidates（排除当前文件，优先同目录），跑 LangGraph 状态机，收集 links
                final_links: list[dict] = []

                for idx, chunk in enumerate(chunks_for_infer):
                    if total_llm_calls > MAX_LLM_CALLS_PER_FILE:
                        logger.warning(
                            "Call budget exceeded trace_id=%s file=%s llm_calls=%d > max=%d, stop remaining chunks.",
                            trace_id,
                            filepath,
                            total_llm_calls,
                            MAX_LLM_CALLS_PER_FILE,
                        )
                        break

                    # 轻量 query transformation：提升检索质量（不改变后续 linker/critic 逻辑契约）
                    query_text = _transform_query_for_retrieval(chunk)
                    candidates = (
                        vector_store.search_similar(
                            query_text,
                            top_k=10,
                            exclude_file=filepath,
                            prefer_same_dir=True,
                        )
                        if query_text
                        else []
                    )

                    # 1) Hard Filtering：去 self / 去短文本 / header 去重，减少 LLM 上下文干扰
                    cur_file_norm = str(filepath).replace("\\", "/").lower()
                    candidates = _hard_filter_candidates(
                        candidates,
                        exclude_file_norm=cur_file_norm,
                    )

                    # 2) Heuristic Pre-Filter：
                    #    按长度 / 信息密度启发式排序，保留 Top-5，再交给 LLM rerank
                    candidates = _heuristic_prefilter_candidates(
                        candidates,
                        top_n=HEURISTIC_CANDIDATE_TOP_N,
                    )

                    # 3) LLM Rerank（轻量模型）：
                    #    输入 Top-5，输出 Top-2~3（ID列表），失败回退到原始 Top-3
                    rerank_cfg = cfg.llm_reasoning or cfg.llm
                    rerank_llm = LLMFactory.get_llm(rerank_cfg)
                    candidates, rerank_calls = await _conditional_rerank(
                        llm=rerank_llm,
                        current_text=query_text,
                        candidates=candidates,
                        trace_id=trace_id,
                        max_pick=3,
                    )

                    total_llm_calls += int(rerank_calls or 0)
                    logger.info(
                        "Context Refinement trace_id=%s file=%s chunk_idx=%d candidates_after_rerank=%d llm_calls=%d",
                        trace_id,
                        filepath,
                        idx,
                        len(candidates),
                        total_llm_calls,
                    )

                    if total_llm_calls > MAX_LLM_CALLS_PER_FILE:
                        logger.warning(
                            "Call budget reached after rerank trace_id=%s file=%s llm_calls=%d > max=%d, break.",
                            trace_id,
                            filepath,
                            total_llm_calls,
                            MAX_LLM_CALLS_PER_FILE,
                        )
                        break

                    # 4) Context Truncation（纯压缩）：
                    #    current_chunk <= 300 chars, candidate <= 200 chars
                    current_chunk_trunc = dict(chunk)
                    raw_current_content = str(chunk.get("content") or "")
                    current_chunk_trunc["content"] = raw_current_content[:CURRENT_CHUNK_MAX_CHARS]

                    candidates_trunc: list[dict[str, object]] = []
                    for c in candidates:
                        cc = dict(c)
                        cc["content"] = str(c.get("content") or "")[:CANDIDATE_MAX_CHARS]
                        candidates_trunc.append(cc)

                    trunc_current_len = len(str(current_chunk_trunc.get("content") or ""))
                    trunc_candidate_lens = [len(str(cc.get("content") or "")) for cc in candidates_trunc]
                    logger.info(
                        "Context Compression trace_id=%s file=%s chunk_idx=%d current_len=%d->%d candidate_lens=%s",
                        trace_id,
                        filepath,
                        idx,
                        len(raw_current_content),
                        trunc_current_len,
                        trunc_candidate_lens,
                    )

                    # KnowledgeState 初始化（只放必要字段）
                    init_state: KnowledgeState = {
                        "task_id": f"{current_md5}:{idx}",
                        "trace_id": trace_id,
                        "source_file": filepath,
                        "current_chunk": current_chunk_trunc,
                        "candidates": candidates_trunc,
                        "proposed_links": [],
                        "critique_log": [],
                        "retry_count": 0,
                        # 任务指标：由各节点按增量方式累计
                        "metrics": {"llm_call_count": 0},
                        # 把配置注入 state，供 conditional edge / critic 节点读取
                        "max_retries": int(cfg.max_retries),
                    }

                    try:
                        # 节点已切换为异步 LLM 调用，必须使用 ainvoke 以避免阻塞事件循环
                        llm_started_at = time.perf_counter()
                        out_state = await graph.ainvoke(init_state)
                        llm_seconds += time.perf_counter() - llm_started_at
                    except Exception as e:
                        logger.error("LangGraph 运行失败，跳过该 chunk: %s #%s，原因: %s", filepath, idx, e)
                        worker_error_count += 1
                        # chunk 级失败不终止整文件任务，仅记录观测
                        task_error = f"chunk_failed#{idx}: {e}"
                        continue

                    links = out_state.get("proposed_links") if isinstance(out_state, dict) else None
                    if isinstance(links, list):
                        final_links.extend([l for l in links if isinstance(l, dict)])
                    metrics = out_state.get("metrics") if isinstance(out_state, dict) else None
                    if isinstance(metrics, dict):
                        total_llm_calls += int(metrics.get("llm_call_count") or 0)

                    # 调用熔断：超出 LLM 预算后，跳过后续 chunk 的复杂推理
                    if total_llm_calls > MAX_LLM_CALLS_PER_FILE:
                        logger.warning(
                            "Call budget reached after linker trace_id=%s file=%s llm_calls=%d > max=%d, stop remaining chunks.",
                            trace_id,
                            filepath,
                            total_llm_calls,
                            MAX_LLM_CALLS_PER_FILE,
                        )
                        break

                # 5) 回写 AI-Insights 区块（非侵入式 upsert）
                wr = await writer.write_links_async(filepath, final_links)
                if not wr.ok:
                    logger.error("回写失败: %s，原因: %s", filepath, wr.error)
                    worker_error_count += 1
                    status = "FAILED"
                    task_error = f"writer_failed: {wr.error}"

                # 6) 更新缓存（成功处理后写入新 hash）
                cache_manager.update_cache(filepath, current_md5)
            finally:
                if status == "SUCCESS" and task_error:
                    status = "PARTIAL_SUCCESS"
                duration_seconds = time.perf_counter() - task_started_at
                logger.info(
                    "[TASK_SUMMARY] TraceID=%s File=%s Status=%s Duration=%.2fs LLM_Calls=%d Embed=%.2fs LLM=%.2fs Error=%s WorkerTasks=%d WorkerErrors=%d",
                    trace_id,
                    filepath,
                    status,
                    duration_seconds,
                    total_llm_calls,
                    embed_seconds,
                    llm_seconds,
                    task_error or "None",
                    worker_task_count,
                    worker_error_count,
                )
        finally:
            # queue.task_done()：语义必须在“任务实际结束（含失败/异常）”时调用
            queue.task_done()

    while True:
        # 背压：pending 过多时，暂停消费以避免任务无限堆积
        if len(pending_tasks) >= max_pending_tasks:
            now = time.time()
            if now - last_backpressure_log_at > 10:
                last_backpressure_log_at = now
                logger.warning(
                    "Backpressure: pending_tasks=%d >= max_pending_tasks=%d, pause queue consumption.",
                    len(pending_tasks),
                    max_pending_tasks,
                )
            await asyncio.sleep(0.05)
            continue

        filepath = await queue.get()
        task = asyncio.create_task(task_wrapper(filepath))
        pending_tasks.add(task)
        task.add_done_callback(lambda t: pending_tasks.discard(t))


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
