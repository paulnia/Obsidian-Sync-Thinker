from __future__ import annotations

from typing import Any

from typing_extensions import NotRequired, TypedDict


class KnowledgeState(TypedDict):
    """
    LangGraph 核心状态结构（TypedDict）。

    重要约束（`.cursorrules`）：
    - 节点函数必须“视 state 为不可变”，不要原地修改 list/dict
    - 节点函数只能返回“需要更新的字段”，不要返回整个 state
    """

    # 任务/来源定位
    task_id: str
    source_file: str

    # 当前正在处理的 chunk（来自 MarkdownHandler）
    current_chunk: dict[str, Any]

    # 候选集合（例如向量检索返回的 chunk 列表）
    candidates: list[dict[str, Any]]

    # 链接候选（由 linker 产出，后续可写回或进入图推理）
    proposed_links: list[dict[str, Any]]

    # 审查日志（critic 追加）
    critique_log: list[str]

    # 重试计数（conditional edge 必须尊重该字段）
    retry_count: int

    # 可选字段：用于节点之间传递流程状态（例如 retry/approved/rejected）
    status: NotRequired[str]

    # 可选字段：最大重试次数（从配置注入，避免在节点/条件边中硬编码）
    max_retries: NotRequired[int]
