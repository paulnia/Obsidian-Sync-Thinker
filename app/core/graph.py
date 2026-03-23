from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from app.config import get_config
from app.core.nodes import critic_node, linker_node
from app.core.state import KnowledgeState

logger = logging.getLogger(__name__)


def should_continue(state: KnowledgeState) -> Literal["linker", "__end__"]:
    """
    条件边：决定 critic_node 之后走向哪里。

    规则（符合 `.cursorrules`）：
    - 必须尊重 `retry_count`
    - 必须根据“打回状态/放行状态”决定是否回退到 linker_node

    约定：
    - critic_node 将 status 置为：
      - "retry"：表示打回，需要回到 linker_node 重新生成
      - "approved"：表示放行，走向 END
      - "rejected"：表示打回且不再继续（这里也走 END）
    """

    status = str(state.get("status") or "")
    retry_count = int(state.get("retry_count") or 0)
    max_retries = int(state.get("max_retries") or 2)

    # 明确打回但仍允许重试：回到 linker
    if status == "retry" and retry_count < max_retries:
        return "linker"

    # 其余情况（approved / rejected / 达到上限）结束
    return "__end__"


def build_graph() -> Any:
    """
    组装并编译 LangGraph。

    注意：
    - 节点函数必须只返回增量更新字段（我们在 nodes.py 已遵守）
    - StateGraph 会把增量合并回 state（由 LangGraph 执行器负责）
    """
    cfg = get_config().data
    enable_critic = bool(getattr(cfg, "enable_critic", True))

    g = StateGraph(KnowledgeState)
    g.add_node("linker", linker_node)
    g.set_entry_point("linker")
    if not enable_critic:
        logger.info("Critic disabled by config (enable_critic=false)：仅执行 linker->END")
        # 仅执行一次 linker：用于降成本/调试快速验证
        g.add_edge("linker", END)
        return g.compile()

    g.add_node("critic", critic_node)
    # 主链路：linker -> critic -> 条件边（回 linker 或 END）
    g.add_edge("linker", "critic")
    g.add_conditional_edges("critic", should_continue, {"linker": "linker", "__end__": END})

    return g.compile()


# 对外暴露编译后的 graph 实例
graph = build_graph()

