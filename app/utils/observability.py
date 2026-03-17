from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)


def setup_langsmith(project_name: str | None = None) -> None:
    """
    根据环境变量自动启用 LangSmith 观测（LangChain Tracing v2）。

    约定（由用户在本机/部署环境中配置）：
    - LANGSMITH_API_KEY 或 LANGCHAIN_API_KEY：任意其一存在即认为启用 LangSmith
    - 可选：
      - LANGCHAIN_PROJECT：项目名称（未设置时可由入参 project_name 兜底）
      - LANGCHAIN_ENDPOINT：如无特殊需要，保持默认 https://api.smith.langchain.com
    代码内绝不硬编码 API Key，只读取现有环境变量，并在检测到 Key 时开启 tracing。
    """

    # 兼容两种环境变量命名
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        # 未配置 API Key 时直接返回，不做任何副作用
        logger.info("LangSmith 未配置（未检测到 LANGSMITH_API_KEY / LANGCHAIN_API_KEY），跳过观测初始化。")
        return

    # 某些 LangChain / LangSmith 版本只读取 LANGCHAIN_API_KEY，这里在进程内补齐一份，避免因为变量名差异导致不生效
    if not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = api_key

    # 启用 LangChain Tracing v2
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    # 默认为官方 LangSmith Endpoint，除非用户显式覆盖
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    # 项目名称：优先使用已有环境变量，其次使用入参
    if project_name and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = project_name

    logger.info(
        "LangSmith 观测已启用：project=%s, endpoint=%s",
        os.getenv("LANGCHAIN_PROJECT") or "(default)",
        os.getenv("LANGCHAIN_ENDPOINT"),
    )

