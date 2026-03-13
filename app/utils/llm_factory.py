from __future__ import annotations

import logging
from typing import Any

from app.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    LLM 工厂：根据配置返回对应的 ChatModel 实例。

    支持：
    - ollama: langchain_community.chat_models.ChatOllama
    - openai/deepseek: langchain_openai.ChatOpenAI（兼容 OpenAI 协议的 provider）
    - mock: 显式抛出（避免误用）
    """

    @staticmethod
    def get_llm(config: LLMConfig) -> Any:
        provider = (config.provider or "").strip().lower()

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            # 强制 JSON：不同版本参数名可能不同，这里做兼容降级
            try:
                return ChatOllama(
                    base_url=config.base_url,
                    model=config.model,
                    temperature=config.temperature,
                    format="json",
                )
            except TypeError:
                logger.info("当前 ChatOllama 不支持 format='json' 参数，降级为纯提示词约束 JSON。")
                return ChatOllama(
                    base_url=config.base_url,
                    model=config.model,
                    temperature=config.temperature,
                )

        if provider in {"openai", "deepseek"}:
            from langchain_openai import ChatOpenAI

            # OpenAI/DeepSeek 等：统一走 ChatOpenAI
            # 说明：response_format 的 JSON 强制在不同版本中支持差异较大，这里同样以提示词为主。
            return ChatOpenAI(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                temperature=config.temperature,
            )

        if provider == "mock":
            raise NotImplementedError("provider=mock 已弃用：请在 config/settings.yaml 中配置真实 provider。")

        raise ValueError(f"未知 llm.provider: {config.provider}")

