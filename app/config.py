from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LLMConfig:
    """
    大模型配置（结构化）。

    provider:
    - "ollama": 本地 Ollama（默认）
    - "openai"/"deepseek": OpenAI 协议兼容服务
    - "mock": 调试占位
    """

    provider: str
    model: str
    base_url: str
    api_key: str
    temperature: float


@dataclass(frozen=True)
class ConfigData:
    """
    OST 配置数据（强类型）。

    说明：
    - 配置文件来自项目根目录 `config/settings.yaml`
    - 路径字段允许相对路径（相对项目根目录解析）
    """

    vault_path: str
    db_path: str
    debounce_timeout: int
    max_retries: int
    llm: LLMConfig
    # 可选：用于“混合架构”（本地 + 云端）的推理模型配置
    # - 未配置时默认与 llm 相同
    llm_reasoning: LLMConfig | None = None


class Config:
    """
    配置加载器（单例风格）。

    设计目标：
    - 读取 `config/settings.yaml`
    - 找不到配置时抛出明确异常，提示用户检查路径
    - 提供强类型访问入口，避免散落的 magic string
    """

    _instance: "Config | None" = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # 避免单例重复初始化
        if hasattr(self, "_loaded") and getattr(self, "_loaded") is True:
            return

        self._root_dir = Path(__file__).resolve().parents[1]  # 项目根目录
        self._settings_path = self._root_dir / "config" / "settings.yaml"
        self._data = self._load()
        self._loaded = True

    @property
    def data(self) -> ConfigData:
        return self._data

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def vault_dir(self) -> Path:
        """Obsidian Vault 路径（解析为绝对路径）。"""

        return self._resolve_path(self._data.vault_path)

    @property
    def chroma_dir(self) -> Path:
        """ChromaDB 持久化目录（解析为绝对路径）。"""

        return self._resolve_path(self._data.db_path)

    def _resolve_path(self, p: str) -> Path:
        """把配置中的路径解析为绝对路径（相对路径基于项目根目录）。"""

        path = Path(p).expanduser()
        if path.is_absolute():
            return path
        return (self._root_dir / path).resolve()

    def _load(self) -> ConfigData:
        """
        读取并校验 YAML 配置。

        严格要求：
        - 找不到 yaml 文件必须抛出 FileNotFoundError，并提示用户检查路径
        """

        if not self._settings_path.exists():
            raise FileNotFoundError(
                f"找不到配置文件: {self._settings_path.as_posix()}。请检查 `config/settings.yaml` 是否存在。"
            )

        raw_text = self._settings_path.read_text(encoding="utf-8", errors="replace")
        loaded: Any = yaml.safe_load(raw_text)
        if not isinstance(loaded, dict):
            raise ValueError("配置文件格式错误：settings.yaml 顶层必须是 YAML mapping（键值对）。")

        # 逐字段读取并提供默认/校验
        vault_path = str(loaded.get("vault_path") or "")
        db_path = str(loaded.get("db_path") or "")
        debounce_timeout = int(loaded.get("debounce_timeout") or 10)
        max_retries = int(loaded.get("max_retries") or 3)

        def _parse_llm(key: str) -> LLMConfig | None:
            raw = loaded.get(key)
            if raw is None:
                return None
            if not isinstance(raw, dict):
                raise ValueError(f"配置错误：{key} 必须是一个字典。示例见 `config/settings.yaml`。")

            provider = str(raw.get("provider") or "ollama").strip()
            model = str(raw.get("model") or "").strip()
            base_url = str(raw.get("base_url") or "").strip()
            api_key = str(raw.get("api_key") or "").strip()
            temperature = float(raw.get("temperature") if raw.get("temperature") is not None else 0.1)

            if not model:
                raise ValueError(f"配置缺失：{key}.model 不能为空。")
            if not base_url and provider in {"ollama", "openai", "deepseek"}:
                raise ValueError(f"配置缺失：{key}.base_url 不能为空（用于 Ollama/OpenAI 协议服务）。")

            return LLMConfig(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
            )

        llm = _parse_llm("llm")
        if llm is None:
            raise ValueError("配置缺失：llm 不能为空。")

        llm_reasoning = _parse_llm("llm_reasoning")

        if not vault_path:
            raise ValueError("配置缺失：vault_path 不能为空。")
        if not db_path:
            raise ValueError("配置缺失：db_path 不能为空。")
        if debounce_timeout <= 0:
            raise ValueError("配置错误：debounce_timeout 必须为正整数。")
        if max_retries < 0:
            raise ValueError("配置错误：max_retries 不能为负数。")
        return ConfigData(
            vault_path=vault_path,
            db_path=db_path,
            debounce_timeout=debounce_timeout,
            max_retries=max_retries,
            llm=llm,
            llm_reasoning=llm_reasoning,
        )


def get_config() -> Config:
    """获取全局配置单例。"""

    return Config()

