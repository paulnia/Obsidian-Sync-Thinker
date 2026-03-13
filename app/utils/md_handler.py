from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

import yaml

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _Section:
    """内部数据结构：用于在切分过程中暂存分段内容。"""

    header_path: str  # 形如："" / "一级标题" / "一级标题 > 二级标题"
    lines: list[str]  # 该段落的正文行（可包含标题行本身）


class MarkdownHandler:
    """
    Markdown 底层处理器：
    - 读取 Markdown 文件（UTF-8）
    - 提取 YAML frontmatter（若存在）
    - 按 `#` / `##` 标题做分层（hierarchical）Smart Chunking

    注意：
    - 为了“智能”地切块，本实现会避免在代码块（``` 或 ~~~）内部误识别标题。
    - 切块结果会保留文件路径、标题路径、YAML 字典到 metadata 中，满足 Obsidian 场景检索与回写的需求。
    """

    _H_RE = re.compile(r"^(#{1,2})\s+(?P<title>.+?)\s*$")
    _FENCE_RE = re.compile(r"^\s*(```|~~~)")
    # 关键：用于剔除 AI 回写区块，避免“程序自我触发”导致 MD5 永远变化
    _AI_BLOCK_RE = re.compile(
        r"<!--\s*OST:AI-INSIGHTS:START\s*-->.*?<!--\s*OST:AI-INSIGHTS:END\s*-->\s*",
        re.DOTALL,
    )

    def parse_and_chunk(self, filepath: str) -> list[dict[str, Any]]:
        """
        读取并切分 Markdown 文件。

        参数：
        - filepath: Markdown 文件绝对或相对路径

        返回：
        - chunks: list[dict]，每个元素结构固定为：
          {
            "content": "...",
            "metadata": {
              "file": filepath,
              "header": "标题名(可为层级路径)",
              "yaml": {...}
            }
          }

        编码说明：
        - 按需求使用 UTF-8 读取；如遇到异常字符，使用 errors="replace" 以“优雅降级”方式继续处理。
        """

        # 1) 读取文件内容（UTF-8）；遵循“优雅降级”原则，尽量不因单个文件编码问题中断整个流水线。
        try:
            raw_text = self._read_text_utf8(filepath)
        except OSError as e:
            # 文件无法读取时，返回空切块；上层可据此决定是否重试/记录日志。
            # 这里不抛异常，避免阻塞 Watchdog / 队列消费者等异步任务链路。
            return [
                {
                    "content": "",
                    "metadata": {"file": filepath, "header": "", "yaml": {"_error": str(e)}},
                }
            ]

        # 关键修复：切块前先剔除 AI-Insights 区块，保证切块只基于“笔记核心内容”
        raw_text = self._strip_ai_insights(raw_text)

        # 2) 提取 YAML frontmatter（仅当文件起始处存在 '---' 分隔时视为 frontmatter）
        yaml_dict, body_text = self._extract_frontmatter(raw_text)

        # 3) 分层切块（# / ##），并生成最终 chunks
        sections = self._chunk_by_headers(body_text)
        chunks: list[dict[str, Any]] = []

        for sec in sections:
            content = "".join(sec.lines).strip()
            # 过滤掉“完全空”的分段，避免产生无意义 chunk
            if not content:
                continue

            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        "file": filepath,
                        # header 字段要求保留标题文本；这里使用“层级路径”可同时保留 # 与 ## 的上下文。
                        "header": sec.header_path,
                        "yaml": yaml_dict,
                    },
                }
            )

        # 若没有任何标题、也没有正文，则返回一个空 chunk（仍携带 yaml 元数据），便于上层缓存/幂等判断。
        if not chunks:
            chunks.append(
                {
                    "content": body_text.strip(),
                    "metadata": {"file": filepath, "header": "", "yaml": yaml_dict},
                }
            )

        logger.info("成功将文件拆分为 %s 个 Chunk", len(chunks))
        return chunks

    async def parse_and_chunk_async(self, filepath: str) -> list[dict[str, Any]]:
        """
        异步版本（推荐在整体异步架构中使用）。

        说明：
        - `.cursorrules` 要求 I/O 尽量走 `asyncio`，因此这里提供 async 版本：
          - 文件读取通过 `asyncio.to_thread(...)` 放到线程池，避免阻塞事件循环。
          - 其余解析/切分是纯 CPU 文本处理，保持同步执行即可。
        """

        # 将“阻塞式文件读取”放到线程池执行，避免阻塞 async 事件循环
        try:
            raw_text = await asyncio.to_thread(self._read_text_utf8, filepath)
        except OSError as e:
            return [
                {
                    "content": "",
                    "metadata": {"file": filepath, "header": "", "yaml": {"_error": str(e)}},
                }
            ]

        # 关键修复：切块前先剔除 AI-Insights 区块，保证切块只基于“笔记核心内容”
        raw_text = self._strip_ai_insights(raw_text)

        yaml_dict, body_text = self._extract_frontmatter(raw_text)
        sections = self._chunk_by_headers(body_text)

        chunks: list[dict[str, Any]] = []
        for sec in sections:
            content = "".join(sec.lines).strip()
            if not content:
                continue
            chunks.append(
                {
                    "content": content,
                    "metadata": {"file": filepath, "header": sec.header_path, "yaml": yaml_dict},
                }
            )

        if not chunks:
            chunks.append(
                {"content": body_text.strip(), "metadata": {"file": filepath, "header": "", "yaml": yaml_dict}}
            )

        logger.info("成功将文件拆分为 %s 个 Chunk", len(chunks))
        return chunks

    async def compute_core_md5_async(self, filepath: str) -> str:
        """
        计算“核心内容 MD5”（异步）。

        关键点：
        - 必须在 MD5 计算前，使用正则彻底剔除 AI-Insights 区块
        - 这样 writer 回写不会改变核心内容 MD5，从而让缓存系统拦截自我触发
        """

        def _sync() -> str:
            raw = self._read_text_utf8(filepath)
            normalized = self._strip_ai_insights(raw)
            # 统一换行与尾部空白，避免不同平台写入造成的 MD5 噪声
            normalized = normalized.replace("\r\n", "\n").rstrip() + "\n"
            return hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()

        return await asyncio.to_thread(_sync)

    def _read_text_utf8(self, filepath: str) -> str:
        """以 UTF-8 读取文本（errors='replace'），供 sync/async 两种入口复用。"""

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _strip_ai_insights(self, raw_text: str) -> str:
        """
        剔除 AI 回写区块（START..END 之间所有内容）。

        严格要求（用户指定）：
        - 必须使用 `re.sub` + `re.DOTALL`，确保跨行内容被彻底移除
        """

        return re.sub(self._AI_BLOCK_RE, "", raw_text)

    def _extract_frontmatter(self, raw_text: str) -> tuple[dict[str, Any], str]:
        """
        提取 YAML frontmatter，并返回 (yaml_dict, markdown_body)。

        frontmatter 规则（Obsidian 常见约定）：
        - 文件开头第一行必须是 '---'
        - 直到下一行出现独立的 '---' 结束
        - 中间内容视为 YAML
        """

        lines = raw_text.splitlines(keepends=True)
        if not lines:
            return {}, ""

        # frontmatter 仅在文件第一行出现 '---' 时生效（严格一些，避免把正文中的分隔线误当作 YAML）
        if lines[0].strip() != "---":
            return {}, raw_text

        # 查找结束的 '---'
        end_idx: int | None = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break

        # 未找到结束分隔线：视为无效 frontmatter，整文件按正文处理
        if end_idx is None:
            return {}, raw_text

        yaml_text = "".join(lines[1:end_idx])
        body_text = "".join(lines[end_idx + 1 :])

        # YAML 解析：失败则优雅降级为 {}
        try:
            loaded = yaml.safe_load(yaml_text)
            yaml_dict = loaded if isinstance(loaded, dict) else {}
        except Exception as e:
            # 将错误信息塞进 yaml 字典，方便上层观察/记录，但不阻塞处理流程
            yaml_dict = {"_yaml_error": str(e)}

        return yaml_dict, body_text

    def _chunk_by_headers(self, body_text: str) -> list[_Section]:
        """
        按 Markdown 标题 `#` / `##` 进行分层切分。

        Smart 规则：
        - 跳过 fenced code block（``` 或 ~~~）内的标题识别
        - `#` 开新一级章节
        - `##` 开新二级章节，header_path 形如："{H1} > {H2}"
        - 标题行会被保留在 chunk content 的开头，方便向量化/检索时保留语义锚点
        """

        lines = body_text.splitlines(keepends=True)
        sections: list[_Section] = []

        current_h1: str = ""
        current_header_path: str = ""
        current_lines: list[str] = []

        in_fence = False

        def flush() -> None:
            """把当前缓存的段落落盘到 sections。"""
            nonlocal current_lines
            if current_lines:
                sections.append(_Section(header_path=current_header_path, lines=current_lines))
                current_lines = []

        for line in lines:
            # fenced code block 开关：遇到 ``` 或 ~~~ 行就切换状态
            if self._FENCE_RE.match(line):
                in_fence = not in_fence
                current_lines.append(line)
                continue

            # 在代码块内部，不识别标题，直接累积内容
            if in_fence:
                current_lines.append(line)
                continue

            m = self._H_RE.match(line)
            if not m:
                current_lines.append(line)
                continue

            level = len(m.group(1))
            title = (m.group("title") or "").strip()

            # 遇到新标题：先把上一段落 flush，再开始新段落
            flush()

            if level == 1:
                current_h1 = title
                current_header_path = title
            else:
                # level == 2
                # 若没有 H1，上层标题留空，仍保留 H2 文本
                current_header_path = f"{current_h1} > {title}".strip(" >")

            # 标题行本身也放进 chunk content，增强语义完整性
            current_lines.append(line)

        # 文件结束：flush 最后一个段落
        flush()

        # 如果全文没有任何标题，sections 可能只有一个段落或为空；
        # 这里保持 sections 原样返回，由上层决定是否构造兜底 chunk。
        return sections
