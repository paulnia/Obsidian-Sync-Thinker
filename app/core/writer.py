from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class WriteResult:
    """回写结果（便于上层统计/调试）。"""

    filepath: str
    ok: bool
    error: str | None = None


class InsightsWriter:
    """
    非侵入式回写器：把 AI 产出的 links 以 Callout 形式写入 Markdown。

    安全策略：
    - 不覆盖整篇文件内容
    - 使用“有标记的区块”进行可重复更新（避免重复追加）
    - 使用 UTF-8 写入，并保留文件原有内容
    """

    _START = "<!-- OST:AI-INSIGHTS:START -->"
    _END = "<!-- OST:AI-INSIGHTS:END -->"
    _BLOCK_RE = re.compile(
        r"<!--\s*OST:AI-INSIGHTS:START\s*-->.*?<!--\s*OST:AI-INSIGHTS:END\s*-->\s*",
        re.DOTALL,
    )
    # 兼容“旧格式/无标记”callout：移除从 > [!AI-Insights] 开始的连续引用行
    _LEGACY_CALLOUT_RE = re.compile(r"(?m)^\s*>\s*\[!AI-Insights\]\s*$\n(?:^\s*>.*$\n?)*")

    async def write_links_async(self, filepath: str, links: list[dict[str, Any]]) -> WriteResult:
        """
        异步写入：通过 `asyncio.to_thread` 把实际磁盘 I/O 放到线程池，避免阻塞事件循环。
        """

        try:
            await asyncio.to_thread(self._write_links_sync, filepath, links)
            return WriteResult(filepath=filepath, ok=True)
        except Exception as e:
            return WriteResult(filepath=filepath, ok=False, error=str(e))

    def _write_links_sync(self, filepath: str, links: list[dict[str, Any]]) -> None:
        """
        同步写入实现（供 to_thread 调用）。
        """

        p = Path(filepath)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(filepath)

        # 读取全文（UTF-8，编码异常时替换字符，避免崩溃）
        raw_content = p.read_text(encoding="utf-8", errors="replace")

        block = self._render_callout_block(links)
        new_text = self._rewrite_idempotent(raw_content, block)

        # 仅当内容发生变化时才写回，减少无意义的文件变动
        if new_text != raw_content:
            p.write_text(new_text, encoding="utf-8", newline="\n")

    def _render_callout_block(self, links: list[dict[str, Any]]) -> str:
        """
        把 links 渲染为 Obsidian Callout + Todo 列表：
        > [!AI-Insights]
        > - [ ] [AI 建议] [[file#header]] - 理由: ...

        约定：
        - 初次写入时一律为未勾选 `[ ]`，表示“待人工确认”
        - 若用户在 Obsidian 中改为 `[x]`，即视为已采纳；若删除该行，则视为拒绝
        """

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines: list[str] = []
        lines.append(self._START)
        lines.append("> [!AI-Insights]")
        lines.append(f"> 更新时间: {now}")

        if not links:
            lines.append("> - （无）")
        else:
            for link in links:
                src = link.get("source") if isinstance(link, dict) else {}
                tgt = link.get("target") if isinstance(link, dict) else {}
                src = src if isinstance(src, dict) else {}
                tgt = tgt if isinstance(tgt, dict) else {}

                rel = str(link.get("relation") or "关联")
                rationale = str(link.get("rationale") or "")
                tgt_file = str(tgt.get("file") or "")
                tgt_header = str(tgt.get("header") or "")

                # 以 Todo 列表形式呈现 AI 建议，使用 Obsidian 支持的 task 语法
                # 例如：- [ ] [AI 建议] [[Dijkstra算法#复杂度分析]] - 理由: ...
                if tgt_file or tgt_header:
                    link_label = f"{tgt_file}#{tgt_header}" if tgt_header else tgt_file
                    display = f"[[{link_label}]]"
                else:
                    display = "(未知目标)"

                item = f"> - [ ] [AI 建议] {display} - **{rel}**"
                lines.append(item)
                if rationale:
                    lines.append(f">   - 理由: {rationale}")

        lines.append(self._END)
        return "\n".join(lines).strip() + "\n"

    def _rewrite_idempotent(self, raw_text: str, ai_insight_block: str) -> str:
        """
        幂等回写（按用户指定的固定步骤）：
        1) 读取全量 raw_content（入参 raw_text）
        2) 使用正则彻底删除旧的 AI 区块（START..END，DOTALL）
        3) 对 clean_content 做 rstrip()，清除末尾空行
        4) 拼接最新区块并覆写：
           new_content = clean_content + "\\n\\n" + ai_insight_block + "\\n"

        说明：
        - 这样可以防止无限叠加
        - 也能保证 AI 区块始终位于文件末尾
        """

        # 1) 删除旧区块（支持跨行 DOTALL）
        clean_content = re.sub(self._BLOCK_RE, "", raw_text)
        clean_content = re.sub(self._LEGACY_CALLOUT_RE, "", clean_content)

        # 2) 清理末尾空行
        clean_content = clean_content.rstrip()

        # 3) 拼接新块并覆写（确保末尾换行稳定）
        new_content = clean_content + "\n\n" + ai_insight_block.rstrip() + "\n"
        return new_content

