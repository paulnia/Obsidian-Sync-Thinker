from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from app.config import get_config
from app.core.state import KnowledgeState
from app.utils.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


def clean_llm_json(raw_text: str) -> str:
    """
    清洗 LLM 返回的字符串，尽量提取出“干净的 JSON 区段”。

    规则：
    1. 若存在 ```json ... ``` 或 ``` ... ``` 包裹，先剥离代码块标记。
    2. 在剩余文本中，定位第一个 '[' 或 '{' 作为起点，最后一个 ']' 或 '}' 作为终点，截取中间内容。
    3. 若无法可靠定位，则退化为 raw_text.strip()。
    """

    text = raw_text.strip()
    if not text:
        return text

    # 1) 去掉 ```json / ``` 代码块包装
    codeblock_re = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    m = codeblock_re.search(text)
    if m:
        text = m.group(1).strip()

    # 2) 在文本中寻找 JSON 起止位置
    start_match = re.search(r"[\[\{]", text)
    end_match = None
    # 从后往前找最后一个 ] 或 }
    for ch in ("]", "}"):
        idx = text.rfind(ch)
        if idx != -1:
            end_match = idx
            break

    if start_match and end_match is not None and end_match > start_match.start():
        return text[start_match.start() : end_match + 1].strip()

    # 3) 兜底：返回原始去空白文本
    return text

async def linker_node(state: KnowledgeState) -> dict[str, Any]:
    """
    链接生成节点（Mock LLM 版本）。

    目标：
    - 读取 `candidates`
    - 生成 2 条 `proposed_links`

    注意（LangGraph 约束）：
    - 不要原地修改 state 内的列表/字典
    - 只返回需要更新的字段（不要返回整个 state）
    """

    logger.info("进入 Linker Node")

    cfg = get_config().data
    llm = LLMFactory.get_llm(cfg.llm)

    current_chunk = state.get("current_chunk") or {}
    candidates = state.get("candidates") or []

    # Prompt：强制输出纯 JSON 数组
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个知识图谱专家。请分析当前笔记片段与提供的候选片段之间是否存在深层的逻辑关联（如：因果、递进、对比、前置条件）。"
                "你必须且只能输出纯 JSON 数组，格式为："
                '[{{\"target_file\": \"...\", \"target_header\": \"...\", \"relation_type\": \"...\", \"reason\": \"...\"}}]。'
                "不要任何 Markdown 标记，不要代码块，不要多余文字。",
            ),
            (
                "human",
                "当前片段(current_chunk):\n{current_chunk}\n\n候选片段列表(candidates):\n{candidates}\n",
            ),
        ]
    )

    try:
        msg = await (prompt | llm).ainvoke(
            {"current_chunk": json.dumps(current_chunk, ensure_ascii=False), "candidates": json.dumps(candidates, ensure_ascii=False)}
        )
        raw_text = getattr(msg, "content", "") or ""
        cleaned = clean_llm_json(raw_text)
        data = json.loads(cleaned)
        if not isinstance(data, list):
            raise ValueError("linker 输出不是 JSON 数组")
    except json.JSONDecodeError as e:
        logger.error("Linker JSON 解析失败，返回空 proposed_links。原因: %s", e)
        logger.error("解析失败的原始 LLM 输出: %s", raw_text)
        return {"proposed_links": []}
    except Exception as e:
        logger.error("Linker LLM 调用失败，返回空 proposed_links。原因: %s", e)
        try:
            logger.error("解析失败的原始 LLM 输出: %s", raw_text)
        except Exception:
            pass
        return {"proposed_links": []}

    # 转换为内部 proposed_links 格式（写入时使用）
    src_md = current_chunk.get("metadata") if isinstance(current_chunk, dict) else {}
    src_md = src_md if isinstance(src_md, dict) else {}
    src_file = str(src_md.get("file") or state.get("source_file") or "")
    src_header = str(src_md.get("header") or "")

    proposed_links: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        proposed_links.append(
            {
                "source": {"file": src_file, "header": src_header},
                "target": {"file": str(item.get("target_file") or ""), "header": str(item.get("target_header") or "")},
                "relation": str(item.get("relation_type") or ""),
                "rationale": str(item.get("reason") or ""),
            }
        )

    return {"proposed_links": proposed_links}


async def critic_node(state: KnowledgeState) -> dict[str, Any]:
    """
    审查节点（Mock 版本）。

    逻辑（按你的要求）：
    - 若 retry_count < 2：
      - 往 critique_log 追加一条意见
      - retry_count + 1
      - status 标记为 "retry"
    - 若 retry_count >= 2：
      - 视为“达到上限”，放行或打回（这里用 Mock：直接放行）
      - status 标记为 "approved"

    注意（LangGraph 约束）：
    - 不要原地修改 critique_log
    - 只返回需要更新的字段（不要返回整个 state）
    """

    logger.info("进入 Critic Node, 当前 retry_count: %s", state.get("retry_count", 0))

    cfg = get_config().data
    llm = LLMFactory.get_llm(cfg.llm)

    retry_count = int(state.get("retry_count") or 0)
    max_retries = int(state.get("max_retries") or cfg.max_retries or 2)
    critique_log = list(state.get("critique_log") or [])

    current_chunk = state.get("current_chunk") or {}
    proposed_links = state.get("proposed_links") or []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严苛的逻辑审查员。之前的 Agent 提出了一些关联建议。"
                "请审查这些关联是否真的是“强逻辑关联”，剔除仅仅是因为共同关键词的“弱关联”。"
                "你必须输出纯 JSON 对象，格式为："
                '{{\"approved\": true/false, \"feedback\": \"打回理由或通过说明\", \"valid_links\": [...]}}。'
                "不要任何 Markdown 标记，不要代码块，不要多余文字。",
            ),
            (
                "human",
                "当前片段(current_chunk):\n{current_chunk}\n\n待审查链接(proposed_links):\n{proposed_links}\n",
            ),
        ]
    )

    try:
        msg = await (prompt | llm).ainvoke(
            {
                "current_chunk": json.dumps(current_chunk, ensure_ascii=False),
                "proposed_links": json.dumps(proposed_links, ensure_ascii=False),
            }
        )
        raw_text = getattr(msg, "content", "") or ""
        cleaned = clean_llm_json(raw_text)
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            raise ValueError("critic 输出不是 JSON 对象")
    except json.JSONDecodeError as e:
        logger.error("Critic JSON 解析失败，按打回处理。原因: %s", e)
        logger.error("解析失败的原始 LLM 输出: %s", raw_text)
        obj = {"approved": False, "feedback": f"JSON 解析失败: {e}", "valid_links": []}
    except Exception as e:
        logger.error("Critic LLM 调用失败，按打回处理。原因: %s", e)
        try:
            logger.error("解析失败的原始 LLM 输出: %s", raw_text)
        except Exception:
            pass
        obj = {"approved": False, "feedback": f"LLM 调用失败: {e}", "valid_links": []}

    approved = bool(obj.get("approved"))
    feedback = str(obj.get("feedback") or "")
    valid_links = obj.get("valid_links")
    valid_links = valid_links if isinstance(valid_links, list) else []

    # 若通过，或达到重试上限：放行（把 valid_links 转成内部格式）
    if approved or retry_count >= max_retries:
        critique_log.append(feedback or "通过")

        final_links: list[dict[str, Any]] = []
        for item in valid_links:
            if isinstance(item, dict):
                # 支持两种输入：已经是内部格式，或是 linker 的简化格式
                if "source" in item and "target" in item:
                    final_links.append(item)
                    continue
                final_links.append(
                    {
                        "source": (item.get("source") or {}),
                        "target": (item.get("target") or {}),
                        "relation": str(item.get("relation") or item.get("relation_type") or ""),
                        "rationale": str(item.get("rationale") or item.get("reason") or ""),
                    }
                )

        return {"critique_log": critique_log, "proposed_links": final_links, "status": "approved"}

    # 未通过且未到上限：打回并重试
    critique_log.append(feedback or "打回：关联不够强")
    return {"critique_log": critique_log, "retry_count": retry_count + 1, "status": "retry"}
