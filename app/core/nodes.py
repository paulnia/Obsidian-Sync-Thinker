from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import get_config
from app.core.state import KnowledgeState
from app.utils.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ProposedLink(BaseModel):
    target_file: str = Field(description="目标文件名")
    target_header: str = Field(description="目标小标题")
    relation_type: str = Field(description="逻辑关联类型，如：因果、递进、对比")
    reason: str = Field(description="建立此关联的详细理由")


class LinkerResponse(BaseModel):
    proposed_links: list[ProposedLink] = Field(default_factory=list, description="关联建议列表，无关联时为空列表")


class CriticResponse(BaseModel):
    approved: bool = Field(description="是否通过审查")
    feedback: str = Field(description="打回理由或通过说明")
    valid_links: list[dict[str, Any]] = Field(default_factory=list, description="审查后保留的链接，与 proposed_links 格式一致")


def clean_llm_json(raw_text: str) -> str:
    """
    清洗 LLM 返回的字符串，尽量提取出“干净的 JSON 区段”。

    规则：
    1. 若存在 ```json ... ``` 或 ``` ... ``` 包裹，先剥离代码块标记。
    2. 在剩余文本中，定位第一个 '[' 或 '{' 作为起点，最后一个 ']' 或 '}' 作为终点，截取中间内容。
    3. 若无法可靠定位，则退化为 raw_text.strip()。
    """

    text = str(raw_text or "").strip()
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
        core = text[start_match.start() : end_match + 1]
    else:
        core = text

    # 3) 进一步宽松修正：
    # - 全角逗号/顿号替换为半角逗号
    # - 删除行尾多余逗号（在 ] 或 } 前）
    core = core.replace("，", ",").replace("、", ",")
    core = re.sub(r",\s*(\]|\})", r"\1", core)

    return core.strip()


async def _ainvoke_structured_with_fallback(
    prompt: ChatPromptTemplate,
    llm: Any,
    schema_model: type[BaseModel],
    payload: dict[str, Any],
    *,
    who: str,
) -> BaseModel:
    """
    尝试使用 LangChain 的 structured_output；若 provider/模型返回了代码块包裹 JSON 等非严格格式，
    则自动回退为“原始文本 + clean_llm_json + json.loads + Pydantic 校验”。
    """

    # 1) 优先走 structured_output（更稳定、更省事）
    try:
        structured_llm = llm.with_structured_output(schema_model)  # type: ignore[attr-defined]
        return await (prompt | structured_llm).ainvoke(payload)
    except Exception as e:
        logger.warning("%s structured_output 失败，尝试回退到手工 JSON 解析。原因: %s", who, e)

    # 2) 回退：拿原始文本，清洗后按 JSON 解析，再用 Pydantic 校验
    raw = await (prompt | llm).ainvoke(payload)
    raw_text = getattr(raw, "content", None)
    if raw_text is None:
        raw_text = str(raw)

    cleaned = clean_llm_json(str(raw_text))

    # 尝试严格 JSON 解析；失败时记录原文片段，便于诊断
    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        logger.error(
            "%s 回退 JSON 解析仍失败，将抛出异常。cleaned_snippet=%r, error=%s",
            who,
            cleaned[:200],
            e,
        )
        raise

    return schema_model.model_validate(parsed)

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
    llm_cfg = cfg.llm_reasoning or cfg.llm
    llm = LLMFactory.get_llm(llm_cfg)

    current_chunk = state.get("current_chunk") or {}
    candidates = state.get("candidates") or []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严格遵守 JSON Schema 的知识图谱专家。请分析当前笔记片段与提供的候选片段之间是否存在深层的逻辑关联（如：因果、递进、对比、前置条件）。\n\n"
                "【输出硬性要求】\n"
                "1. 只能输出一个 JSON 对象，且必须完全符合以下 Schema：\n"
                "   {{\n"
                '     \"proposed_links\": [\n'
                "       {{\n"
                '         \"target_file\": \"string\",\n'
                '         \"target_header\": \"string\",\n'
                '         \"relation_type\": \"string\",\n'
                '         \"reason\": \"string\"\n'
                "       }}\n"
                "     ]\n"
                "   }}\n"
                "2. 严禁输出任何 Markdown、``` 代码块包裹、注释、额外字段或多余文本；\n"
                "3. 即使没有找到任何关联，也必须返回 {{\"proposed_links\": []}}；\n"
                "4. 字段名必须完全匹配，不能增加、删除或改名。\n\n"
                "输出示例：\n"
                "  - 有关联时：{{\"proposed_links\": [{{\"target_file\": \"foo.md\", \"target_header\": \"示例\", \"relation_type\": \"因果\", \"reason\": \"...\"}}]}}\n"
                "  - 无关联时：{{\"proposed_links\": []}}\n",
            ),
            (
                "human",
                "当前片段(current_chunk):\n{current_chunk}\n\n候选片段列表(candidates):\n{candidates}\n",
            ),
        ]
    )

    try:
        result = await _ainvoke_structured_with_fallback(
            prompt,
            llm,
            LinkerResponse,
            {
                "current_chunk": json.dumps(current_chunk, ensure_ascii=False),
                "candidates": json.dumps(candidates, ensure_ascii=False),
            },
            who="Linker",
        )
        result = result if isinstance(result, LinkerResponse) else LinkerResponse.model_validate(result)
    except Exception as e:
        logger.error("Linker 结构化提取失败，返回空 proposed_links。原因: %s", e)
        return {"proposed_links": []}

    n = len(result.proposed_links)
    logger.info("Linker 完成，生成 %d 条 proposed_links", n)
    for i, item in enumerate(result.proposed_links):
        logger.debug("  [%d] target=%s#%s relation=%s", i + 1, item.target_file, item.target_header, item.relation_type)

    src_md = current_chunk.get("metadata") if isinstance(current_chunk, dict) else {}
    src_md = src_md if isinstance(src_md, dict) else {}
    src_file = str(src_md.get("file") or state.get("source_file") or "")
    src_header = str(src_md.get("header") or "")

    proposed_links: list[dict[str, Any]] = []
    for item in result.proposed_links:
        proposed_links.append(
            {
                "source": {"file": src_file, "header": src_header},
                "target": {"file": item.target_file, "header": item.target_header},
                "relation": item.relation_type,
                "rationale": item.reason,
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
    llm_cfg = cfg.llm_reasoning or cfg.llm
    llm = LLMFactory.get_llm(llm_cfg)

    retry_count = int(state.get("retry_count") or 0)
    max_retries = int(state.get("max_retries") or cfg.max_retries or 2)
    critique_log = list(state.get("critique_log") or [])

    current_chunk = state.get("current_chunk") or {}
    proposed_links = state.get("proposed_links") or []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严苛的逻辑审查员。之前的 Agent 提出了一些关联建议。\n"
                "请审查这些关联是否真的是“强逻辑关联”，剔除仅仅是因为共同关键词的“弱关联”。\n\n"
                "【输出硬性要求】\n"
                "1. 只能输出一个 JSON 对象，且必须完全符合以下 Schema：\n"
                "   {{\n"
                '     \"approved\": true 或 false,\n'
                '     \"feedback\": \"string\",\n'
                '     \"valid_links\": [\n'
                "       {{\n"
                "         // 与传入的 proposed_links 格式一致，或已经是内部格式 {{source, target, relation, rationale}}\n"
                "       }}\n"
                "     ]\n"
                "   }}\n"
                "2. 严禁输出任何 Markdown、``` 代码块包裹、注释、额外字段或多余文本；\n"
                "3. 若无待审查链接或全部打回，valid_links 可为空数组 []；\n"
                "4. 字段名必须完全匹配，不能增加、删除或改名。\n",
            ),
            (
                "human",
                "当前片段(current_chunk):\n{current_chunk}\n\n待审查链接(proposed_links):\n{proposed_links}\n",
            ),
        ]
    )

    try:
        result = await _ainvoke_structured_with_fallback(
            prompt,
            llm,
            CriticResponse,
            {
                "current_chunk": json.dumps(current_chunk, ensure_ascii=False),
                "proposed_links": json.dumps(proposed_links, ensure_ascii=False),
            },
            who="Critic",
        )
        result = result if isinstance(result, CriticResponse) else CriticResponse.model_validate(result)
        approved = bool(result.approved)
        feedback = str(result.feedback)
        valid_links = result.valid_links if isinstance(result.valid_links, list) else []
    except Exception as e:
        logger.error("Critic 结构化提取失败，按打回处理。原因: %s", e)
        approved = False
        feedback = str(e)
        valid_links = []

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
