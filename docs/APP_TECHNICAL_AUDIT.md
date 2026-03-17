# Obsidian-Sync-Thinker (OST) — App 技术审计与架构文档

本文档对 `app` 目录下的代码进行深度审计，按模块职能、文件级 API、全链路工作流与关键技术特性进行整理。

---

## 1. 模块职能概览 (Folder Level)

| 目录 | 架构层次 | 核心目的 |
|------|----------|----------|
| **`app/`（根）** | 配置层 | 提供全局配置单例与强类型配置数据，统一 YAML 加载与路径解析。 |
| **`app/core/`** | 推理层 + 输出层 | 定义 LangGraph 状态、节点（Linker/Critic）、图构建与条件边，以及非侵入式 Markdown 回写。 |
| **`app/service/`** | 适配层 | 将 Obsidian Vault 的文件系统事件通过 Watchdog 监听、防抖后注入 asyncio 任务队列，实现 I/O 线程与异步 worker 解耦。 |
| **`app/database/`** | 数据层 | 提供 MD5 缓存（SQLite）实现增量/幂等判断，以及 ChromaDB 向量库的 chunk 写入与语义检索。 |
| **`app/utils/`** | 支撑层 | 提供 Markdown 解析/切块、LLM 工厂、日志配置等通用能力。 |

---

## 2. 文件详细说明 (File Level)

### 2.1 `app/config.py`

- **功能定位**：从 `config/settings.yaml` 加载并校验 OST 配置，提供单例访问与强类型数据结构。
- **核心组件/类**：
  - **`LLMConfig`**：大模型配置（provider / model / base_url / api_key / temperature），支持 ollama、openai/deepseek、mock。
  - **`ConfigData`**：完整配置（vault_path、db_path、debounce_timeout、max_retries、llm）。
  - **`Config`**：单例配置加载器，解析 YAML、校验必填项与取值范围，并提供 `vault_dir` / `chroma_dir` 等解析后的路径。

| 函数/API | 职责 | 关键技术点 |
|----------|------|------------|
| `get_config() -> Config` | 返回全局配置单例。 | 单例模式。 |
| `Config._load() -> ConfigData` | 读取 YAML、校验并构造 ConfigData。 | **PyYAML** `safe_load`；路径相对项目根解析。 |
| `Config._resolve_path(p: str) -> Path` | 将配置中的相对路径解析为绝对路径（基于项目根）。 | `Path.expanduser()`、相对路径拼接。 |

---

### 2.2 `app/core/state.py`

- **功能定位**：定义 LangGraph 使用的全局状态结构，约束节点“只返回增量更新、不原地修改 state”。
- **核心组件/类**：
  - **`KnowledgeState`**：TypedDict，包含 task_id、source_file、current_chunk、candidates、proposed_links、critique_log、retry_count、status、max_retries 等，供 linker/critic 与条件边使用。

- 无对外函数，仅类型定义；与 **LangGraph** 的 `StateGraph(KnowledgeState)` 配合使用。

---

### 2.3 `app/core/nodes.py`

- **功能定位**：实现 LangGraph 的两个核心节点（Linker、Critic）及 LLM 输出的 JSON 清洗工具；负责“关联建议生成”与“逻辑审查 + 重试控制”。
- **核心组件/类**：
  - **`ProposedLink`**：Pydantic 模型，表示单条关联（target_file、target_header、relation_type、reason）。
  - **`LinkerResponse`**：Pydantic 模型，包含 `proposed_links: list[ProposedLink]`，用于 **结构化输出**。
  - **`CriticResponse`**：Pydantic 模型，包含 approved、feedback、valid_links，用于 Critic 的 **结构化输出**。

| 函数/API | 职责 | 关键技术点 |
|----------|------|------------|
| `clean_llm_json(raw_text: str) -> str` | 从 LLM 原始文本中提取“干净 JSON”：剥离 \`\`\`json...\`\`\`，定位首尾 `[`/`{` 与 `]`/`}` 截取。 | **正则**：代码块、起止括号；**JSON 容错解析**的预处理。 |
| `linker_node(state: KnowledgeState) -> dict[str, Any]` | 读取 current_chunk、candidates，调用 LLM 生成 proposed_links，并转为内部格式（source/target/relation/rationale）。 | **LangChain** `ChatPromptTemplate`、**with_structured_output(LinkerResponse)**、**Pydantic**；异常时返回空 proposed_links。 |
| `critic_node(state: KnowledgeState) -> dict[str, Any]` | 审查 proposed_links，返回 approved/feedback/valid_links；未通过且 retry_count < max_retries 时 status=retry，否则 status=approved。 | **with_structured_output(CriticResponse)**；**只返回增量字段**（critique_log、proposed_links、retry_count、status）。 |

---

### 2.4 `app/core/graph.py`

- **功能定位**：组装 LangGraph 状态图，定义 linker → critic → 条件边的闭环（含“打回重试”逻辑）。
- **核心组件/类**：无独立业务类；对外暴露编译后的 `graph` 实例。

| 函数/API | 职责 | 关键技术点 |
|----------|------|------------|
| `should_continue(state: KnowledgeState) -> Literal["linker", "__end__"]` | 条件边：status==retry 且 retry_count < max_retries 时回到 linker，否则到 END。 | **LangGraph** `add_conditional_edges`；尊重 retry_count/max_retries。 |
| `build_graph() -> Any` | 构建 StateGraph(KnowledgeState)，添加 linker、critic 节点及边，编译后返回。 | **LangGraph** `StateGraph`、`set_entry_point`、`add_edge`、`add_conditional_edges`、`compile()`。 |

- **对外符号**：`graph`（`build_graph()` 的返回值），供 `main.py` 中 `graph.ainvoke(init_state)` 使用。

---

### 2.5 `app/core/writer.py`

- **功能定位**：将 AI 产出的 links 以“有标记的 Callout 区块”非侵入式写回 Markdown，支持幂等更新与旧格式清理。
- **核心组件/类**：
  - **`WriteResult`**：回写结果（filepath、ok、error）。
  - **`InsightsWriter`**：固定标记 `<!-- OST:AI-INSIGHTS:START/END -->`，渲染 Callout，幂等替换区块。

| 函数/API | 职责 | 关键技术点 |
|----------|------|------------|
| `write_links_async(filepath, links) -> WriteResult` | 异步写入：通过 `asyncio.to_thread` 调用同步写，避免阻塞事件循环。 | **asyncio.to_thread**。 |
| `_write_links_sync(filepath, links) -> None` | 读文件 → 渲染区块 → 幂等重写 → 仅当内容变化时写回。 | UTF-8 `errors="replace"`；**仅当 new_text != raw_content 时写回**。 |
| `_render_callout_block(links) -> str` | 将 links 转为 `> [!AI-Insights]` 及列表项（含 relation、target、rationale）。 | 固定 START/END 注释包裹。 |
| `_rewrite_idempotent(raw_text, ai_insight_block) -> str` | 删除旧 AI 区块（含 legacy callout 正则）、rstrip、拼接新区块。 | **正则** `_BLOCK_RE`、`_LEGACY_CALLOUT_RE`（DOTALL）；幂等、区块置尾。 |

---

### 2.6 `app/service/watcher.py`

- **功能定位**：监听 Obsidian Vault 目录的文件变更（Watchdog），按路径防抖后将 filepath 放入 asyncio.Queue，严格解耦“文件 I/O 线程”与“异步 worker”。
- **核心组件/类**：
  - **`DebounceConfig`**：防抖时长（默认 10 秒）。
  - **`ObsidianWatcher`**：持有 Observer、Queue、EventLoop、按 filepath 的 Timer 字典。

| 函数/API | 职责 | 关键技术点 |
|----------|------|------------|
| `start() -> None` | 启动 Watchdog Observer（递归监听 vault），注册 Handler（modified/created/moved）。 | **Watchdog** `Observer`、`FileSystemEventHandler`；目录不存在或导入失败时优雅不启动。 |
| `stop() -> None` | 停止 Observer、取消所有防抖 Timer、清空 _timers。 | 线程安全清理。 |
| `_on_fs_event(event) -> None` | 过滤目录事件、取路径、调用 _should_include 后防抖入队。 | 仅处理文件；**不阻塞** watchdog 线程。 |
| `_should_include(filepath: str) -> bool` | 排除 `.obsidian`、仅保留 `.md`。 | Path.parts / suffix。 |
| `_debounce_emit(filepath: str) -> None` | 按 filepath 维护 Timer，到期后通过 `loop.call_soon_threadsafe(_enqueue_filepath)` 入队。 | **threading.Timer**；**call_soon_threadsafe** 跨线程安全入队。 |
| `_enqueue_filepath(filepath: str) -> None` | 在事件循环线程中 put_nowait；队列满时 create_task(put(...))。 | **asyncio.Queue**；避免阻塞 loop。 |

---

### 2.7 `app/database/cache_manager.py`

- **功能定位**：基于 SQLite 的“文件路径 → MD5”缓存，用于增量判断：未变化的文件不重复 embed/infer。
- **核心组件/类**：
  - **`CacheRecord`**：filepath、md5、updated_at（类型/调试用）。
  - **`CacheManager`**：SQLite 表 `file_md5_cache`，提供 is_modified / update_cache。

| 函数/API | 职责 | 关键技术点 |
|----------|------|------------|
| `is_modified(filepath, current_hash) -> bool` | 缓存无此 path 或 md5 与 current_hash 不同则返回 True；相同则返回 False（跳过处理）。 | **增量更新（MD5 校验）** 的核心实现；读失败时返回 True 保证一致性。 |
| `update_cache(filepath, new_hash) -> None` | Upsert 该 filepath 的 md5 与 updated_at。 | **sqlite3**；`ON CONFLICT DO UPDATE`；写失败静默。 |
| `_connect() -> sqlite3.Connection` | 连接 SQLite，timeout=30，autocommit，WAL。 | `isolation_level=None`、PRAGMA。 |
| `_init_schema()` / `_get_md5()` | 建表（幂等）、按 filepath 查 md5。 | 内部实现。 |

---

### 2.8 `app/database/vector_store.py`

- **功能定位**：ChromaDB 持久化向量库管理，对 MarkdownHandler 产出的 chunks 做 upsert，并提供语义检索供 Linker 使用。
- **核心组件/类**：
  - **`VectorStoreResult`**：单条检索结果（content、metadata、distance、id）。
  - **`VectorStoreManager`**：ChromaDB PersistentClient、DefaultEmbeddingFunction、collection upsert/query。

| 函数/API | 职责 | 关键技术点 |
|----------|------|------------|
| `add_chunks(chunks: list[dict]) -> None` | 为每个 chunk 生成稳定 ID（file+header+content 的 MD5），upsert 到 collection。 | **ChromaDB** `upsert`；**hashlib.md5** 稳定 ID；metadata 含 file/header。 |
| `search_similar(query_text, top_k=5) -> list[dict]` | 语义检索，返回 documents/metadatas/distances/ids 组成的 list[dict]。 | **ChromaDB** `query`；防御式解析嵌套 list 结构。 |

---

### 2.9 `app/utils/md_handler.py`

- **功能定位**：Markdown 读取、YAML frontmatter 提取、按 `#`/`##` 的层级切块；提供“核心内容 MD5”（剔除 AI 区块）以配合缓存防自我触发。
- **核心组件/类**：
  - **`_Section`**：内部用，header_path + lines。
  - **`MarkdownHandler`**：解析与切块入口；含 AI 区块剔除与 frontmatter 解析。

| 函数/API | 职责 | 关键技术点 |
|----------|------|------------|
| `parse_and_chunk(filepath) -> list[dict]` | 读文件 → 剔 AI 区块 → 提 frontmatter → 按标题切块 → 返回 chunks（content + metadata.file/header/yaml）。 | **YAML** safe_load；**_strip_ai_insights**；**_chunk_by_headers**（跳过代码块内标题）。 |
| `parse_and_chunk_async(filepath) -> list[dict]` | 异步版：文件读取用 `asyncio.to_thread(_read_text_utf8)`，其余同 sync。 | **asyncio.to_thread**。 |
| `compute_core_md5_async(filepath) -> str` | 读文件 → 剔 AI 区块 → 统一换行与 rstrip → 计算 MD5（hexdigest）。 | **增量更新（MD5）** 与“防自我触发”的关键：回写 AI 区块不改变核心 MD5。 |
| `_strip_ai_insights(raw_text) -> str` | 用正则删除 `<!-- OST:AI-INSIGHTS:START -->...<!-- END -->`。 | **re.sub**、**DOTALL**；与 writer 标记一致。 |
| `_extract_frontmatter(raw_text) -> tuple[dict, str]` | 识别首行 `---` 到下一个 `---`，中间 YAML 解析为 dict；失败则 yaml_dict 含 _yaml_error。 | **yaml.safe_load**；容错。 |
| `_chunk_by_headers(body_text) -> list[_Section]` | 按 `#`/`##` 分层切分，跳过 \`\`\`/~~~ 内标题，header_path 形如 "H1" 或 "H1 > H2"。 | **正则** _H_RE、_FENCE_RE；标题行保留在 content 中。 |

---

### 2.10 `app/utils/logger.py`

- **功能定位**：配置全局 logging（控制台 + 滚动文件），避免重复添加 handler。
- **核心组件/类**：无。

| 函数/API | 职责 | 关键技术点 |
|----------|------|------------|
| `setup_logging(log_path=None) -> None` | 若未配置过，则设置 root logger、StreamHandler、RotatingFileHandler（10MB×5），统一格式。 | **logging.handlers.RotatingFileHandler**；`_ost_configured` 防重复。 |

---

### 2.11 `app/utils/llm_factory.py`

- **功能定位**：根据 `LLMConfig` 的 provider 返回对应的 LangChain ChatModel（Ollama / OpenAI 兼容 / mock 抛错）。
- **核心组件/类**：
  - **`LLMFactory`**：静态方法 `get_llm(config)`。

| 函数/API | 职责 | 关键技术点 |
|----------|------|------------|
| `LLMFactory.get_llm(config: LLMConfig) -> Any` | provider=ollama → ChatOllama；openai/deepseek → ChatOpenAI；mock → NotImplementedError。 | **langchain_ollama**、**langchain_openai**；Ollama 的 format="json" 存在时使用、否则降级。 |

---

## 3. 全链路工作流分析（The "Life of a Request"）

从“监听到 Obsidian 文件变动”到“最终修改回写”的完整生命周期如下。

### 3.1 事件产生与入队（适配层）

1. **Watchdog**（`app/service/watcher.py`）在独立线程中监听 Vault 目录的 `modified` / `created` / `moved` 事件。
2. **`_on_fs_event`** 过滤掉目录、非 `.md`、`.obsidian` 下文件，得到 `filepath`。
3. **`_debounce_emit(filepath)`** 按路径防抖：同一 path 在 `debounce_timeout` 秒内多次变动只保留最后一次；Timer 到期后在 **事件循环线程** 中执行 **`_enqueue_filepath(filepath)`**，将 `filepath` 放入 **`asyncio.Queue[str]`**。
4. 数据流：**文件系统事件 → filepath 字符串 → Queue**。此处完成 **I/O 线程与 asyncio 的解耦**。

### 3.2 队列消费与预处理（main.py + 数据层/支撑层）

5. **`worker_loop(queue)`**（`main.py`）从 Queue 中 `await queue.get()` 得到 `filepath`。
6. **MD5 幂等**：
   - 调用 **`MarkdownHandler.compute_core_md5_async(filepath)`**（`app/utils/md_handler.py`）：读文件 → **剔除 AI-Insights 区块** → 统一换行 → 计算 MD5。
   - 调用 **`CacheManager.is_modified(filepath, current_md5)`**（`app/database/cache_manager.py`）：若缓存中 MD5 与当前一致，**直接 continue**，不进入后续步骤。
7. **解析与切块**：**`MarkdownHandler.parse_and_chunk_async(filepath)`** → 得到 **chunks**（每个含 content、metadata.file/header/yaml）。
8. **向量库更新**：**`VectorStoreManager.add_chunks(chunks)`**，对当前文件 chunks 做 **upsert**（稳定 ID = file+header+content 的 MD5）。

数据流：**filepath → 核心 MD5 → 缓存比较 → chunks → 向量库**。

### 3.3 LangGraph 推理（推理层）

9. 对 **每个 chunk**：
   - 用 `chunk["content"]` 作为 query，**`vector_store.search_similar(query_text, top_k=5)`** 得到 **candidates**。
   - 构造 **`KnowledgeState`**：task_id、source_file、**current_chunk**、**candidates**、proposed_links=[]、critique_log=[]、retry_count=0、max_retries。
   - **`await graph.ainvoke(init_state)`**（`app/core/graph.py` 的 `graph`）进入 LangGraph。

10. **图内数据流与闭环**：
    - **入口**：**linker** 节点。
    - **linker_node(state)**（`app/core/nodes.py`）：用 LLM + **with_structured_output(LinkerResponse)** 从 current_chunk + candidates 生成 **proposed_links**，写入 state（仅返回 `{"proposed_links": proposed_links}`）。
    - **边**：linker → **critic**（固定边）。
    - **critic_node(state)**：用 LLM + **with_structured_output(CriticResponse)** 审查 proposed_links，得到 approved、feedback、valid_links；若 **approved 或 retry_count >= max_retries**，则 status=**approved**，并令 proposed_links = final_links；否则 status=**retry**，retry_count+1，critique_log 追加 feedback。
    - **条件边** **`should_continue(state)`**：若 status==**retry** 且 retry_count < max_retries → 回到 **linker**；否则 → **END**。
    - 因此 **Linker 与 Critic 的闭环** 由 LangGraph 的 **conditional_edges(critic, should_continue, {"linker": "linker", "__end__": END})** 实现：打回则重跑 linker，通过或达重试上限则结束。

11. **出图后**：`out_state["proposed_links"]` 即为审查后的 links；main 中把所有 chunk 的 links 合并为 **final_links**。

数据流：**state（current_chunk, candidates）→ linker → proposed_links → critic → approved/retry → 条件回到 linker 或 END → 最终 proposed_links**。

### 3.4 回写与缓存更新（输出层 + 数据层）

12. **`InsightsWriter.write_links_async(filepath, final_links)`**（`app/core/writer.py`）：通过 **asyncio.to_thread** 执行 **`_write_links_sync`**，读文件 → **幂等替换** AI-Insights 区块（正则删旧块再拼新块）→ 仅当内容变化时写回。
13. **`CacheManager.update_cache(filepath, current_md5)`**：将当前“核心内容 MD5”写入 SQLite，下次同一文件未改动时 **is_modified** 为 False，避免重复处理。

数据流：**final_links → Callout 区块 → 幂等写回文件；filepath + current_md5 → 缓存**。

### 3.5 小结图（数据与节点关系）

```
[Watchdog] → filepath → [Queue]
                            ↓
[worker_loop] ← filepath
    ↓
[MD5 核心] → [CacheManager.is_modified] → 未变则跳过
    ↓
[MarkdownHandler.parse_and_chunk_async] → chunks
    ↓
[VectorStoreManager.add_chunks(chunks)]
    ↓
for each chunk:
    candidates = VectorStoreManager.search_similar(chunk.content)
    init_state = { current_chunk, candidates, ... }
    out_state = await graph.ainvoke(init_state)
    final_links += out_state["proposed_links"]
    ↓
[InsightsWriter.write_links_async(filepath, final_links)]
[CacheManager.update_cache(filepath, current_md5)]
```

**LangGraph 内部**：  
**linker** → **critic** → **should_continue** → linker（retry）或 **END**（approved/达上限）。

---

## 4. 关键技术特性

### 4.1 增量更新（MD5 校验）

- **目的**：文件内容未变化时不重复做向量化与推理，避免无效负载与“写回导致再次触发”的放大。
- **实现位置**：
  - **`app/database/cache_manager.py`**：**`is_modified(filepath, current_hash)`** 比较缓存中的 MD5 与当前传入的 hash；**`update_cache(filepath, new_hash)`** 在成功处理后将新 hash 写入 SQLite。
  - **`app/utils/md_handler.py`**：**`compute_core_md5_async(filepath)`** 计算“核心内容”的 MD5：先 **`_strip_ai_insights(raw_text)`** 再统一换行与 rstrip 后做 MD5。这样 **Writer 回写 AI-Insights 区块不会改变核心内容 MD5**，缓存可正确判断“内容未变”，从而避免自我触发死循环。
- **调用链**：`main.py` 中先 `current_md5 = await md_handler.compute_core_md5_async(filepath)`，再 `if not cache_manager.is_modified(filepath, current_md5): continue`；处理结束后 `cache_manager.update_cache(filepath, current_md5)`。

### 4.2 JSON 容错解析

- **目的**：从 LLM 可能带 Markdown 代码块或前后缀文字的返回中，可靠提取 JSON 字符串。
- **实现位置**：**`app/core/nodes.py`** 中的 **`clean_llm_json(raw_text: str) -> str`**。
  - 规则：若存在 \`\`\`json...\`\`\` 或 \`\`\`...\`\`\`，先剥离代码块；在剩余文本中定位第一个 `[` 或 `{` 与最后一个 `]` 或 `}`，截取该区段；无法定位则退回 `raw_text.strip()`。
- **当前用法**：Linker/Critic 节点使用 **LangChain 的 `with_structured_output(LinkerResponse/CriticResponse)`**（Pydantic 结构化输出），由框架负责解析；**`clean_llm_json`** 作为备用工具，可在未来若改用“原始文本 + 手工解析”时用于预处理，或作为 fallback 管线中的一步。

### 4.3 结构化输出

- **目的**：让 LLM 输出固定 schema，便于程序消费与类型安全。
- **实现位置**：**`app/core/nodes.py`**。
  - **Pydantic 模型**：**`ProposedLink`**、**`LinkerResponse`**、**`CriticResponse`**（含 approved、feedback、valid_links）。
  - **调用方式**：`llm.with_structured_output(LinkerResponse)` / `CriticResponse`，再 `(prompt | structured_llm).ainvoke(...)`，得到 Pydantic 实例；异常时 linker 返回空 proposed_links，critic 按打回处理并记录日志。

### 4.4 其他设计要点

- **AI 区块与切块/MD5 一致**：**`MarkdownHandler._strip_ai_insights`** 与 **`InsightsWriter`** 使用的 `<!-- OST:AI-INSIGHTS:START/END -->` 正则一致，保证“切块与 MD5 均不包含回写区块”，避免内容漂移与误判修改。
- **幂等回写**：**`InsightsWriter._rewrite_idempotent`** 先删旧 AI 区块再拼接新块，且 **仅当 new_text != raw_content 时写文件**，减少不必要的磁盘与再次触发。
- **线程与 asyncio 解耦**：**Watchdog 回调** 仅做过滤与防抖，通过 **`loop.call_soon_threadsafe(_enqueue_filepath)`** 将入队操作放到事件循环线程；**worker_loop** 纯异步消费，**文件读/写** 通过 **`asyncio.to_thread`** 放入线程池，不阻塞 loop。

---

以上为 `app` 目录的模块职能、文件级 API、全链路工作流与关键技术特性的审计与文档化整理。若后续增删模块或接口，可在此文档基础上做增量更新。
