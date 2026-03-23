## 🚀 Obsidian-Sync-Thinker (OST)

> 一个基于 LangGraph 的本地主动式知识图谱 Agent，用于自动发现 Obsidian 笔记之间的隐性逻辑关联。

---

## ✨ 核心特性

### 🧠 多 Agent 推理

* Linker：生成跨笔记逻辑关系
* （可选）Critic：进行关联质量审查

---

### ⚡ 高性能 RAG Pipeline（重点）

系统采用分层优化的检索与推理链路：

```text
Vector Retrieval (Top-K)
↓
Metadata Hard Filter（规则过滤）
↓
Heuristic Pre-Filter（信息密度筛选）
↓
LLM Rerank（轻量精排）
↓
Context Truncation（上下文裁剪）
↓
Linker 推理
```

---

### 🚀 性能优化（核心亮点）

* **动态上下文裁剪（Context Truncation）**

  * Token 减少约 70%
  * 显著降低首字延迟（TTFT）

* **调用熔断机制（Call Budget Control）**

  * 限制单文件 Chunk 数量
  * 限制 LLM 最大调用次数

* **异步并发控制（Backpressure Control）**

  * 基于 `asyncio.Semaphore`
  * 防止任务堆积导致系统过载

---

### 🔀 多模型调度策略

```text
LOW    → Remote LLM（高性能）
MEDIUM → Local 优先 + Fallback
HIGH   → Local Only（隐私安全）
```

---

### 📊 系统可观测性（Observability）

* TraceID 全链路追踪
* LLM 调用次数统计
* 高精度耗时监控（`perf_counter`）
* 结构化日志输出（可接入 ELK）

---

### 🧱 工程特性

* 异步任务队列（`asyncio.Queue`）
* 文件监听防抖（Watchdog + Debounce）
* MD5 增量更新（避免重复推理）
* 非侵入式写回（Markdown Callout）

---

## 🏗️ 系统架构

```text
[File Watcher]
    ↓
[Debounce + Queue]
    ↓
[Worker + Backpressure]
    ↓
[Vector Store + Cache]
    ↓
[LangGraph Agent]
    ↓
[Markdown Writer]
```

---

## ⚙️ 技术栈

* Python 3.10+
* LangGraph / LangChain
* ChromaDB
* SQLite
* Watchdog
* Ollama + OpenAI-compatible API
