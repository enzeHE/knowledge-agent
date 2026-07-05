# Knowledge Agent - 开发者文档智能问答系统

基于 **LangGraph + FastAPI + RAG** 的个人全栈知识助手，支持 FastAPI / LangChain / LangGraph 技术文档智能问答。

> 求职方向：Agent 开发 / Agent 应用

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client                                   │
│           (Gradio UI / curl / Postman)                          │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP/SSE
┌────────────────────▼────────────────────────────────────────────┐
│                     FastAPI Server                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Chat API (SSE Streaming)     Document API               │   │
│  └────────────────────┬─────────────────────────────────────┘   │
└───────────────────────┼─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│              LangGraph Agent — 7-Node StateGraph                │
│                                                                  │
│  ┌────────────┐     ┌──────────────┐                            │
│  │Input Guard │────▶│Intent Router │──┐                         │
│  │(敏感词/空   │     │(retrieve/    │  │                         │
│  │查询/过长)   │     │ clarify/     │  │                         │
│  └────────────┘     │ general)     │  │                         │
│                     └──────────────┘  │                         │
│                           │           │                         │
│              ┌────────────┘    ┌──────┘                         │
│              ▼                 ▼                                │
│  ┌──────────────────┐  ┌──────────────┐                        │
│  │  Doc Retriever   │  │Clarify/General│                        │
│  │(Rewrite→Retrieve │  │  Handler     │                        │
│  │ →Rerank)         │  └──────┬───────┘                        │
│  └────────┬─────────┘         │                                │
│           ▼                   │                                │
│  ┌──────────────────┐        │                                │
│  │Answer Generator  │◀───────┘                                │
│  │(Output Guard)    │                                         │
│  └────────┬─────────┘                                         │
│           ▼                                                   │
│  ┌──────────────────┐                                         │
│  │Context Manager   │  ← Sliding window (4K token threshold)  │
│  │(自动摘要/总结)     │                                         │
│  └────────┬─────────┘                                         │
│           ▼                                                   │
│  Memory: MemorySaver (checkpoint) + MySQL (long-term summary) │
└───────────────────────────────────────────────────────────────┘
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.12 |
| API 框架 | FastAPI + SSE 流式输出 |
| LLM | Claude Sonnet 4.6 (thinking mode) |
| Agent 编排 | LangGraph（StateGraph, MemorySaver, 条件路由） |
| 向量检索 | Qdrant + BGE-M3 embedding |
| 关键词检索 | BM25Okapi |
| 检索融合 | RRF (Reciprocal Rank Fusion) |
| 查询重写 | LLM-based query decomposition |
| 重排序 | BGE Reranker v2-m3 (cross-encoder) |
| 文本分割 | MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter (cl100k_base) |
| 关系数据库 | MySQL |
| 缓存/队列 | Redis |
| 前端 | Gradio |
| 评估 | RAGAS (60条测试集) |
| 安全守卫 | AC自动机敏感词检测 + 输入/输出过滤 |
| 记忆管理 | 滑动窗口摘要 (4K token) + 跨会话长期记忆 |
| 部署 | Docker + docker-compose |

## 核心流程

```
用户输入 → InputGuard(安全检查) → IntentRouter(意图分类)
    ├─ retrieve → QueryRewrite(2-3子查询) → HybridSearch(BM25+Vector+RRF)
    │              → CrossEncoderRerank → AnswerGen → OutputGuard
    ├─ clarify → 追问处理
    └─ general → 闲聊应答

所有路径 → ContextManager(滑动窗口摘要) → 长期记忆(MySQL)
```

## 快速开始

### 前置要求

- Python 3.11+
- Docker & Docker Compose
- Claude API key（或兼容的中转 API）

### 1. 启动基础设施

```bash
docker-compose up -d
```

启动 MySQL（3307）、Redis（6379）、Qdrant（6333）。

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 Claude API key：

```env
CLAUDE_API_KEY=sk-your-key-here
CLAUDE_API_BASE=https://api.your-proxy.com
CLAUDE_MODEL=claude-sonnet-4-6

MYSQL_URL=mysql+aiomysql://root:root123@localhost:3307/knowledge_agent
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 初始化数据库与导入文档

```bash
python scripts/init_db.py
python scripts/ingest_docs.py   # 导入 FastAPI 等文档到 Qdrant
```

### 5. 启动服务

```bash
# 终端 1: FastAPI
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 终端 2: Gradio 前端
python app/ui/gradio_app.py

# 终端 3: Celery Worker（文档上传需要）
celery -A app.tasks.ingest worker --loglevel=info --pool=solo
```

### 6. 访问

- Gradio UI: http://localhost:7860
- API: http://localhost:8000/api/chat
- Health Check: http://localhost:8000/health

## 项目结构

```
knowledge-agent/
├── app/
│   ├── api/
│   │   ├── chat.py              # 对话接口（SSE 流式 + node-status）
│   │   └── document.py          # 文档上传/管理
│   ├── agent/
│   │   ├── graph.py             # LangGraph 7-Node StateGraph 定义
│   │   ├── guardrails.py        # InputGuard + OutputGuard（敏感词/实体校验）
│   │   ├── context.py           # 滑动窗口摘要管理（cl100k_base 计数）
│   │   ├── tools.py             # 工具定义（search_docs）
│   │   └── memory.py            # 长期记忆管理
│   ├── rag/
│   │   ├── loader.py            # 文档加载
│   │   ├── splitter.py          # 两级分块（MarkdownHeader + Recursive）
│   │   ├── embedder.py          # BGE-M3 封装
│   │   ├── retriever.py         # 混合检索（BM25+向量+RRF+多查询融合）
│   │   ├── query_rewrite.py     # LLM 查询重写（2-3子查询）
│   │   └── reranker.py          # Cross-Encoder 精排（BGE Reranker v2-m3）
│   ├── tasks/
│   │   └── ingest.py            # Celery 异步入库任务
│   ├── ui/
│   │   └── gradio_app.py        # Gradio 前端
│   ├── models/                  # SQLAlchemy 数据模型
│   └── core/
│       ├── config.py            # 配置管理
│       └── db.py                # 数据库连接
├── eval/
│   ├── dataset.json             # 60 条问答对（FastAPI/LangChain/LangGraph）
│   ├── ragas_eval.py            # RAGAS 评估脚本
│   └── ragas_results.json       # 评估结果
├── scripts/                     # 工具脚本
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## API 接口

### POST /api/chat

SSE 流式对话接口。

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How to define path parameters in FastAPI?", "thread_id": "session-1"}'
```

**SSE 事件格式：**

```
data: {"type": "status", "node": "input_guard"}
data: {"type": "status", "node": "intent_router"}
data: {"type": "status", "node": "doc_retriever"}
data: {"type": "status", "node": "answer_generator"}
data: {"type": "text", "content": "FastAPI 是..."}
data: {"type": "contexts", "content": ["[1] doc text...\nSource: file.md"]}
data: [DONE]
```

### POST /api/documents/upload

上传文档到知识库。

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@your-doc.md"
```

## RAGAS 评估结果

60 条测试集（FastAPI 26 + LangChain 20 + LangGraph 14），涵盖 BM25 关键词检索 + 向量语义检索 + RRF 融合 + Cross-Encoder 重排序。

| 指标 | 分数 | 说明 |
|------|------|------|
| Faithfulness | 0.71 | 生成内容忠实于检索文档 |
| Answer Relevancy | 0.72 | 回答与问题相关性 |
| Context Precision | 0.36 | 检索结果精确度 |

> Context Precision 偏低说明检索噪音较多，可通过调整 chunk_size / top_k / RRF 参数优化。
>
> 基于 cl100k_base tokenizer 的分块策略（chunk_size=512, overlap=50）已针对 Markdown 层级结构优化。

## 面试亮点

- **混合检索 Pipeline**: BM25（关键词精确匹配）+ 向量检索（语义理解）+ RRF 融合 → Query Rewrite 查询分解 → Cross-Encoder 重排序，多层过滤提升检索精度
- **LangGraph 状态机**: 7 节点 DAG，包含输入安全守卫、意图路由（retrieve/clarify/general）、检索、生成、输出过滤、滑动窗口摘要管理
- **查询重写**: LLM 自动将模糊/过短的查询拆解为 2-3 个具体子查询，多路检索后跨查询 RRF 融合
- **安全防护**: 输入层（空查询/过长/敏感词 AC 自动机匹配）+ 输出层（技术实体提取 + 来源验证）
- **滑动窗口记忆**: 基于 cl100k_base 的 token 精确计数，触发 4K 阈值后自动压缩历史会话为摘要，兼顾长上下文和 token 成本
- **SSE 流式 + 节点状态**: 前端实时感知 Agent 推理阶段（守卫→路由→检索→生成→上下文管理）
- **异步架构**: 全异步 SSE 流式输出 + Celery 后台文档入库，不阻塞请求
- **评估驱动**: RAGAS 评估框架（60条测试集）量化检索和生成质量

## 常见问题

**Q: Agent 不调用 search_docs 怎么办？**
A: 确保新会话时 SystemMessage 正确传入。已实现 checkpoint 分支处理：新会话传完整 system prompt，已有会话只传新消息。

**Q: Windows 下 SSE API segfault？**
A: 这是 torch + aiomysql C 扩展冲突。已在 chat handler 中预加载 embedding 模型解决。

**Q: 如何更换 Embedding 模型？**
A: 修改 `.env` 中的 `EMBEDDING_MODEL`，然后重建 Qdrant collection。
