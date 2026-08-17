# 可信会议纪要 Agent

面向嘈杂线下会议场景的中文会议智能体。系统将现实会议录音转换为可追溯的会议纪要，并为每一条关键结论提供说话人、原话、时间戳、证据 ID 和人工审校状态。

## 项目目标

传统会议摘要容易出现“总结有了，但无法证明从哪里来”的问题。本项目以证据链为核心，完成：

1. 会议录音预处理与中文语音转写；
2. 说话人分离与发言归因；
3. 关键观点、决策、待办、风险、建议和承诺抽取；
4. 原话、时间戳、说话人、证据 ID 的可追溯绑定；
5. 基于真实 BGE 向量模型的会议证据语义检索；
6. 人工审校与可信 PDF 会议纪要生成；
7. Web 页面与手机端操作体验。

## 核心特性

- 中文会议转写：输出带起止时间的转写片段。
- 说话人归因：使用 Pyannote 生成说话人分段。
- 贡献抽取：按观点、决策、待办、风险、建议、承诺等类型提炼关键内容。
- 可信证据链：每条 claim 都关联原话、证据 ID、时间戳和说话人。
- 质量门禁：检测缺少证据或原话不匹配的结论，标记为 `needs_review`。
- 真实语义检索：使用本地 `BAAI/bge-small-zh-v1.5`、Redis 和 Qdrant 检索会议证据。
- 人工审校：支持确认、驳回和补充审校备注。
- PDF 纪要：生成包含结论、原话、证据和审校状态的规范化中文 PDF。
- 多端支持：提供 FastAPI、Streamlit Web 页面和 Flet 手机端。

## 技术架构

```text
会议录音
  -> 音频标准化
  -> ASR 转写
  -> 说话人分离
  -> 证据片段构建
  -> LangGraph Agent 编排
  -> 贡献抽取与可信性校验
  -> Redis 缓存 / Qdrant 向量索引
  -> Web / 手机端 / 可信 PDF
```

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 后端 API | FastAPI、Pydantic、SQLAlchemy |
| Agent 编排 | LangGraph、LangChain |
| 缓存 | Redis |
| 向量检索 | Qdrant、Sentence Transformers、BAAI/bge-small-zh-v1.5 |
| 语音能力 | Pyannote、Torch、Torchaudio、项目配置的 ASR 模型 |
| PDF | ReportLab |
| Web 前端 | Streamlit |
| 手机端 | Flet |
| 测试 | Pytest |

## 运行环境

- Windows 10/11
- Python 3.10+
- Docker Desktop（用于 Redis、Qdrant、数据库等基础服务）
- 已配置并可访问的 Pyannote 模型
- 本地真实 BGE 模型

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

启动基础服务：

```powershell
docker compose up -d
```

## 准备真实 BGE 模型

模型目录为：

```text
models\bge-small-zh-v1.5
```

模型权重不提交到 Git。首次部署时可通过 ModelScope 下载：

```powershell
.\.venv\Scripts\python.exe -m pip install modelscope
.\.venv\Scripts\python.exe -c "from modelscope import snapshot_download; snapshot_download(model_id='BAAI/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5')"
```

验证 BGE 向量维度：

```powershell
.\.venv\Scripts\python.exe -c "from services.embeddings import embed_texts; print(len(embed_texts(['meeting risk'])[0]))"
```

输出 `512` 表示真实 BGE 模型已成功加载。

## 启动项目

启动后端：

```powershell
uvicorn apps.api.main:app --reload
```

启动 Web 页面：

```powershell
streamlit run apps/web/app.py
```

启动手机端开发版：

```powershell
flet run apps/mobile/main.py
```

手机访问电脑后端时，必须填写电脑局域网地址，例如：

```text
http://192.168.x.x:8000
```

不要在手机端填写 `127.0.0.1`。

## 核心 API

| 方法 | 接口 | 说明 |
| --- | --- | --- |
| POST | `/meetings` | 创建会议 |
| POST | `/meetings/{meeting_id}/audio` | 上传会议音频 |
| POST | `/meetings/{meeting_id}/run-asr` | 执行完整会议分析 |
| GET | `/meetings/{meeting_id}/claims` | 查询关键贡献 |
| GET | `/meetings/{meeting_id}/search?q=关键词` | 语义检索会议证据 |
| POST | `/meetings/{meeting_id}/claims/{claim_id}/review` | 提交人工审校 |
| GET | `/meetings/{meeting_id}/quality-report` | 获取可信性质量报告 |
| GET | `/meetings/{meeting_id}/report/pdf` | 下载可信会议纪要 PDF |
| GET | `/health` | 服务健康检查 |

## 可信性机制

系统不会把模型生成的内容直接视为可信事实。每条关键贡献必须具备：

1. `speaker_id`：对应说话人；
2. `quote`：可核对的会议原话；
3. `start_ms` 和 `end_ms`：原话在音频中的时间范围；
4. `evidence_ids`：证据片段编号；
5. `review_status`：自动生成或人工审校状态。

通过质量报告接口检查：

```text
GET /meetings/{meeting_id}/quality-report
```

只有当结果满足以下条件时，会议纪要才可作为可信交付物：

```json
{
  "status": "passed",
  "evidence_coverage": 1.0,
  "invalid_claim_count": 0
}
```

## 测试

运行质量门禁与 API 自动化测试：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_quality_gate.py `
  tests/test_api_quality.py `
  -q -p no:cacheprovider
```

预期结果：

```text
5 passed
```

## 已知限制

- 单人录音通常只会得到一个 `speaker_00`，这是符合预期的。
- 多人归因效果依赖会议录音质量、重叠发言比例和说话人差异。
- 当前系统通过“证据约束”降低幻觉风险，但人工审校仍是正式会议纪要交付的重要环节。
