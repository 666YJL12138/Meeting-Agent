# Meeting-Agent

Meeting-Agent 是一个会议记录智能体。用户上传会议录音并填写会议基本信息后，系统会自动完成语音识别、说话人区分、会议内容提取，并生成带证据链的会议纪要。

系统可以输出：

- 会议原始转写；
- 不同发言人的发言内容；
- 关键观点和会议决策；
- 行动项及负责人；
- 会议风险；
- 原话、说话人、时间戳和证据 ID；
- 可下载的 PDF 会议纪要；
- 会议内容和原话证据检索；
- 多人重叠语音区间和 exclusive 说话人分段；
- 说话人归因评估指标，包括 DER、JER、漏检率、误报率、混淆率、说话人准确率，以及重叠语音 Precision、Recall、F1；
- Agent 执行耗时、缓存命中状态、审核状态和复核原因；
- 将 PDF 发送到指定邮箱。

## 使用方式

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

音频格式转换依赖 FFmpeg。请先安装 FFmpeg，并确认它已加入系统 `PATH`：

```powershell
ffmpeg -version
```

准备 ASR 和 pyannote 模型，并在项目根目录配置 `.env` 文件。可以先复制示例配置：

```powershell
Copy-Item .env.example .env
```

如果使用本地 pyannote 模型，`PYANNOTE_MODEL` 应指向包含 `config.yaml` 的模型目录；如果使用 Hugging Face 模型，则需要填写 `HF_TOKEN`：

```env
PYANNOTE_MODEL=D:/futurework/Meeting-Agent/Meeting-Agent/models/pyannote-speaker-diarization-community-1-v2
HF_TOKEN=
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=glm4
LLM_API_KEY=ollama
QDRANT_URL=http://127.0.0.1:6333
REDIS_URL=redis://127.0.0.1:6379/0
AGENT_EXTRACT_MODE=batched
AGENT_CACHE_ENABLED=1
LLM_NUM_PREDICT=768
```

### 2. 启动基础服务

```powershell
docker compose up -d
```

该命令会启动 PostgreSQL、Redis、Qdrant 和 MinIO。与此同时，确保 Ollama 已启动并且模型可用：

```powershell
ollama list
```

默认使用的模型是 `glm4`，如未安装，请先在 Ollama 中准备对应模型。

### 3. 启动后端

```powershell
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```

后端接口地址：

```text
http://127.0.0.1:8000
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

### 4. 使用 Web 端

新开终端：

```powershell
streamlit run apps/web/app.py
```

打开 Web 页面后：

1. 填写会议标题、主持人和参会人；
2. 上传会议音频；
3. 点击“一键分析”；
4. 等待分析任务完成；
5. 查看发言人贡献、关键结论、行动项、风险和原始转写；
6. 下载生成的 PDF 报告。

### 5. 使用手机端

将 Android 手机连接到电脑后，在项目目录执行：

```powershell
.\.venv\Scripts\flet.exe run --android apps\mobile\main.py
```

如果需要重新生成 APK：

```powershell
.\.venv\Scripts\flet.exe build apk apps\mobile --yes
```

运行或构建 Android 应用前，需要准备 Android SDK、ADB 和可用的 Android 设备或模拟器。

手机和电脑连接同一个局域网后，手机端后端地址需要填写电脑的局域网 IP，例如：

```text
http://192.168.1.100:8000
```

不要填写：

```text
http://127.0.0.1:8000
```

手机端使用流程：

1. 选择手机中的会议音频；
2. 填写会议基本信息；
3. 点击“一键分析会议”；
4. 查看任务进度；
5. 分析完成后查看会议结果；
6. 打开或下载 PDF 报告。

如果任务状态暂时加载失败，可以点击：

- 刷新任务状态；
- 继续等待分析。

这两个操作不会重新上传音频。

### 6. 测试和评估

运行全部自动化测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

运行端到端测试并查看 ASR、Agent、总耗时和质量报告：

```powershell
$env:AGENT_EXTRACT_MODE="batched"
$env:AGENT_CACHE_ENABLED="0"
.\.venv\Scripts\python.exe benchmark_e2e.py
```

如需对比并行提取模式：

```powershell
$env:AGENT_EXTRACT_MODE="parallel"
$env:AGENT_CACHE_ENABLED="0"
.\.venv\Scripts\python.exe benchmark_e2e.py
```

说话人归因评估需要参考标注，可通过以下接口提交参考分段：

```text
POST /meetings/{meeting_id}/diarization/evaluate
GET  /meetings/{meeting_id}/diarization
```

## 输出文件

分析结果默认保存在：

```text
outputs/asr/
outputs/agent/
outputs/reports/
```

PDF 报告位置：

```text
outputs/reports/<meeting_id>_trusted_minutes.pdf
```
