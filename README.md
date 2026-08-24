# Meeting-Agent

Meeting-Agent 是一个会议记录智能体。用户上传会议录音并填写会议基本信息后，系统会自动完成语音识别、说话人区分、会议内容提取，并生成可信会议纪要。

系统可以输出：

- 会议原始转写；
- 不同发言人的发言内容；
- 关键观点和会议决策；
- 行动项及负责人；
- 会议风险；
- 原话、说话人、时间戳和证据 ID；
- 可下载的 PDF 会议纪要；
- 会议内容和原话证据检索；
- 将 PDF 发送到指定邮箱。

## 使用方式

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

准备本地模型，并在项目根目录配置 `.env` 文件：

```env
PYANNOTE_MODEL=D:/futurework/Meeting-Agent/Meeting-Agent/models/pyannote-speaker-diarization-community-1-v2
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=glm4
LLM_API_KEY=ollama
QDRANT_URL=http://127.0.0.1:6333
REDIS_URL=redis://127.0.0.1:6379/0
```

### 2. 启动基础服务

```powershell
docker compose up -d
```

同时确保 Ollama 服务已经启动，并且本地模型可以正常使用。

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
.\.venv\Scripts\python.exe -m flet run --android apps\mobile\main.py
```

如果需要重新生成 APK：

```powershell
.\.venv\Scripts\python.exe -m flet build apk apps\mobile
```

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
