import requests
import time
import streamlit as st
from urllib.parse import quote


API_BASE = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="可信会议纪要 Agent",
    page_icon=None,
    layout="wide",
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .sub-title {
        color: #667085;
        font-size: 14px;
        margin-bottom: 24px;
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        margin-top: 16px;
        margin-bottom: 8px;
    }
    .metric-box {
        border: 1px solid #EAECF0;
        padding: 14px;
        border-radius: 8px;
        background: #FCFCFD;
    }
    .claim-box {
        border: 1px solid #D0D5DD;
        border-left: 4px solid #2E90FA;
        padding: 12px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
        background: #FFFFFF;
    }
    .quote-box {
        background: #F9FAFB;
        border: 1px solid #EAECF0;
        padding: 10px;
        border-radius: 6px;
        color: #344054;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_post(path: str, json_body: dict | None = None, files: dict | None = None):
    url = f"{API_BASE}{path}"
    response = requests.post(url, json=json_body, files=files, timeout=300)
    response.raise_for_status()
    return response.json()


def start_async_analysis(
    uploaded_file,
    title: str,
    host: str,
    language: str,
    participants: list[str],
):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
            or "application/octet-stream",
        )
    }

    data = {
        "title": title,
        "host": host,
        "language": language,
        "participants": "\n".join(
            participants
        ),
        "send_email": "false",
    }

    response = requests.post(
        f"{API_BASE}/meetings/analyze",
        data=data,
        files=files,
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def get_job_status(job_id: str) -> dict:
    response = requests.get(
        f"{API_BASE}/jobs/{job_id}",
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_job_result(job_id: str) -> dict:
    response = requests.get(
        f"{API_BASE}/jobs/{job_id}/result",
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_job_pdf(job_id: str) -> bytes:
    response = requests.get(
        f"{API_BASE}/jobs/{job_id}/report/pdf",
        timeout=120,
    )
    response.raise_for_status()
    return response.content


def api_get(path: str):
    url = f"{API_BASE}{path}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response


def init_session_state():
    defaults = {
        "meeting": None,
        "meeting_id": "",
        "job_id": "",
        "job_status": None,
        "job_history": [],
        "analysis_result": None,
        "search_results": None,
        "error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_meeting(title: str, host: str, language: str, participants: list[str]):
    payload = {
        "title": title,
        "host": host,
        "language": language,
        "participants": participants,
    }
    meeting = api_post("/meetings", json_body=payload)
    st.session_state.meeting = meeting
    st.session_state.meeting_id = meeting["meeting_id"]
    st.session_state.analysis_result = None


def upload_audio(meeting_id: str, uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "audio/wav",
        )
    }
    meeting = api_post(f"/meetings/{meeting_id}/audio", files=files)
    st.session_state.meeting = meeting


def run_analysis(meeting_id: str):
    result = api_post(f"/meetings/{meeting_id}/run-asr")
    st.session_state.analysis_result = result
    st.session_state.meeting = result


def download_pdf(meeting_id: str) -> bytes:
    response = api_get(f"/meetings/{meeting_id}/report/pdf")
    return response.content


def search_evidence(meeting_id: str, query: str) -> dict:
    encoded_query = quote(query.strip())
    response = api_get(
        f"/meetings/{meeting_id}/search?q={encoded_query}&limit=8"
    )
    return response.json()


def review_claim(
    meeting_id: str,
    claim_id: str,
    review_status: str,
    review_note: str,
) -> dict:
    response = requests.post(
        f"{API_BASE}/meetings/{meeting_id}/claims/{claim_id}/review",
        json={
            "review_status": review_status,
            "review_note": review_note,
            "reviewer": "web_user",
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def run_async_analysis_web(
    uploaded_file,
    title: str,
    host: str,
    language: str,
    participants: list[str],
):
    job = start_async_analysis(
        uploaded_file=uploaded_file,
        title=title,
        host=host,
        language=language,
        participants=participants,
    )

    job_id = job["job_id"]
    meeting_id = job["meeting_id"]

    st.session_state.job_id = job_id
    st.session_state.meeting_id = meeting_id
    st.session_state.job_status = job
    st.session_state.job_history = job.get(
        "history",
        [],
    )
    st.session_state.analysis_result = None
    st.session_state.meeting = {
        "meeting_id": meeting_id,
        "title": title,
        "host": host,
        "language": language,
        "status": job.get(
            "status",
            "queued",
        ),
        "progress": job.get(
            "progress",
            10,
        ),
        "audio_info": {},
        "speakers": [],
    }

    st.success(
        f"任务已创建：{job_id}"
    )

    progress_bar = st.progress(
        int(job.get("progress", 0))
    )
    status_placeholder = st.empty()
    history_placeholder = st.empty()

    while True:
        current = get_job_status(job_id)

        progress_value = int(
            current.get("progress", 0)
        )
        status_value = current.get(
            "status",
            "-",
        )
        stage_value = current.get(
            "stage",
            "-",
        )

        progress_bar.progress(
            progress_value
        )
        status_placeholder.info(
            f"状态：{status_value}，"
            f"进度：{progress_value}%\n\n"
            f"当前阶段：{stage_value}"
        )

        history = current.get(
            "history",
            [],
        )
        st.session_state.job_status = current
        st.session_state.job_history = history
        st.session_state.meeting.update(
            {
                "status": status_value,
                "progress": progress_value,
            }
        )

        if history:
            history_placeholder.dataframe(
                [
                    {
                        "时间": item.get(
                            "timestamp",
                            "",
                        ),
                        "状态": item.get(
                            "status",
                            "",
                        ),
                        "进度": (
                            f"{item.get('progress', 0)}%"
                        ),
                        "阶段": item.get(
                            "stage",
                            "",
                        ),
                    }
                    for item in history
                ],
                use_container_width=True,
                hide_index=True,
            )

        if status_value in {
            "completed",
            "failed",
            "cancelled",
        }:
            break

        time.sleep(2)

    if status_value == "completed":
        result = get_job_result(job_id)
        st.session_state.analysis_result = result
        st.session_state.meeting.update(
            {
                "status": "completed",
                "progress": 100,
                "audio_info": result.get(
                    "audio_info",
                    {},
                ),
                "speakers": result.get(
                    "speaker_ids",
                    [],
                ),
            }
        )
        st.success("会议分析完成。")
        return result

    if status_value == "failed":
        st.error(
            "会议分析失败："
            + str(current.get("error"))
        )
        return None

    st.warning("任务已取消。")
    return None


def ordered_job_history(history: list[dict]) -> list[dict]:
    """Keep old malformed timelines readable while preserving stable order."""
    return [
        item
        for _, item in sorted(
            enumerate(history),
            key=lambda pair: (
                int(pair[1].get("progress", 0)),
                pair[0],
            ),
        )
    ]


def render_job_history():
    job_status = (
        st.session_state.job_status
        or {}
    )
    history = st.session_state.job_history

    if not job_status and not history:
        st.info("提交任务后，这里会显示处理进度。")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "任务进度",
            f"{job_status.get('progress', 0)}%",
        )

    with col2:
        st.metric(
            "任务状态",
            job_status.get("status", "-"),
        )

    with col3:
        st.metric(
            "处理阶段",
            job_status.get("stage", "-"),
        )

    if history:
        display_history = ordered_job_history(
            history
        )
        st.markdown(
            '<div class="section-title">'
            "完整处理轨迹"
            "</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            [
                {
                    "时间": item.get(
                        "timestamp",
                        "",
                    ),
                    "状态": item.get(
                        "status",
                        "",
                    ),
                    "进度": (
                        f"{item.get('progress', 0)}%"
                    ),
                    "阶段": item.get(
                        "stage",
                        "",
                    ),
                }
                for item in display_history
            ],
            use_container_width=True,
            hide_index=True,
        )


def sync_completed_job_result() -> None:
    """Load the final Agent result whenever a completed Job is visible."""
    job_status = (
        st.session_state.job_status
        or {}
    )

    if (
        job_status.get("status") != "completed"
        or st.session_state.analysis_result
    ):
        return

    job_id = st.session_state.job_id
    if not job_id:
        return

    try:
        result = get_job_result(job_id)
        st.session_state.analysis_result = result

        meeting = st.session_state.meeting or {}
        meeting.update(
            {
                "status": "completed",
                "progress": 100,
                "audio_info": result.get(
                    "audio_info",
                    {},
                ),
                "speakers": result.get(
                    "speaker_ids",
                    [],
                ),
            }
        )
        st.session_state.meeting = meeting
    except Exception as exc:
        st.session_state.error = (
            f"任务已完成，但加载分析结果失败：{exc}"
        )


def render_meeting_status(
    meeting: dict | None,
    job_status: dict | None = None,
):
    job_status = job_status or {}
    if not meeting and not job_status:
        st.info("请先创建会议。")
        return

    meeting = meeting or {}
    current_status = {
        **meeting,
        "status": job_status.get(
            "status",
            meeting.get("status", "-"),
        ),
        "progress": job_status.get(
            "progress",
            meeting.get("progress", 0),
        ),
    }

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "会议状态",
            current_status.get("status", "-"),
        )

    with col2:
        st.metric(
            "处理进度",
            f"{current_status.get('progress', 0)}%",
        )

    with col3:
        audio_info = current_status.get(
            "audio_info"
        ) or {}
        duration = audio_info.get("duration_seconds")
        st.metric("音频时长", f"{duration:.2f} 秒" if duration else "-")

    with col4:
        speakers = current_status.get(
            "speakers"
        ) or []
        st.metric("发言人数", len(speakers))


def render_speaker_summaries(result: dict):
    summaries = result.get("speaker_summaries") or []

    st.markdown('<div class="section-title">发言人关键贡献</div>', unsafe_allow_html=True)

    if not summaries:
        st.warning("暂无发言人贡献摘要。")
        return

    for summary in summaries:
        speaker_name = summary.get("display_name") or summary.get("speaker_id")
        st.subheader(f"发言人：{speaker_name}")

        key_points = summary.get("key_points") or []
        for point in key_points:
            evidence_ids = ", ".join(point.get("evidence_ids", []))
            st.markdown(
                f"""
                <div class="claim-box">
                    <b>类型：</b>{point.get("claim_type", "-")}
                    &nbsp;&nbsp;
                    <b>置信度：</b>{point.get("confidence", "-")}
                    <br/>
                    <b>要点：</b>{point.get("statement", "")}
                    <br/>
                    <b>证据：</b>{evidence_ids}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_claims(result: dict):
    claims = result.get("claims") or []

    st.markdown('<div class="section-title">可追溯关键结论</div>', unsafe_allow_html=True)

    if not claims:
        st.warning("暂无关键结论。")
        return

    for claim in claims:
        evidence_ids = ", ".join(claim.get("evidence_ids", []))
        start_ms = claim.get("start_ms", 0)
        end_ms = claim.get("end_ms", 0)

        st.markdown(
            f"""
            <div class="claim-box">
                <b>{claim.get("claim_type", "-")}</b>
                &nbsp;|&nbsp;
                发言人：{claim.get("speaker_id", "-")}
                &nbsp;|&nbsp;
                时间：{start_ms}ms - {end_ms}ms
                <br/>
                <b>总结：</b>{claim.get("statement", "")}
                <div class="quote-box">
                    <b>原话：</b>{claim.get("quote", "")}
                </div>
                <br/>
                <b>证据ID：</b>{evidence_ids}
                &nbsp;&nbsp;
                <b>置信度：</b>{claim.get("confidence", "-")}
                &nbsp;&nbsp;
                <b>审核状态：</b>{claim.get("review_status", "-")}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_transcript(result: dict):
    spans = result.get("transcript_spans") or []

    st.markdown('<div class="section-title">原始转写片段</div>', unsafe_allow_html=True)

    if not spans:
        st.warning("暂无转写文本。")
        return

    for span in spans:
        st.write(
            f"[{span.get('start_ms', 0)}-{span.get('end_ms', 0)}] "
            f"{span.get('speaker_id', '-')}: {span.get('text', '')}"
        )


def render_search_results():
    data = st.session_state.search_results
    if not data:
        return

    st.markdown("### 证据检索结果")
    hits = data.get("hits", [])
    if not hits:
        st.info("没有找到匹配的原话证据")
        return

    for hit in hits:
        payload = hit.get("payload", {})
        st.markdown(
            f"""
            <div class="claim-box">
                <b>相似度：</b>{hit.get("score", "-")}<br/>
                <b>类型：</b>{payload.get("kind", "-")}
                &nbsp;|&nbsp;
                <b>发言人：</b>{payload.get("speaker_id", "-")}<br/>
                <b>时间：</b>{payload.get("start_ms", "-")} - {payload.get("end_ms", "-")} ms
                <div class="quote-box">{payload.get("text", "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


init_session_state()

st.markdown('<div class="main-title">可信会议纪要 Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">面向真实会议场景的发言人归因、关键贡献抽取与可追溯 PDF 生成工具</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([0.34, 0.66])

with left:
    st.markdown("### 一键生成可信会议纪要")

    title = st.text_input(
        "会议标题",
        value="真实会议测试",
    )

    host = st.text_input(
        "主持人",
        value="主持人",
    )

    language = st.selectbox(
        "会议语言",
        ["zh-CN", "en-US"],
        index=0,
    )

    participants_text = st.text_area(
        "参会人，每行一个",
        value="张三\n李四\n王五",
    )

    uploaded_file = st.file_uploader(
        "选择会议音频",
        type=[
            "wav",
            "mp3",
            "m4a",
            "aac",
            "flac",
        ],
    )

    start_button = st.button(
        "开始一键分析",
        use_container_width=True,
        type="primary",
    )

    if start_button:
        if uploaded_file is None:
            st.warning("请先选择会议音频。")
        else:
            participants = [
                item.strip()
                for item in participants_text.splitlines()
                if item.strip()
            ]

            try:
                run_async_analysis_web(
                    uploaded_file=uploaded_file,
                    title=title,
                    host=host,
                    language=language,
                    participants=participants,
                )
            except Exception as exc:
                st.error(
                    f"启动会议分析失败：{exc}"
                )

    st.divider()

    if st.session_state.job_id:
        st.markdown("### 当前任务")

        st.code(
            st.session_state.job_id,
            language="text",
        )

        if st.button(
            "刷新任务状态",
            use_container_width=True,
        ):
            try:
                current = get_job_status(
                    st.session_state.job_id
                )
                st.session_state.job_status = current
                st.session_state.job_history = (
                    current.get("history", [])
                )
                if current.get("status") == "completed":
                    st.session_state.analysis_result = (
                        get_job_result(
                            st.session_state.job_id
                        )
                    )
                st.rerun()
            except Exception as exc:
                st.error(
                    f"刷新任务失败：{exc}"
                )

        final_status = (
            st.session_state.job_status or {}
        ).get("status")

        if final_status == "completed":
            try:
                pdf_data = get_job_pdf(
                    st.session_state.job_id
                )

                st.download_button(
                    "下载可信会议纪要 PDF",
                    data=pdf_data,
                    file_name=(
                        f"{st.session_state.meeting_id}"
                        "_trusted_minutes.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(
                    f"PDF下载失败：{exc}"
                )

with right:
    sync_completed_job_result()
    st.markdown("### 会议状态")
    render_job_history()
    render_meeting_status(
        st.session_state.meeting,
        st.session_state.job_status,
    )
    render_search_results()

    result = st.session_state.analysis_result

    if result:
        tab1, tab2, tab3 = st.tabs(["发言人贡献", "可追溯结论", "原始转写"])

        with tab1:
            render_speaker_summaries(result)

        with tab2:
            render_claims(result)

        with tab3:
            render_transcript(result)
    else:
        st.info("完成会议分析后，这里会展示发言人贡献、关键结论和原始转写。")
