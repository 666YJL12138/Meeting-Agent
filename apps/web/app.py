from html import escape
import requests
import time
import streamlit as st
from urllib.parse import quote

from services.meeting_defaults import (
    DEFAULT_MEETING_HOST,
    DEFAULT_MEETING_LANGUAGE,
    DEFAULT_MEETING_PARTICIPANTS,
    DEFAULT_MEETING_TITLE,
)


API_BASE = "http://127.0.0.1:8000"


def html_text(value) -> str:
    return escape(str(value if value is not None else ""))


def claim_type_text(claim: dict) -> str:
    return str(claim.get("claim_type") or "关键结论")


def is_action_claim(claim: dict) -> bool:
    value = claim_type_text(claim).lower()
    return any(item in value for item in ("action", "待办", "行动", "任务"))


def is_risk_claim(claim: dict) -> bool:
    value = claim_type_text(claim).lower()
    return "risk" in value or "风险" in value


def speaker_display_name(item: dict, result: dict | None = None) -> str:
    result = result or {}
    speaker_id = item.get("speaker_id") or "未识别说话人"
    mapping = result.get("speaker_mapping") or result.get("speaker_name_map") or {}
    return (
        item.get("speaker_name")
        or item.get("display_name")
        or mapping.get(speaker_id)
        or speaker_id
    )


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
    .action-box {
        border-left-color: #12B76A;
    }
    .risk-box {
        border-left-color: #F04438;
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
    """Display backend event order; progress is not a sort key."""
    return list(history or [])


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
        speaker_name = speaker_display_name(summary, result)
        st.subheader(f"发言人：{speaker_name}")

        key_points = summary.get("key_points") or []
        for point in key_points:
            evidence_ids = ", ".join(point.get("evidence_ids", []))
            st.markdown(
                f"""
                <div class="claim-box">
                    <b>类型：</b>{html_text(point.get("claim_type", "-"))}
                    &nbsp;&nbsp;
                    <b>置信度：</b>{html_text(point.get("confidence", "-"))}
                    <br/>
                    <b>要点：</b>{html_text(point.get("statement", ""))}
                    <br/>
                    <b>证据：</b>{html_text(evidence_ids)}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_claims(result: dict):
    claims = [
        claim for claim in (result.get("claims") or [])
        if not is_action_claim(claim) and not is_risk_claim(claim)
    ]

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
                <b>{html_text(claim_type_text(claim))}</b>
                &nbsp;|&nbsp;
                发言人：{html_text(speaker_display_name(claim, result))}
                &nbsp;|&nbsp;
                时间：{html_text(start_ms)}ms - {html_text(end_ms)}ms
                <br/>
                <b>总结：</b>{html_text(claim.get("statement", ""))}
                <div class="quote-box">{'<b>原话：</b>' + html_text(claim.get("quote", ""))}
                </div>
                <br/>
                <b>证据ID：</b>{html_text(evidence_ids)}
                &nbsp;&nbsp;
                <b>置信度：</b>{html_text(claim.get("confidence", "-"))}
                &nbsp;&nbsp;
                <b>审核状态：</b>{html_text(claim.get("review_status", "-"))}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_action_items(result: dict):
    claims = [
        claim for claim in result.get("action_items", [])
    ]
    if not claims:
        claims = [
            claim for claim in result.get("claims", [])
            if is_action_claim(claim)
        ]

    st.markdown("### 行动项")
    if not claims:
        st.info("暂无行动项。")
        return

    for index, claim in enumerate(claims, start=1):
        st.markdown(
            f"""
            <div class="claim-box action-box">
                <b>行动项 {index}</b>
                &nbsp;|&nbsp; 负责人：{html_text(speaker_display_name(claim, result))}
                &nbsp;|&nbsp; 支撑度：{html_text(claim.get("confidence", "-"))}
                <br/><b>内容：</b>{html_text(
                    claim.get("statement") or claim.get("summary", "")
                )}
                <br/><b>证据：</b>{html_text(
                    ", ".join(claim.get("evidence_ids", []))
                )}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_risks(result: dict):
    claims = list(result.get("risks", []))
    if not claims:
        claims = [
            claim for claim in result.get("claims", [])
            if is_risk_claim(claim)
        ]

    st.markdown("### 风险项")
    if not claims:
        st.info("暂无风险项。")
        return

    for index, claim in enumerate(claims, start=1):
        st.markdown(
            f"""
            <div class="claim-box risk-box">
                <b>风险 {index}</b>
                &nbsp;|&nbsp; 来源：{html_text(speaker_display_name(claim, result))}
                &nbsp;|&nbsp; 支撑度：{html_text(claim.get("confidence", "-"))}
                <br/><b>描述：</b>{html_text(
                    claim.get("statement") or claim.get("summary", "")
                )}
                <br/><b>证据：</b>{html_text(
                    ", ".join(claim.get("evidence_ids", []))
                )}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_evidence_chain(result: dict):
    evidence = result.get("evidence_links") or []

    st.markdown("### 证据链")
    if not evidence:
        st.info("暂无证据链。")
        return

    st.dataframe(
        [
            {
                "证据ID": item.get("evidence_id", "-"),
                "发言人": speaker_display_name(item, result),
                "时间": (
                    f"{item.get('start_ms', 0)}ms - "
                    f"{item.get('end_ms', 0)}ms"
                ),
                "原话": item.get("quote") or item.get("text", ""),
                "ASR置信度": item.get("asr_confidence", "-"),
                "声纹置信度": item.get("speaker_confidence", "-"),
                "来源": item.get("speaker_source", "unknown"),
            }
            for item in evidence
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_result_metrics(result: dict):
    claims = result.get("claims") or []
    action_items = result.get("action_items") or [
        claim for claim in claims if is_action_claim(claim)
    ]
    risks = result.get("risks") or [
        claim for claim in claims if is_risk_claim(claim)
    ]
    speakers = result.get("speaker_ids") or [
        item.get("speaker_id")
        for item in result.get("evidence_links", [])
        if item.get("speaker_id")
    ]

    columns = st.columns(4)
    metrics = [
        ("发言人数", len(set(speakers))),
        ("关键结论", len(claims) - len(action_items) - len(risks)),
        ("行动项", len(action_items)),
        ("风险项", len(risks)),
    ]
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value)


def render_transcript(result: dict):
    spans = result.get("transcript_spans") or []

    st.markdown('<div class="section-title">原始转写片段</div>', unsafe_allow_html=True)

    if not spans:
        st.warning("暂无转写文本。")
        return

    for span in spans:
        st.write(
            f"[{span.get('start_ms', 0)}-{span.get('end_ms', 0)}] "
            f"{speaker_display_name(span, result)}: {span.get('text', '')}"
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
                <b>相似度：</b>{html_text(hit.get("score", "-"))}<br/>
                <b>类型：</b>{html_text(payload.get("kind", "-"))}
                &nbsp;|&nbsp;
                <b>发言人：</b>{html_text(payload.get("speaker_id", "-"))}<br/>
                <b>时间：</b>{html_text(payload.get("start_ms", "-"))}
                - {html_text(payload.get("end_ms", "-"))} ms
                <div class="quote-box">{html_text(payload.get("text", ""))}</div>
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
        value=DEFAULT_MEETING_TITLE,
    )

    host = st.text_input(
        "主持人",
        value=DEFAULT_MEETING_HOST,
    )

    language = st.selectbox(
        "会议语言",
        ["zh-CN", "en-US"],
        index=0 if DEFAULT_MEETING_LANGUAGE == "zh-CN" else 1,
    )

    participants_text = st.text_area(
        "参会人，每行一个",
        value=DEFAULT_MEETING_PARTICIPANTS,
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

    result = st.session_state.analysis_result

    if result:
        render_result_metrics(result)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "会议摘要",
            "行动项",
            "风险项",
            "证据链",
            "原始转写",
        ])

        with tab1:
            render_speaker_summaries(result)
            render_claims(result)

        with tab2:
            render_action_items(result)

        with tab3:
            render_risks(result)

        with tab4:
            render_evidence_chain(result)
            render_search_results()

        with tab5:
            render_transcript(result)
    else:
        st.info("完成会议分析后，这里会展示发言人贡献、关键结论和原始转写。")
