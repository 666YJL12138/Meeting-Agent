import requests
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


def api_get(path: str):
    url = f"{API_BASE}{path}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response


def init_session_state():
    defaults = {
        "meeting": None,
        "meeting_id": "",
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


def render_meeting_status(meeting: dict | None):
    if not meeting:
        st.info("请先创建会议。")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("会议状态", meeting.get("status", "-"))

    with col2:
        st.metric("处理进度", f"{meeting.get('progress', 0)}%")

    with col3:
        audio_info = meeting.get("audio_info") or {}
        duration = audio_info.get("duration_seconds")
        st.metric("音频时长", f"{duration:.2f} 秒" if duration else "-")

    with col4:
        speakers = meeting.get("speakers") or []
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
    st.markdown("### 创建会议")

    title = st.text_input("会议标题", value="Meeting Agent Test")
    host = st.text_input("主持人", value="Zhang San")
    language = st.selectbox("会议语言", ["zh-CN", "en-US"], index=0)
    participants_text = st.text_area("参会人，每行一个", value="Zhang San\nLi Si")

    if st.button("创建会议", use_container_width=True):
        try:
            participants = [
                item.strip()
                for item in participants_text.splitlines()
                if item.strip()
            ]
            create_meeting(title, host, language, participants)
            st.success("会议创建成功。")
        except Exception as exc:
            st.error(f"创建会议失败：{exc}")

    st.divider()

    st.markdown("### 上传音频")

    meeting_id = st.text_input(
        "会议ID",
        value=st.session_state.meeting_id,
        placeholder="创建会议后会自动填入",
    )

    uploaded_file = st.file_uploader(
        "选择会议音频",
        type=["wav", "mp3", "m4a", "aac", "flac"],
    )

    if st.button("上传音频", use_container_width=True):
        if not meeting_id:
            st.warning("请先创建会议或填写会议ID。")
        elif uploaded_file is None:
            st.warning("请选择音频文件。")
        else:
            try:
                upload_audio(meeting_id, uploaded_file)
                st.success("音频上传成功。")
            except Exception as exc:
                st.error(f"上传音频失败：{exc}")

    if st.button("运行会议分析", use_container_width=True):
        if not meeting_id:
            st.warning("请先创建会议或填写会议ID。")
        else:
            try:
                with st.spinner("正在进行 ASR、说话人归因、贡献抽取，请稍候..."):
                    run_analysis(meeting_id)
                st.success("会议分析完成。")
            except Exception as exc:
                st.error(f"运行分析失败：{exc}")

    st.divider()

    if st.button("生成并下载 PDF", use_container_width=True):
        if not meeting_id:
            st.warning("请先创建会议或填写会议ID。")
        else:
            try:
                pdf_bytes = download_pdf(meeting_id)
                st.download_button(
                    label="下载可信会议纪要 PDF",
                    data=pdf_bytes,
                    file_name=f"{meeting_id}_trusted_minutes.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"生成 PDF 失败：{exc}")

    st.divider()
    st.markdown("### 证据检索")
    search_query = st.text_input(
        "检索关键词",
        placeholder="例如：预算、风险、负责人、排期",
    )

    if st.button("搜索原话证据", use_container_width=True):
        if not meeting_id:
            st.warning("请先创建并分析会议")
        elif not search_query.strip():
            st.warning("请输入检索关键词")
        else:
            try:
                st.session_state.search_results = search_evidence(
                    meeting_id,
                    search_query,
                )
            except Exception as exc:
                st.error(f"证据检索失败：{exc}")

    current_result = st.session_state.analysis_result or {}
    review_claims = current_result.get("claims", [])

    if review_claims:
        st.divider()
        st.markdown("### 人工审校")
        claim_ids = [item["claim_id"] for item in review_claims]
        selected_claim_id = st.selectbox("选择结论", claim_ids)
        review_status = st.selectbox(
            "审校状态",
            ["reviewed", "confirmed", "rejected", "needs_review"],
        )
        review_note = st.text_area("审校备注")

        if st.button("提交审校结果", use_container_width=True):
            try:
                updated = review_claim(
                    meeting_id,
                    selected_claim_id,
                    review_status,
                    review_note,
                )
                for item in review_claims:
                    if item.get("claim_id") == selected_claim_id:
                        item.update(updated)
                        break
                st.success("审校结果已保存")
            except Exception as exc:
                st.error(f"提交审校失败：{exc}")

with right:
    st.markdown("### 会议状态")
    render_meeting_status(st.session_state.meeting)
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
