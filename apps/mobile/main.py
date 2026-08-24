from pathlib import Path
import io
from urllib.parse import quote
import asyncio
import flet as ft
import requests


DEFAULT_API_BASE = "http://10.193.23.250:8000"
DEFAULT_AUDIO_PATH = r"D:\futurework\Meeting-Agent\Meeting-Agent\day2_chinese_meeting_sample.wav"
DEFAULT_AUDIO_LABEL = "尚未选择音频文件"

CLAIM_TYPE_LABELS = {
    "viewpoint": "观点",
    "decision": "决策",
    "action_item": "待办",
    "risk": "风险",
    "suggestion": "建议",
    "commitment": "承诺",
    "fact": "事实",
}

STATUS_LABELS = {
    "created": "已创建",
    "audio_uploaded": "音频已上传",
    "completed": "分析完成",
    "failed": "失败",
}


def format_ms(value) -> str:
    if value is None:
        return "-"
    seconds = int(value // 1000)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def label_claim_type(value: str) -> str:
    return CLAIM_TYPE_LABELS.get(value or "", value or "-")


def label_status(value: str) -> str:
    return STATUS_LABELS.get(value or "", value or "-")


def is_action_claim(claim: dict) -> bool:
    value = str(claim.get("claim_type") or "").lower()
    return any(item in value for item in ("action", "待办", "行动", "任务"))


def is_risk_claim(claim: dict) -> bool:
    value = str(claim.get("claim_type") or "").lower()
    return "risk" in value or "风险" in value


def post_analyze(
    api_base: str,
    title: str,
    host: str,
    participants: str,
    audio_name: str,
    audio_bytes: bytes,
):
    data = {
        "title": title,
        "host": host,
        "language": "zh-CN",
        "participants": participants,
        "send_email": "false",
    }

    files = {
        "file": (
            audio_name,
            io.BytesIO(audio_bytes),
            "application/octet-stream",
        )
    }

    response = requests.post(
        f"{api_base}/meetings/analyze",
        data=data,
        files=files,
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


async def poll_job_async(
    api_base: str,
    job_id: str,
    on_update,
    on_error=None,
    max_consecutive_errors: int = 5,
):
    consecutive_errors = 0
    last_job = None

    while True:
        try:
            job = await asyncio.to_thread(
                get_json,
                api_base,
                f"/jobs/{job_id}",
            )
            consecutive_errors = 0
            last_job = job
        except requests.RequestException as exc:
            consecutive_errors += 1
            if on_error:
                await on_error(exc, consecutive_errors)

            if consecutive_errors >= max_consecutive_errors:
                return last_job

            await asyncio.sleep(3)
            continue

        await on_update(job)

        if job.get("status") in {
            "completed",
            "failed",
            "cancelled",
        }:
            return job

        await asyncio.sleep(2)


def post_json(api_base: str, path: str, payload: dict | None = None):
    response = requests.post(f"{api_base}{path}", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def post_file(api_base: str, path: str, file_path: str):
    with open(file_path, "rb") as file:
        files = {"file": (Path(file_path).name, file, "application/octet-stream")}
        response = requests.post(f"{api_base}{path}", files=files, timeout=300)
    response.raise_for_status()
    return response.json()


def post_file_bytes(api_base: str, path: str, file_name: str, file_bytes: bytes):
    files = {"file": (file_name, io.BytesIO(file_bytes), "application/octet-stream")}
    response = requests.post(f"{api_base}{path}", files=files, timeout=300)
    response.raise_for_status()
    return response.json()


def get_bytes(api_base: str, path: str) -> bytes:
    response = requests.get(f"{api_base}{path}", timeout=120)
    response.raise_for_status()
    return response.content


def get_json(api_base: str, path: str):
    response = requests.get(f"{api_base}{path}", timeout=120)
    response.raise_for_status()
    return response.json()


def main(page: ft.Page):
    page.title = "可信会议纪要"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = "#F6F8FB"
    page.padding = 0
    page.scroll = ft.ScrollMode.AUTO

    state = {
        "meeting_id": "",
        "job_id": "",
        "audio_path": "",
        "audio_name": "",
        "audio_bytes": None,
        "result": None,
        "job_status": None,
        "history": [],
        "polling": False,
    }

    api_base = ft.TextField(label="后端地址", value=DEFAULT_API_BASE)
    title = ft.TextField(label="会议标题", value="Meeting Agent Test")
    host = ft.TextField(label="主持人", value="Zhang San")
    participants = ft.TextField(
        label="参会人",
        value="Zhang San\nLi Si",
        multiline=True,
        min_lines=2,
        max_lines=4,
    )
    audio_path_input = ft.TextField(label="音频文件", value=DEFAULT_AUDIO_LABEL, read_only=True)
    meeting_id_text = ft.Text("未创建", size=12, color="#344054")
    status_text = ft.Text("等待创建会议", size=14, weight=ft.FontWeight.BOLD, color="#175CD3")
    audio_text = ft.Text("请填写本地音频路径", size=12, color="#667085")
    debug_log = ft.Text("准备就绪", size=12, color="#175CD3")
    progress = ft.ProgressRing(visible=False)
    progress_bar = ft.ProgressBar(
        value=0,
        visible=True,
        color="#175CD3",
        bgcolor="#D0D5DD",
    )
    metrics_row = ft.Row(
        spacing=8,
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )
    search_query = ft.TextField(
        label="检索关键词",
        hint_text="例如：风险、预算、负责人",
    )

    summary_list = ft.Column(spacing=10)
    claims_list = ft.Column(spacing=10)
    action_list = ft.Column(spacing=10)
    risk_list = ft.Column(spacing=10)
    evidence_list = ft.Column(spacing=10)
    transcript_list = ft.Column(spacing=6)
    history_list = ft.Column(spacing=6)
    search_results = ft.Column(spacing=8)
    file_picker = ft.FilePicker()
    page.services.append(file_picker)
    def card(content, bgcolor="#FFFFFF", padding=14):
        return ft.Container(
            content=content,
            padding=padding,
            bgcolor=bgcolor,
            border_radius=14,
        )

    def section_title(text: str):
        return ft.Text(text, size=17, weight=ft.FontWeight.BOLD, color="#101828")

    def show_message(message: str):
        debug_log.value = message
        print(message, flush=True)
        page.update()

    def set_busy(is_busy: bool, message: str | None = None):
        progress.visible = is_busy
        if message:
            status_text.value = message
        page.update()

    def refresh_status(meeting: dict):
        state["meeting_id"] = meeting.get("meeting_id", state["meeting_id"])
        meeting_id_text.value = state["meeting_id"] or "未创建"
        status_text.value = label_status(meeting.get("status"))
        if meeting.get("audio_uri"):
            audio_text.value = f"音频已上传：{state['audio_name'] or Path(meeting.get('audio_uri')).name}"
        else:
            audio_text.value = DEFAULT_AUDIO_LABEL

    async def choose_audio_async():
        try:
            files = await file_picker.pick_files(
                dialog_title="请选择会议音频",
                file_type=ft.FilePickerFileType.AUDIO,
                allow_multiple=False,
                with_data=True,
            )
            if not files:
                show_message("已取消选择音频")
                return

            picked = files[0]
            state["audio_name"] = picked.name or "meeting_audio.wav"
            state["audio_path"] = picked.path or ""
            state["audio_bytes"] = picked.bytes
            audio_path_input.value = f"已选择：{state['audio_name']}"
            audio_text.value = f"已选择音频：{state['audio_name']}"
            show_message(f"已选择音频：{state['audio_name']}")
            page.update()
        except Exception as exc:
            show_message(f"选择音频失败：{exc}")

    def choose_audio(_):
        page.run_task(choose_audio_async)

    def render_empty_results():
        summary_list.controls.clear()
        claims_list.controls.clear()
        action_list.controls.clear()
        risk_list.controls.clear()
        evidence_list.controls.clear()
        transcript_list.controls.clear()
        summary_list.controls.append(ft.Text("完成分析后展示发言人贡献。", color="#667085"))
        claims_list.controls.append(ft.Text("完成分析后展示可追溯结论。", color="#667085"))
        action_list.controls.append(ft.Text("完成分析后展示行动项。", color="#667085"))
        risk_list.controls.append(ft.Text("完成分析后展示风险项。", color="#667085"))
        evidence_list.controls.append(ft.Text("完成分析后展示证据链。", color="#667085"))
        transcript_list.controls.append(ft.Text("完成分析后展示原始转写。", color="#667085"))

    def render_history(history: list[dict]):
        history_list.controls.clear()

        if not history:
            history_list.controls.append(
                ft.Text(
                    "暂无任务历史",
                    color="#667085",
                )
            )
            page.update()
            return

        for item in history:
            progress_value = item.get(
                "progress",
                0,
            )

            status_value = item.get(
                "status",
                "-",
            )

            stage_value = item.get(
                "stage",
                "-",
            )
            timestamp_value = item.get(
                "timestamp",
                "",
            )

            history_list.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                f"{progress_value}%  {stage_value}",
                                size=13,
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.Text(
                                status_value,
                                size=11,
                                color="#667085",
                            ),
                            ft.Text(
                                timestamp_value,
                                size=10,
                                color="#98A2B3",
                            ),
                        ],
                        spacing=2,
                    ),
                    padding=10,
                    bgcolor="#F2F4F7",
                    border_radius=8,
                )
            )

        page.update()

    def apply_job_update(current: dict):
        state["job_status"] = current
        state["history"] = current.get("history", [])

        progress_value = int(current.get("progress", 0))
        stage_value = current.get("stage", "-")
        status_value = current.get("status", "-")

        progress_bar.value = progress_value / 100
        status_text.value = f"{progress_value}%  {stage_value}"
        audio_text.value = f"任务状态：{status_value}"
        render_history(state["history"])
        page.update()

    async def on_job_update(current: dict):
        apply_job_update(current)

    async def on_job_error(exc: Exception, attempt: int):
        current = state.get("job_status") or {}
        progress_value = int(current.get("progress", 0))
        stage_value = current.get("stage", "当前阶段")

        status_text.value = (
            f"{progress_value}%  状态查询暂时失败"
        )
        audio_text.value = (
            f"任务仍可能执行中：{stage_value}，正在第 {attempt} 次重试"
        )

        if attempt >= 5:
            show_message(
                "暂时无法刷新任务状态，后台任务可能仍在继续。"
                "请点击“刷新任务状态”或“继续等待分析”。"
            )
        else:
            show_message(
                f"状态查询暂时失败，将自动重试（{attempt}/5）：{exc}"
            )

    async def load_job_result(job: dict):
        if job.get("status") != "completed":
            return

        set_busy(True, "正在加载分析结果")
        try:
            result = await asyncio.to_thread(
                get_json,
                api_base.value.strip(),
                f"/jobs/{state['job_id']}/result",
            )
        except requests.RequestException as exc:
            show_message(
                f"任务已完成，但结果暂时加载失败：{exc}"
                "；请点击“刷新任务状态”后重试。"
            )
            return

        state["result"] = result
        render_result(result)
        status_text.value = "100%  分析完成"
        audio_text.value = "会议分析已完成"
        show_message("会议分析完成")

    async def handle_final_job(job: dict | None):
        if not job:
            show_message(
                "本次没有拿到最新任务状态，后台任务可能仍在执行。"
                "请点击“刷新任务状态”或“继续等待分析”。"
            )
            return

        apply_job_update(job)
        status_value = job.get("status")

        if status_value == "completed":
            await load_job_result(job)
        elif status_value == "failed":
            show_message(
                "会议分析失败：" + str(job.get("error") or "未知错误")
            )
        elif status_value == "cancelled":
            show_message("会议分析已取消")
        else:
            show_message(
                "任务仍在后台处理中，请点击“继续等待分析”查看后续进度。"
            )

    def render_result(result: dict):
        summary_list.controls.clear()
        claims_list.controls.clear()
        action_list.controls.clear()
        risk_list.controls.clear()
        evidence_list.controls.clear()
        transcript_list.controls.clear()

        summaries = result.get("speaker_summaries", [])
        if not summaries:
            summary_list.controls.append(ft.Text("暂无发言人贡献。", color="#667085"))

        for summary in summaries:
            speaker_name = summary.get("display_name") or summary.get("speaker_id")
            rows = [
                ft.Text(f"发言人：{speaker_name}", size=15, weight=ft.FontWeight.BOLD),
                ft.Text(f"贡献数量：{summary.get('claim_count', 0)}", size=12, color="#667085"),
            ]

            for point in summary.get("key_points", []):
                rows.append(
                    card(
                        ft.Column(
                            [
                                ft.Text(label_claim_type(point.get("claim_type")), weight=ft.FontWeight.BOLD, color="#175CD3"),
                                ft.Text(point.get("statement", ""), size=13),
                                ft.Text(f"证据：{', '.join(point.get('evidence_ids', []))}", size=11, color="#667085"),
                            ],
                            spacing=4,
                        ),
                        bgcolor="#F2F4F7",
                        padding=10,
                    )
                )

            summary_list.controls.append(card(ft.Column(rows, spacing=8)))

        claims = result.get("claims", [])
        action_items = result.get("action_items") or [
            claim for claim in claims if is_action_claim(claim)
        ]
        risks = result.get("risks") or [
            claim for claim in claims if is_risk_claim(claim)
        ]
        conclusions = [
            claim for claim in claims
            if claim not in action_items and claim not in risks
        ]

        if not conclusions:
            claims_list.controls.append(ft.Text("暂无可追溯结论。", color="#667085"))

        for claim in conclusions:
            claims_list.controls.append(
                card(
                    ft.Column(
                        [
                            ft.Text(
                                f"{label_claim_type(claim.get('claim_type'))} | {claim.get('speaker_id', '-')}",
                                size=15,
                                weight=ft.FontWeight.BOLD,
                                color="#101828",
                            ),
                            ft.Text(
                                f"{format_ms(claim.get('start_ms'))}-{format_ms(claim.get('end_ms'))}",
                                size=12,
                                color="#667085",
                            ),
                            ft.Text(f"总结：{claim.get('statement', '')}", size=13),
                            card(ft.Text(f"原话：{claim.get('quote', '')}", size=12, color="#344054"), bgcolor="#F9FAFB", padding=10),
                            ft.Text(
                                f"证据ID：{', '.join(claim.get('evidence_ids', []))}\n"
                                f"置信度：{claim.get('confidence', '-')}；审核状态：{claim.get('review_status', '-')}",
                                size=11,
                                color="#667085",
                            ),
                        ],
                        spacing=7,
                    )
                )
            )

        if not action_items:
            action_list.controls.append(ft.Text("暂无行动项。", color="#667085"))
        for index, claim in enumerate(action_items, start=1):
            action_list.controls.append(
                card(
                    ft.Column(
                        [
                            ft.Text(
                                f"行动项 {index} · {claim.get('speaker_id', '-')}",
                                size=15,
                                weight=ft.FontWeight.BOLD,
                                color="#067647",
                            ),
                            ft.Text(
                                claim.get("statement") or claim.get("summary", ""),
                                size=13,
                            ),
                            ft.Text(
                                f"证据：{', '.join(claim.get('evidence_ids', []))}\n"
                                f"支撑度：{claim.get('confidence', '-')}",
                                size=11,
                                color="#667085",
                            ),
                        ],
                        spacing=7,
                    ),
                    bgcolor="#ECFDF3",
                )
            )

        if not risks:
            risk_list.controls.append(ft.Text("暂无风险项。", color="#667085"))
        for index, claim in enumerate(risks, start=1):
            risk_list.controls.append(
                card(
                    ft.Column(
                        [
                            ft.Text(
                                f"风险 {index} · {claim.get('speaker_id', '-')}",
                                size=15,
                                weight=ft.FontWeight.BOLD,
                                color="#B42318",
                            ),
                            ft.Text(
                                claim.get("statement") or claim.get("summary", ""),
                                size=13,
                            ),
                            ft.Text(
                                f"证据：{', '.join(claim.get('evidence_ids', []))}\n"
                                f"支撑度：{claim.get('confidence', '-')}",
                                size=11,
                                color="#667085",
                            ),
                        ],
                        spacing=7,
                    ),
                    bgcolor="#FEF3F2",
                )
            )

        evidence = result.get("evidence_links", [])
        if not evidence:
            evidence_list.controls.append(ft.Text("暂无证据链。", color="#667085"))
        for item in evidence:
            evidence_list.controls.append(
                card(
                    ft.Column(
                        [
                            ft.Text(
                                item.get("evidence_id", "-"),
                                size=13,
                                weight=ft.FontWeight.BOLD,
                                color="#175CD3",
                            ),
                            ft.Text(
                                f"{item.get('speaker_id', '-')} · "
                                f"{format_ms(item.get('start_ms'))}-"
                                f"{format_ms(item.get('end_ms'))}",
                                size=11,
                                color="#667085",
                            ),
                            ft.Text(
                                item.get("quote") or item.get("text", ""),
                                size=13,
                                color="#344054",
                            ),
                            ft.Text(
                                f"ASR：{item.get('asr_confidence', '-')}"
                                f" · 声纹：{item.get('speaker_confidence', '-')}"
                                f" · 来源：{item.get('speaker_source', 'unknown')}",
                                size=10,
                                color="#667085",
                            ),
                        ],
                        spacing=5,
                    ),
                    bgcolor="#F9FAFB",
                )
            )

        spans = result.get("transcript_spans", [])
        if not spans:
            transcript_list.controls.append(ft.Text("暂无原始转写。", color="#667085"))

        for span in spans:
            transcript_list.controls.append(
                card(
                    ft.Text(
                        f"[{format_ms(span.get('start_ms'))}-{format_ms(span.get('end_ms'))}] "
                        f"{span.get('speaker_id', '-')}: {span.get('text', '')}",
                        size=12,
                        color="#344054",
                    ),
                    bgcolor="#FFFFFF",
                    padding=10,
                    )
                )

        speaker_ids = result.get("speaker_ids") or [
            item.get("speaker_id")
            for item in evidence
            if item.get("speaker_id")
        ]
        metrics_row.controls = [
            card(
                ft.Column(
                    [
                        ft.Text(str(len(set(speaker_ids))), size=19, weight=ft.FontWeight.BOLD, color="#175CD3"),
                        ft.Text("发言人数", size=10, color="#667085"),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=2,
                ),
                bgcolor="#EAF2FF",
                padding=9,
            ),
            card(
                ft.Column(
                    [
                        ft.Text(str(len(conclusions)), size=19, weight=ft.FontWeight.BOLD, color="#175CD3"),
                        ft.Text("关键结论", size=10, color="#667085"),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=2,
                ),
                bgcolor="#EAF2FF",
                padding=9,
            ),
            card(
                ft.Column(
                    [
                        ft.Text(str(len(action_items)), size=19, weight=ft.FontWeight.BOLD, color="#067647"),
                        ft.Text("行动项", size=10, color="#667085"),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=2,
                ),
                bgcolor="#ECFDF3",
                padding=9,
            ),
            card(
                ft.Column(
                    [
                        ft.Text(str(len(risks)), size=19, weight=ft.FontWeight.BOLD, color="#B42318"),
                        ft.Text("风险项", size=10, color="#667085"),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=2,
                ),
                bgcolor="#FEF3F2",
                padding=9,
            ),
        ]
        page.update()

    def create_meeting(_):
        show_message("正在创建会议...")
        try:
            set_busy(True, "正在创建会议")
            payload = {
                "title": title.value,
                "host": host.value,
                "language": "zh-CN",
                "participants": [
                    item.strip()
                    for item in participants.value.splitlines()
                    if item.strip()
                ],
            }
            meeting = post_json(api_base.value.strip(), "/meetings", payload)
            refresh_status(meeting)
            show_message("会议创建成功")
        except Exception as exc:
            show_message(f"创建失败：{exc}")
        finally:
            set_busy(False)

    def upload_audio(_):
        show_message("正在上传音频...")
        if not state["meeting_id"]:
            show_message("请先创建会议")
            return
        if not state["audio_bytes"] and not state["audio_path"]:
            show_message("请先选择音频文件")
            return

        try:
            set_busy(True, "正在上传音频")
            if state["audio_bytes"]:
                meeting = post_file_bytes(
                    api_base.value.strip(),
                    f"/meetings/{state['meeting_id']}/audio",
                    state["audio_name"] or "meeting_audio.wav",
                    state["audio_bytes"],
                )
            else:
                meeting = post_file(
                    api_base.value.strip(),
                    f"/meetings/{state['meeting_id']}/audio",
                    state["audio_path"],
                )
            refresh_status(meeting)
            show_message("音频上传成功")
        except Exception as exc:
            show_message(f"上传失败：{exc}")
        finally:
            set_busy(False)

    async def run_analysis_async():
        if state["polling"]:
            show_message("任务正在轮询，请勿重复启动分析")
            return

        audio_bytes = state["audio_bytes"]

        if not audio_bytes and state["audio_path"]:
            audio_bytes = await asyncio.to_thread(
                Path(state["audio_path"]).read_bytes
            )

        if not audio_bytes:
            show_message("请先选择音频文件")
            return

        api_url = api_base.value.strip()
        if not api_url:
            show_message("请填写后端地址")
            return

        try:
            set_busy(True, "正在创建分析任务")
            show_message("正在上传音频并创建后台任务...")

            job = await asyncio.to_thread(
                post_analyze,
                api_url,
                title.value.strip() or "未命名会议",
                host.value.strip() or "-",
                participants.value.strip(),
                state["audio_name"] or "meeting_audio.wav",
                audio_bytes,
            )

            state["job_id"] = job["job_id"]
            state["meeting_id"] = job["meeting_id"]
            state["job_status"] = job
            state["history"] = job.get("history", [])
            state["result"] = None

            meeting_id_text.value = state["meeting_id"]
            render_history(state["history"])
            show_message(
                f"任务已创建：{state['job_id']}"
            )

            state["polling"] = True
            final_job = await poll_job_async(
                api_url,
                state["job_id"],
                on_job_update,
                on_job_error,
            )

            await handle_final_job(final_job)

        except Exception as exc:
            show_message(f"分析失败：{exc}")
        finally:
            state["polling"] = False
            set_busy(False)

    def run_analysis(_):
        page.run_task(run_analysis_async)

    async def refresh_job_status_async():
        if not state["job_id"]:
            show_message("当前没有可刷新的分析任务")
            return

        if state["polling"]:
            show_message("任务正在自动轮询，请稍候")
            return

        try:
            set_busy(True, "正在刷新任务状态")
            current = await asyncio.to_thread(
                get_json,
                api_base.value.strip(),
                f"/jobs/{state['job_id']}",
            )
            apply_job_update(current)
            await handle_final_job(current)
        except requests.RequestException as exc:
            show_message(
                f"刷新失败：{exc}。任务可能仍在后台执行，请稍后再次刷新。"
            )
        except Exception as exc:
            show_message(f"刷新任务状态失败：{exc}")
        finally:
            set_busy(False)

    def refresh_job_status(_):
        page.run_task(refresh_job_status_async)

    async def resume_job_polling_async():
        if not state["job_id"]:
            show_message("当前没有可继续等待的分析任务")
            return

        if state["polling"]:
            show_message("任务正在轮询，请勿重复点击")
            return

        try:
            state["polling"] = True
            set_busy(True, "正在继续等待分析")
            show_message("已恢复任务轮询，不会重新上传音频")

            final_job = await poll_job_async(
                api_base.value.strip(),
                state["job_id"],
                on_job_update,
                on_job_error,
            )
            await handle_final_job(final_job)
        except Exception as exc:
            show_message(
                f"继续等待时发生临时错误：{exc}。"
                "任务仍可通过“刷新任务状态”继续查看。"
            )
        finally:
            state["polling"] = False
            set_busy(False)

    def resume_job_polling(_):
        page.run_task(resume_job_polling_async)

    def download_pdf(_):
        show_message("正在生成 PDF...")
        if not state["meeting_id"] and not state["job_id"]:
            show_message("请先完成会议分析")
            return

        try:
            set_busy(True, "正在生成 PDF")
            if state["job_id"]:
                pdf_api_path = (
                    f"/jobs/{state['job_id']}/report/pdf"
                )
            else:
                pdf_api_path = (
                    f"/meetings/{state['meeting_id']}"
                    "/report/pdf"
                )

            data = get_bytes(
                api_base.value.strip(),
                pdf_api_path,
            )
            output_dir = Path("outputs/reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir
                / f"{state['meeting_id']}_trusted_minutes.pdf"
            )
            output_path.write_bytes(data)
            show_message(f"PDF 已保存：{output_path}")
        except Exception as exc:
            show_message(f"PDF 生成失败：{exc}")
        finally:
            set_busy(False)

    def open_pdf_report(_):
        if not state["meeting_id"] and not state["job_id"]:
            show_message("请先完成会议分析")
            return

        if state["job_id"]:
            pdf_api_path = (
                f"/jobs/{state['job_id']}/report/pdf"
            )
        else:
            pdf_api_path = (
                f"/meetings/{state['meeting_id']}"
                "/report/pdf"
            )

        pdf_url = (
            f"{api_base.value.strip()}{pdf_api_path}"
        )
        page.launch_url(pdf_url)
        show_message("已打开 PDF 报告")
    def search_evidence(_):
        search_results.controls.clear()

        if not state["meeting_id"]:
            show_message("请先创建并分析会议")
            return

        query = search_query.value.strip()
        if not query:
            show_message("请输入检索关键词")
            return

        try:
            set_busy(True, "正在检索会议证据")
            data = get_json(
                api_base.value.strip(),
                f"/meetings/{state['meeting_id']}/search"
                f"?q={quote(query)}&limit=8",
            )

            hits = data.get("hits", [])
            if not hits:
                search_results.controls.append(
                    ft.Text("没有找到匹配的原话证据", color="#667085")
                )

            for hit in hits:
                payload = hit.get("payload", {})
                search_results.controls.append(
                    card(
                        ft.Column(
                            [
                                ft.Text(
                                    f"相似度：{hit.get('score', '-')}",
                                    size=11,
                                    color="#667085",
                                ),
                                ft.Text(
                                    f"发言人：{payload.get('speaker_id', '-')}",
                                    size=12,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                ft.Text(
                                    f"时间：{format_ms(payload.get('start_ms'))}"
                                    f" - {format_ms(payload.get('end_ms'))}",
                                    size=11,
                                    color="#667085",
                                ),
                                ft.Text(
                                    payload.get("text", ""),
                                    size=13,
                                    color="#344054",
                                ),
                            ],
                            spacing=5,
                        ),
                        bgcolor="#F9FAFB",
                        padding=10,
                    )
                )

            page.update()
            show_message("证据检索完成")
        except Exception as exc:
            show_message(f"证据检索失败：{exc}")
        finally:
            set_busy(False)

    render_empty_results()

    header = ft.Container(
        content=ft.Column(
            [
                ft.Text("可信会议纪要", size=26, weight=ft.FontWeight.BOLD, color="#101828"),
                ft.Text("真实会议发言归因、关键贡献抽取、可信 PDF 生成", size=13, color="#475467"),
            ],
            spacing=4,
        ),
        padding=18,
        bgcolor="#FFFFFF",
        border_radius=0,
    )

    status_card = card(
        ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("当前状态", size=15, weight=ft.FontWeight.BOLD),
                        progress,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                status_text,
                progress_bar,
                ft.Text("会议ID", size=11, color="#667085"),
                meeting_id_text,
                ft.Text("音频", size=11, color="#667085"),
                audio_text,
                debug_log,
                metrics_row,
            ],
            spacing=8,
        ),
        bgcolor="#EAF2FF",
    )

    form_card = card(
        ft.Column(
            [
                section_title("1. 会议信息"),
                api_base,
                title,
                host,
                participants,
                ft.ElevatedButton("创建会议", on_click=create_meeting, expand=True),
            ],
            spacing=10,
        )
    )

    audio_card = card(
        ft.Column(
            [
                section_title("2. 音频上传"),
                audio_path_input,
                ft.Row(
                    [
                        ft.ElevatedButton("选择音频文件", on_click=choose_audio, expand=True),
                        ft.ElevatedButton("上传已选音频", on_click=upload_audio, expand=True),
                    ],
                    spacing=10,
                ),
            ],
            spacing=10,
        )
    )

    action_card = card(
        ft.Column(
            [
                section_title("3. 生成结果"),
                ft.ElevatedButton("一键分析会议", on_click=run_analysis, expand=True),
                ft.Row(
                    [
                        ft.OutlinedButton("刷新任务状态", on_click=refresh_job_status, expand=True),
                        ft.OutlinedButton("继续等待分析", on_click=resume_job_polling, expand=True),
                    ],
                    spacing=8,
                ),
                ft.Row(
                    [
                        ft.OutlinedButton("生成 PDF 报告", on_click=download_pdf, expand=True),
                        ft.OutlinedButton("打开 PDF 报告", on_click=open_pdf_report, expand=True),
                    ],
                    spacing=8,
                ),
            ],
            spacing=10,
        )
    )

    search_card = card(
        ft.Column(
            [
                section_title("4. 证据检索"),
                search_query,
                ft.ElevatedButton(
                    "搜索原话证据",
                    on_click=search_evidence,
                    expand=True,
                ),
                search_results,
            ],
            spacing=10,
        )
    )

    page.add(
        header,
        ft.Container(
            content=ft.Column(
                [
                    status_card,
                    form_card,
                    audio_card,
                    action_card,
                    section_title("任务历史"),
                    history_list,
                    ft.Tabs(
                        length=5,
                        selected_index=0,
                        animation_duration=200,
                        content=ft.Column(
                            [
                                ft.TabBar(
                                    tabs=[
                                        ft.Tab(label="摘要"),
                                        ft.Tab(label="行动项"),
                                        ft.Tab(label="风险项"),
                                        ft.Tab(label="证据链"),
                                        ft.Tab(label="转写"),
                                    ],
                                    scrollable=True,
                                ),
                                ft.TabBarView(
                                    expand=True,
                                    controls=[
                                        ft.Container(
                                            content=ft.Column(
                                                [summary_list, claims_list],
                                                spacing=12,
                                            ),
                                            padding=ft.Padding(
                                                top=10, right=0, bottom=0, left=0
                                            ),
                                        ),
                                        ft.Container(
                                            content=action_list,
                                            padding=ft.Padding(
                                                top=10, right=0, bottom=0, left=0
                                            ),
                                        ),
                                        ft.Container(
                                            content=risk_list,
                                            padding=ft.Padding(
                                                top=10, right=0, bottom=0, left=0
                                            ),
                                        ),
                                        ft.Container(
                                            content=ft.Column(
                                                [search_card, evidence_list],
                                                spacing=12,
                                            ),
                                            padding=ft.Padding(
                                                top=10, right=0, bottom=0, left=0
                                            ),
                                        ),
                                        ft.Container(
                                            content=transcript_list,
                                            padding=ft.Padding(
                                                top=10, right=0, bottom=0, left=0
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                            expand=True,
                        ),
                        height=560,
                    ),
                ],
                spacing=14,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            padding=14,
        ),
    )


ft.app(target=main)
