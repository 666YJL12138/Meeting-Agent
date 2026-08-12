from pathlib import Path

import flet as ft
import requests


DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_AUDIO_PATH = r"D:\futurework\Meeting-Agent\Meeting-Agent\day2_chinese_meeting_sample.wav"

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


def get_bytes(api_base: str, path: str) -> bytes:
    response = requests.get(f"{api_base}{path}", timeout=120)
    response.raise_for_status()
    return response.content


def main(page: ft.Page):
    page.title = "可信会议纪要"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = "#F6F8FB"
    page.padding = 0
    page.scroll = ft.ScrollMode.AUTO

    state = {
        "meeting_id": "",
        "audio_path": "",
        "result": None,
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
    audio_path_input = ft.TextField(label="音频文件路径", value=DEFAULT_AUDIO_PATH)

    meeting_id_text = ft.Text("未创建", size=12, color="#344054")
    status_text = ft.Text("等待创建会议", size=14, weight=ft.FontWeight.BOLD, color="#175CD3")
    audio_text = ft.Text("请填写本地音频路径", size=12, color="#667085")
    debug_log = ft.Text("准备就绪", size=12, color="#175CD3")
    progress = ft.ProgressRing(visible=False)

    summary_list = ft.Column(spacing=10)
    claims_list = ft.Column(spacing=10)
    transcript_list = ft.Column(spacing=6)

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
        audio_text.value = "音频已上传" if meeting.get("audio_uri") else "请填写本地音频路径"

    def render_empty_results():
        summary_list.controls.clear()
        claims_list.controls.clear()
        transcript_list.controls.clear()
        summary_list.controls.append(ft.Text("完成分析后展示发言人贡献。", color="#667085"))
        claims_list.controls.append(ft.Text("完成分析后展示可追溯结论。", color="#667085"))
        transcript_list.controls.append(ft.Text("完成分析后展示原始转写。", color="#667085"))

    def render_result(result: dict):
        summary_list.controls.clear()
        claims_list.controls.clear()
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
        if not claims:
            claims_list.controls.append(ft.Text("暂无可追溯结论。", color="#667085"))

        for claim in claims:
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
        state["audio_path"] = audio_path_input.value.strip()
        if not state["audio_path"]:
            show_message("请先填写音频文件路径")
            return
        if not Path(state["audio_path"]).exists():
            show_message(f"音频文件不存在：{state['audio_path']}")
            return

        try:
            set_busy(True, "正在上传音频")
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

    def run_analysis(_):
        show_message("正在分析会议...")
        if not state["meeting_id"]:
            show_message("请先创建会议")
            return

        try:
            set_busy(True, "正在分析会议")
            result = post_json(
                api_base.value.strip(),
                f"/meetings/{state['meeting_id']}/run-asr",
            )
            state["result"] = result
            refresh_status(result)
            render_result(result)
            show_message("会议分析完成")
        except Exception as exc:
            show_message(f"分析失败：{exc}")
        finally:
            set_busy(False)

    def download_pdf(_):
        show_message("正在生成 PDF...")
        if not state["meeting_id"]:
            show_message("请先创建会议")
            return

        try:
            set_busy(True, "正在生成 PDF")
            data = get_bytes(
                api_base.value.strip(),
                f"/meetings/{state['meeting_id']}/report/pdf",
            )
            output_dir = Path("outputs/reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{state['meeting_id']}_trusted_minutes.pdf"
            output_path.write_bytes(data)
            show_message(f"PDF 已保存：{output_path}")
        except Exception as exc:
            show_message(f"PDF 生成失败：{exc}")
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
                ft.Text("会议ID", size=11, color="#667085"),
                meeting_id_text,
                ft.Text("音频", size=11, color="#667085"),
                audio_text,
                debug_log,
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
                ft.ElevatedButton("创建会议", on_click=create_meeting, width=360),
            ],
            spacing=10,
        )
    )

    audio_card = card(
        ft.Column(
            [
                section_title("2. 音频上传"),
                audio_path_input,
                ft.ElevatedButton("上传音频", on_click=upload_audio, width=360),
            ],
            spacing=10,
        )
    )

    action_card = card(
        ft.Column(
            [
                section_title("3. 生成结果"),
                ft.ElevatedButton("运行会议分析", on_click=run_analysis, width=360),
                ft.OutlinedButton("生成 PDF 报告", on_click=download_pdf, width=360),
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
                    section_title("发言人贡献"),
                    summary_list,
                    section_title("可追溯结论"),
                    claims_list,
                    section_title("原始转写"),
                    transcript_list,
                ],
                spacing=14,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            padding=14,
        ),
    )


ft.app(target=main)
