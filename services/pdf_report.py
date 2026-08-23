from pathlib import Path
from html import escape
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


FONT_NAME = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def format_ms(value: int | float | None) -> str:
    if value is None:
        return "-"
    seconds = int(value // 1000)
    minute = seconds // 60
    second = seconds % 60
    return f"{minute:02d}:{second:02d}"


def safe_text(value) -> str:
    if value is None:
        return ""
    return escape(str(value))


def format_confidence_breakdown(value: dict | None) -> str:
    if not isinstance(value, dict) or not value:
        return "-"

    labels = {
        "evidence_binding": "证据绑定",
        "quote_match": "原话匹配",
        "speaker_match": "说话人匹配",
        "time_match": "时间匹配",
        "asr_quality": "ASR质量",
        "diarization_quality": "声纹质量",
    }

    parts = []
    for key, label in labels.items():
        if key not in value:
            continue
        try:
            parts.append(f"{label}={float(value[key]):.2f}")
        except (TypeError, ValueError):
            parts.append(f"{label}={safe_text(value[key])}")

    return "；".join(parts) or "-"


def build_styles():
    styles = getSampleStyleSheet()

    styles["Title"].fontName = FONT_NAME
    styles["Title"].fontSize = 22
    styles["Title"].leading = 30
    styles["Title"].alignment = TA_CENTER

    styles["Heading1"].fontName = FONT_NAME
    styles["Heading1"].fontSize = 15
    styles["Heading1"].leading = 22
    styles["Heading1"].spaceBefore = 14
    styles["Heading1"].spaceAfter = 8

    styles["Heading2"].fontName = FONT_NAME
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 18
    styles["Heading2"].spaceBefore = 8
    styles["Heading2"].spaceAfter = 6

    styles["BodyText"].fontName = FONT_NAME
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 16

    styles.add(ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=8,
        leading=12,
        textColor=colors.HexColor("#555555"),
    ))

    styles.add(ParagraphStyle(
        name="Quote",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=9,
        leading=14,
        leftIndent=8,
        textColor=colors.HexColor("#333333"),
    ))

    return styles


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(200 * mm, 12 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def generate_meeting_pdf(meeting: dict) -> str:
    meeting_id = meeting["meeting_id"]
    output_path = REPORT_DIR / f"{meeting_id}_trusted_minutes.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
    )

    styles = build_styles()
    story = []

    story.append(Paragraph("可信会议纪要", styles["Title"]))
    story.append(Spacer(1, 10))

    audio_info = meeting.get("audio_info") or {}
    duration = audio_info.get("duration_seconds")

    meta_rows = [
        ["会议ID", safe_text(meeting_id)],
        ["会议标题", safe_text(meeting.get("title", "未命名会议"))],
        ["主持人", safe_text(meeting.get("host", "-"))],
        ["语言", safe_text(meeting.get("language", "zh-CN"))],
        ["会议状态", safe_text(meeting.get("status", "-"))],
        ["音频时长", f"{duration:.2f} 秒" if duration else "-"],
        ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]

    meta_table = Table(meta_rows, colWidths=[30 * mm, 130 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F4F7")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D5DD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "评分说明：证据支撑度由系统根据证据绑定、原话匹配、"
        "说话人/时间一致性、ASR质量和声纹归因质量计算，"
        "不是语言模型自报的概率。",
        styles["Small"],
    ))

    story.append(Paragraph("一、发言人关键贡献概览", styles["Heading1"]))

    summaries = meeting.get("speaker_summaries") or []
    if not summaries:
        story.append(Paragraph("暂无发言人摘要，请先运行 /run-asr。", styles["BodyText"]))
    else:
        for summary in summaries:
            speaker_name = summary.get("display_name") or summary.get("speaker_id")
            story.append(Paragraph(f"发言人：{safe_text(speaker_name)}", styles["Heading2"]))

            key_points = summary.get("key_points") or []
            for point in key_points:
                evidence_ids = ", ".join(point.get("evidence_ids", []))
                text = (
                    f"类型：{safe_text(point.get('claim_type'))}；"
                    f"证据支撑度：{point.get('confidence', '-')}; "
                    f"证据：{safe_text(evidence_ids)}<br/>"
                    f"评分明细：{safe_text(format_confidence_breakdown(point.get('confidence_breakdown')))}<br/>"
                    f"{safe_text(point.get('statement'))}"
                )
                story.append(Paragraph(text, styles["BodyText"]))
                story.append(Spacer(1, 4))

    story.append(PageBreak())
    story.append(Paragraph("二、可追溯关键结论明细", styles["Heading1"]))

    claims = meeting.get("claims") or []
    if not claims:
        story.append(Paragraph("暂无关键结论。", styles["BodyText"]))
    else:
        for index, claim in enumerate(claims, start=1):
            title = (
                f"{index}. {safe_text(claim.get('claim_type'))} | "
                f"{safe_text(claim.get('speaker_id'))} | "
                f"{format_ms(claim.get('start_ms'))}-{format_ms(claim.get('end_ms'))}"
            )
            story.append(Paragraph(title, styles["Heading2"]))

            story.append(Paragraph(
                f"总结：{safe_text(claim.get('statement'))}",
                styles["BodyText"],
            ))

            story.append(Paragraph(
                f"原话：{safe_text(claim.get('quote'))}",
                styles["Quote"],
            ))

            story.append(Paragraph(
                f"证据ID：{safe_text(', '.join(claim.get('evidence_ids', [])))}；"
                f"证据支撑度：{claim.get('confidence', '-')}; "
                f"评分明细：{safe_text(format_confidence_breakdown(claim.get('confidence_breakdown')))}；"
                f"审核状态：{safe_text(claim.get('review_status'))}",
                styles["Small"],
            ))
            story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("三、原始证据片段附录", styles["Heading1"]))

    for evidence in meeting.get("evidence_links") or []:
        line = (
            f"{safe_text(evidence.get('evidence_id'))} | "
            f"{safe_text(evidence.get('speaker_id'))} | "
            f"{format_ms(evidence.get('start_ms'))}-{format_ms(evidence.get('end_ms'))}<br/>"
            f"{safe_text(evidence.get('quote'))}"
        )
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 5))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    return str(output_path)
