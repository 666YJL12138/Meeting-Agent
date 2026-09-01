from datetime import datetime
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from services.diarization_evaluation_view import (
    evaluation_metric_rows,
    evaluation_reason_text,
    evaluation_status_text,
)


FONT_NAME = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#17324D")
LIGHT_BLUE = colors.HexColor("#EAF2FF")
LIGHT_GRAY = colors.HexColor("#F5F7FA")
BORDER = colors.HexColor("#D0D5DD")
TEXT = colors.HexColor("#1D2939")
MUTED = colors.HexColor("#667085")


def format_ms(value: int | float | None) -> str:
    if value is None:
        return "-"
    seconds = max(0, int(float(value) // 1000))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def safe_text(value) -> str:
    if value is None:
        return ""
    return escape(str(value)).replace("\n", "<br/>")


def format_confidence(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


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


def _review_reason_text(value) -> str:
    if not value:
        return "-"
    if isinstance(value, list):
        return "；".join(str(item) for item in value if item)
    return str(value)


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        fontName=FONT_NAME,
        fontSize=24,
        leading=32,
        alignment=TA_CENTER,
        textColor=NAVY,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        fontName=FONT_NAME,
        fontSize=10,
        leading=16,
        alignment=TA_CENTER,
        textColor=MUTED,
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="Section",
        fontName=FONT_NAME,
        fontSize=15,
        leading=22,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="Body",
        fontName=FONT_NAME,
        fontSize=9.5,
        leading=15,
        textColor=TEXT,
    ))
    styles.add(ParagraphStyle(
        name="Small",
        fontName=FONT_NAME,
        fontSize=7.5,
        leading=11,
        textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        name="Table",
        fontName=FONT_NAME,
        fontSize=7.8,
        leading=11,
        textColor=TEXT,
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        fontName=FONT_NAME,
        fontSize=7.8,
        leading=11,
        textColor=colors.white,
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="MetricValue",
        fontName=FONT_NAME,
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        textColor=NAVY,
    ))
    styles.add(ParagraphStyle(
        name="MetricLabel",
        fontName=FONT_NAME,
        fontSize=7.5,
        leading=10,
        alignment=TA_CENTER,
        textColor=MUTED,
    ))
    return styles


def _p(value, style) -> Paragraph:
    return Paragraph(safe_text(value), style)


def _p_html(value, style) -> Paragraph:
    """Build a Paragraph from already escaped text with controlled breaks."""
    return Paragraph(str(value), style)


def _speaker_name(item: dict, speaker_mapping: dict[str, str]) -> str:
    speaker_id = item.get("speaker_id", "-")
    return (
        item.get("speaker_name")
        or item.get("display_name")
        or speaker_mapping.get(speaker_id)
        or speaker_id
    )


def _claim_type(claim: dict) -> str:
    return str(claim.get("claim_type") or "关键结论")


def _is_action(claim: dict) -> bool:
    value = _claim_type(claim).lower()
    return any(token in value for token in ("action", "待办", "行动", "任务"))


def _is_risk(claim: dict) -> bool:
    value = _claim_type(claim).lower()
    return "risk" in value or "风险" in value


def _is_conclusion(claim: dict) -> bool:
    return not _is_action(claim) and not _is_risk(claim)


def _table_style(header_color=NAVY, right_align_columns=()) -> TableStyle:
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for column in right_align_columns:
        commands.append(("ALIGN", (column, 1), (column, -1), "RIGHT"))
    return TableStyle(commands)


def _metadata_table(meeting: dict, styles) -> Table:
    audio_info = meeting.get("audio_info") or {}
    duration = audio_info.get("duration_seconds")
    rows = [
        ["会议标题", meeting.get("title", "未命名会议")],
        ["主持人", meeting.get("host", "-")],
        ["会议ID", meeting.get("meeting_id", "-")],
        [
            "语言 / 状态",
            f"{meeting.get('language', 'zh-CN')} / {meeting.get('status', '-')}",
        ],
        ["音频时长", f"{float(duration):.2f} 秒" if duration else "-"],
        ["报告生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    table = Table(
        [[_p(left, styles["Table"]), _p(right, styles["Table"])] for left, right in rows],
        colWidths=[32 * mm, 128 * mm],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _metric_table(claims: list[dict], evidence: list[dict], styles) -> Table:
    speakers = {
        item.get("speaker_id")
        for item in claims + evidence
        if item.get("speaker_id")
    }
    values = []
    confidence_values = []
    for claim in claims:
        try:
            confidence_values.append(float(claim.get("confidence")))
        except (TypeError, ValueError):
            continue
    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0
    )
    values.extend([
        ("发言人数", len(speakers)),
        ("关键结论", sum(_is_conclusion(item) for item in claims)),
        ("行动项", sum(_is_action(item) for item in claims)),
        ("风险项", sum(_is_risk(item) for item in claims)),
        ("证据片段", len(evidence)),
        ("平均支撑度", format_confidence(average_confidence)),
    ])
    table = Table(
        [
            [_p(value, styles["MetricValue"]) for label, value in values],
            [_p(label, styles["MetricLabel"]) for label, value in values],
        ],
        colWidths=[29 * mm] * len(values),
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("BOX", (0, 0), (-1, -1), 0.4, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def _claims_table(claims: list[dict], styles, speaker_mapping: dict[str, str]) -> Table:
    rows = [[
        _p("类型", styles["TableHeader"]),
        _p("发言人", styles["TableHeader"]),
        _p("结论摘要", styles["TableHeader"]),
        _p("原话", styles["TableHeader"]),
        _p("证据ID", styles["TableHeader"]),
        _p("支撑度", styles["TableHeader"]),
    ]]
    for claim in claims:
        rows.append([
            _p(_claim_type(claim), styles["Table"]),
            _p(_speaker_name(claim, speaker_mapping), styles["Table"]),
            _p(claim.get("statement") or claim.get("summary", "-"), styles["Table"]),
            _p(claim.get("quote", "-"), styles["Table"]),
            _p(", ".join(claim.get("evidence_ids") or []) or "-", styles["Table"]),
            _p_html(
                f"{format_confidence(claim.get('confidence'))}<br/>"
                f"{format_confidence_breakdown(claim.get('confidence_breakdown'))}",
                styles["Table"],
            ),
        ])
    table = Table(
        rows,
        colWidths=[19 * mm, 22 * mm, 39 * mm, 46 * mm, 25 * mm, 29 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(_table_style(right_align_columns=(5,)))
    return table


def _action_or_risk_table(
    claims: list[dict],
    styles,
    speaker_mapping: dict[str, str],
    risk=False,
) -> Table:
    rows = [[
        _p("负责人" if not risk else "风险来源", styles["TableHeader"]),
        _p("行动项" if not risk else "风险描述", styles["TableHeader"]),
        _p("时间", styles["TableHeader"]),
        _p("证据", styles["TableHeader"]),
        _p("支撑度", styles["TableHeader"]),
    ]]
    for claim in claims:
        rows.append([
            _p(_speaker_name(claim, speaker_mapping), styles["Table"]),
            _p(claim.get("statement") or claim.get("summary", "-"), styles["Table"]),
            _p(
                f"{format_ms(claim.get('start_ms'))}-"
                f"{format_ms(claim.get('end_ms'))}",
                styles["Table"],
            ),
            _p(", ".join(claim.get("evidence_ids") or []) or "-", styles["Table"]),
            _p(format_confidence(claim.get("confidence")), styles["Table"]),
        ])
    table = Table(
        rows,
        colWidths=[25 * mm, 78 * mm, 27 * mm, 35 * mm, 18 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(_table_style(
        header_color=colors.HexColor("#7A271A") if risk else NAVY,
        right_align_columns=(4,),
    ))
    return table


def _evidence_table(
    evidence: list[dict],
    styles,
    speaker_mapping: dict[str, str],
) -> Table:
    rows = [[
        _p("证据ID", styles["TableHeader"]),
        _p("发言人", styles["TableHeader"]),
        _p("时间", styles["TableHeader"]),
        _p("原话", styles["TableHeader"]),
        _p("ASR / 声纹", styles["TableHeader"]),
    ]]
    for item in evidence:
        rows.append([
            _p(item.get("evidence_id", "-"), styles["Table"]),
            _p(_speaker_name(item, speaker_mapping), styles["Table"]),
            _p(
                f"{format_ms(item.get('start_ms'))}-"
                f"{format_ms(item.get('end_ms'))}",
                styles["Table"],
            ),
            _p(item.get("quote") or item.get("text", "-"), styles["Table"]),
            _p_html(
                f"{format_confidence(item.get('asr_confidence'))} / "
                f"{format_confidence(item.get('speaker_confidence'))}<br/>"
                f"{safe_text(item.get('speaker_source', 'unknown'))}",
                styles["Table"],
            ),
        ])
    table = Table(
        rows,
        colWidths=[34 * mm, 22 * mm, 27 * mm, 57 * mm, 21 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(_table_style())
    return table


def _diarization_evaluation_table(
    evaluation: dict,
    styles,
) -> Table:
    rows = []
    metric_rows = evaluation_metric_rows(evaluation)
    for index in range(0, len(metric_rows), 2):
        left_label, left_value = metric_rows[index]
        row = [
            _p(left_label, styles["Table"]),
            _p(left_value, styles["Table"]),
        ]
        if index + 1 < len(metric_rows):
            right_label, right_value = metric_rows[index + 1]
            row.extend([
                _p(right_label, styles["Table"]),
                _p(right_value, styles["Table"]),
            ])
        else:
            row.extend([_p("", styles["Table"]), _p("", styles["Table"])])
        rows.append(row)

    table = Table(
        rows,
        colWidths=[38 * mm, 42 * mm, 38 * mm, 42 * mm],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("ALIGN", (3, 0), (3, -1), "RIGHT"),
    ]))
    return table


def _draw_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, height - 13 * mm, width - 18 * mm, height - 13 * mm)
    canvas.setFont(FONT_NAME, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, height - 10 * mm, "可信会议纪要")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"第 {doc.page} 页")
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.restoreState()


def generate_meeting_pdf(meeting: dict) -> str:
    meeting_id = meeting["meeting_id"]
    output_path = REPORT_DIR / f"{meeting_id}_trusted_minutes.pdf"
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=19 * mm,
        title=str(meeting.get("title", "可信会议纪要")),
        author="Meeting-Agent",
    )
    styles = build_styles()
    claims = meeting.get("claims") or []
    evidence = meeting.get("evidence_links") or []
    speaker_mapping = meeting.get("speaker_mapping") or {}
    conclusions = [item for item in claims if _is_conclusion(item)]
    actions = [item for item in claims if _is_action(item)]
    risks = [item for item in claims if _is_risk(item)]
    speakers = sorted({
        item.get("speaker_id")
        for item in claims + evidence
        if item.get("speaker_id")
    })
    diarization_evaluation = meeting.get("diarization_evaluation") or {}

    story = [
        Spacer(1, 8 * mm),
        Paragraph("可信会议纪要", styles["ReportTitle"]),
        Paragraph(
            "Meeting-Agent · Evidence-backed Meeting Report",
            styles["ReportSubtitle"],
        ),
        _metadata_table(meeting, styles),
        Spacer(1, 6 * mm),
        Paragraph("报告指标", styles["Section"]),
        _metric_table(claims, evidence, styles),
        Spacer(1, 5 * mm),
        Paragraph(
            "本报告由语音识别、说话人分离、多 Agent 信息抽取和证据链校验流程生成。"
            "关键内容均尽量保留原始证据、说话人和时间范围，置信度为证据加权支撑度，"
            "不代表语言模型自报概率。",
            styles["Small"],
        ),
        Paragraph("说话人归因评估", styles["Section"]),
        Paragraph(
            f"评估状态：{safe_text(evaluation_status_text(diarization_evaluation))}"
            + (
                f"；{safe_text(evaluation_reason_text(diarization_evaluation))}"
                if evaluation_reason_text(diarization_evaluation)
                else ""
            ),
            styles["Body"],
        ),
        (
            _diarization_evaluation_table(diarization_evaluation, styles)
            if diarization_evaluation.get("evaluation_status") == "available"
            else Paragraph(
                "当前结果没有可用于离线评估的参考标注。",
                styles["Small"],
            )
        ),
        Spacer(1, 5 * mm),
        Paragraph("一、执行摘要", styles["Section"]),
        Paragraph(
            f"本次会议共识别 {len(speakers)} 位发言人，形成 "
            f"{len(conclusions)} 条关键结论、{len(actions)} 条行动项和 "
            f"{len(risks)} 条风险项，关联 {len(evidence)} 个证据片段。",
            styles["Body"],
        ),
        Paragraph(
            f"复核原因：{safe_text(_review_reason_text(meeting.get('review_reasons')))}",
            styles["Small"],
        ),
        Paragraph(
            f"发言人：{safe_text('、'.join(speakers))}" if speakers else "发言人：-",
            styles["Body"],
        ),
        PageBreak(),
        Paragraph("二、发言人关键贡献", styles["Section"]),
    ]

    summaries = meeting.get("speaker_summaries") or []
    if summaries:
        contribution_rows = [[
            _p("发言人", styles["TableHeader"]),
            _p("类型", styles["TableHeader"]),
            _p("关键贡献", styles["TableHeader"]),
            _p("证据", styles["TableHeader"]),
            _p("支撑度", styles["TableHeader"]),
        ]]
        for summary in summaries:
            speaker_name = _speaker_name(summary, speaker_mapping)
            for point in summary.get("key_points") or []:
                contribution_rows.append([
                    _p(speaker_name, styles["Table"]),
                    _p(_claim_type(point), styles["Table"]),
                    _p(point.get("statement") or point.get("summary", "-"), styles["Table"]),
                    _p(", ".join(point.get("evidence_ids") or []) or "-", styles["Table"]),
                    _p_html(
                        f"{format_confidence(point.get('confidence'))}<br/>"
                        f"{safe_text(format_confidence_breakdown(point.get('confidence_breakdown')))}",
                        styles["Table"],
                    ),
                ])
        contribution_table = Table(
            contribution_rows,
            colWidths=[25 * mm, 21 * mm, 65 * mm, 28 * mm, 30 * mm],
            repeatRows=1,
            hAlign="LEFT",
        )
        contribution_table.setStyle(_table_style(right_align_columns=(4,)))
        story.append(contribution_table)
    else:
        story.append(Paragraph("暂无发言人贡献摘要。", styles["Body"]))

    story.extend([
        PageBreak(),
        Paragraph("三、关键结论", styles["Section"]),
        (
            _claims_table(conclusions, styles, speaker_mapping)
            if conclusions
            else Paragraph("暂无关键结论。", styles["Body"])
        ),
    ])
    story.extend([
        PageBreak(),
        Paragraph("四、行动项", styles["Section"]),
        (
            _action_or_risk_table(actions, styles, speaker_mapping)
            if actions
            else Paragraph("暂无行动项。", styles["Body"])
        ),
        Paragraph("五、风险项", styles["Section"]),
        (
            _action_or_risk_table(risks, styles, speaker_mapping, risk=True)
            if risks
            else Paragraph("暂无风险项。", styles["Body"])
        ),
        PageBreak(),
        Paragraph("六、证据链附录", styles["Section"]),
        Paragraph(
            "下表保留原始证据片段、时间范围、ASR 置信度、说话人置信度和归因来源，"
            "可用于复核报告中的结论、行动项与风险项。",
            styles["Small"],
        ),
        Spacer(1, 3 * mm),
        (
            _evidence_table(evidence, styles, speaker_mapping)
            if evidence
            else Paragraph("暂无证据片段。", styles["Body"])
        ),
        Spacer(1, 7 * mm),
        Paragraph(
            f"审核状态：{safe_text(meeting.get('status', '-'))}　"
            f"证据条数：{len(evidence)}　生成引擎：Meeting-Agent",
            styles["Small"],
        ),
    ])

    doc.build(
        story,
        onFirstPage=_draw_header_footer,
        onLaterPages=_draw_header_footer,
    )
    return str(output_path)
