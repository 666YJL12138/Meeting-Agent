from services.email_sender import send_email_with_attachment


def _looks_like_question_mark_loss(text: str | None) -> bool:
    if not text:
        return False
    return text.count("?") >= 3 or "？" * 3 in text


def send_report_email_tool(
    meeting_id: str,
    pdf_path: str,
    to_email: str,
    subject: str | None = None,
    body: str | None = None,
) -> dict:
    if _looks_like_question_mark_loss(subject):
        subject = None
    if _looks_like_question_mark_loss(body):
        body = None

    subject = subject or f"会议可信纪要 PDF - {meeting_id}"
    body = body or (
        "您好，附件是本次会议的可信纪要 PDF。\n"
        "该 PDF 中的结论均绑定原话证据、说话人和时间戳。"
    )

    return send_email_with_attachment(
        to_email=to_email,
        subject=subject,
        body=body,
        attachment_path=pdf_path,
    )
