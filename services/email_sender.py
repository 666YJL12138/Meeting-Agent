import os
import smtplib
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path

from dotenv import load_dotenv
from email_validator import validate_email


load_dotenv()


class EmailSendError(RuntimeError):
    pass


def send_email_with_attachment(
    to_email: str,
    subject: str,
    body: str,
    attachment_path: str,
) -> dict:
    email_info = validate_email(to_email, check_deliverability=False)
    normalized_email = email_info.normalized

    file_path = Path(attachment_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Attachment not found: {attachment_path}")

    message = EmailMessage(policy=SMTP)
    message["From"] = os.getenv("SMTP_FROM", os.getenv("SMTP_USER"))
    message["To"] = normalized_email
    message["Subject"] = subject
    message.set_content(body, subtype="plain", charset="utf-8")

    message.add_attachment(
        file_path.read_bytes(),
        maintype="application",
        subtype="pdf",
        filename=file_path.name,
    )

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "465"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")

    missing = [
        name
        for name, value in {
            "SMTP_HOST": host,
            "SMTP_USER": user,
            "SMTP_PASSWORD": password,
            "SMTP_FROM": message["From"],
        }.items()
        if not value
    ]
    if missing:
        raise EmailSendError(f"Missing SMTP config: {', '.join(missing)}")

    try:
        with smtplib.SMTP_SSL(host, port) as smtp:
            smtp.login(user, password)
            smtp.send_message(message)
    except Exception as exc:
        raise EmailSendError(f"SMTP send failed: {exc}") from exc

    return {
        "status": "sent",
        "to_email": normalized_email,
        "attachment": str(file_path),
    }
