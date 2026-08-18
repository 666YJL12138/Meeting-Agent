from tools.email_tools import send_report_email_tool


class NotificationAgent:
    def run(self, state: dict) -> dict | None:
        """Send the generated PDF only when the workflow explicitly requests it."""
        if not state.get("send_email"):
            return None

        to_email = state.get("to_email")
        pdf_path = state.get("pdf_path")
        if not to_email:
            raise ValueError("to_email is required when send_email=true")
        if not pdf_path:
            raise ValueError("pdf_path is required before sending email")

        return send_report_email_tool(
            meeting_id=state["meeting_id"],
            pdf_path=pdf_path,
            to_email=to_email,
            subject=state.get("email_subject"),
            body=state.get("email_body"),
        )