from services.agent_result_adapter import build_pdf_ready_result
from services.pdf_report import generate_meeting_pdf


class ReportAgent:
    def run(self, state: dict) -> dict:
        """Build the traceable report payload and render its PDF."""
        pdf_ready_result = build_pdf_ready_result(state)
        pdf_path = generate_meeting_pdf(pdf_ready_result)

        return {
            "pdf_path": pdf_path,
            "claims": pdf_ready_result["claims"],
            "speaker_summaries": pdf_ready_result["speaker_summaries"],
            "evidence_links": pdf_ready_result["evidence_links"],
        }