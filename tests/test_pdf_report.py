from services import pdf_report


def test_formal_pdf_report_contains_business_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_report, "REPORT_DIR", tmp_path)

    meeting = {
        "meeting_id": "meeting_pdf_001",
        "title": "项目周会",
        "host": "主持人",
        "language": "zh-CN",
        "status": "completed",
        "audio_info": {"duration_seconds": 125.5},
        "claims": [
            {
                "claim_type": "观点",
                "speaker_id": "speaker_00",
                "statement": "本周优先完成接口联调",
                "quote": "这周先把接口联调完成",
                "evidence_ids": ["ev_001"],
                "confidence": 0.91,
                "confidence_breakdown": {
                    "evidence_binding": 0.95,
                    "quote_match": 0.9,
                },
                "start_ms": 1000,
                "end_ms": 3000,
            },
            {
                "claim_type": "行动项",
                "speaker_id": "speaker_01",
                "statement": "李四负责整理联调清单",
                "quote": "我负责整理联调清单",
                "evidence_ids": ["ev_002"],
                "confidence": 0.84,
                "start_ms": 4000,
                "end_ms": 6000,
            },
            {
                "claim_type": "风险",
                "speaker_id": "speaker_02",
                "statement": "测试环境可能不稳定",
                "quote": "测试环境最近不太稳定",
                "evidence_ids": ["ev_003"],
                "confidence": 0.78,
                "start_ms": 7000,
                "end_ms": 9000,
            },
        ],
        "speaker_summaries": [
            {
                "speaker_id": "speaker_00",
                "display_name": "speaker_00",
                "key_points": [
                    {
                        "claim_type": "观点",
                        "statement": "本周优先完成接口联调",
                        "evidence_ids": ["ev_001"],
                        "confidence": 0.91,
                    }
                ],
            }
        ],
        "evidence_links": [
            {
                "evidence_id": "ev_001",
                "speaker_id": "speaker_00",
                "quote": "这周先把接口联调完成",
                "start_ms": 1000,
                "end_ms": 3000,
                "asr_confidence": 0.96,
                "speaker_confidence": 0.88,
                "speaker_source": "pyannote",
            },
            {
                "evidence_id": "ev_002",
                "speaker_id": "speaker_01",
                "quote": "我负责整理联调清单",
                "start_ms": 4000,
                "end_ms": 6000,
                "asr_confidence": 0.94,
                "speaker_confidence": 0.9,
                "speaker_source": "pyannote",
            },
            {
                "evidence_id": "ev_003",
                "speaker_id": "speaker_02",
                "quote": "测试环境最近不太稳定",
                "start_ms": 7000,
                "end_ms": 9000,
                "asr_confidence": 0.9,
                "speaker_confidence": 0.82,
                "speaker_source": "pyannote",
            },
        ],
    }

    output_path = pdf_report.generate_meeting_pdf(meeting)

    report = tmp_path / "meeting_pdf_001_trusted_minutes.pdf"
    assert output_path == str(report)
    assert report.exists()
    assert report.stat().st_size > 1000
