from copy import deepcopy

from services import job_store


def test_job_history_records_progress_changes(
    monkeypatch,
):
    memory = {}

    def fake_save_job(job):
        memory[job["job_id"]] = deepcopy(job)

    def fake_get_job(job_id):
        job = memory.get(job_id)
        return deepcopy(job) if job else None

    monkeypatch.setattr(
        job_store,
        "save_job",
        fake_save_job,
    )
    monkeypatch.setattr(
        job_store,
        "get_job",
        fake_get_job,
    )

    job = job_store.create_job(
        "meeting_history_test"
    )

    job_id = job["job_id"]

    job_store.update_job(
        job_id,
        status="queued",
        progress=10,
        stage="等待后台处理",
    )

    job_store.update_job(
        job_id,
        status="asr_processing",
        progress=30,
        stage="语音识别",
    )

    saved_job = memory[job_id]

    assert len(saved_job["history"]) == 3
    assert (
        saved_job["history"][0]["stage"]
        == "任务已创建"
    )
    assert (
        saved_job["history"][1]["progress"]
        == 10
    )
    assert (
        saved_job["history"][2]["stage"]
        == "语音识别"
    )


def test_job_history_deduplicates_same_stage(
    monkeypatch,
):
    memory = {}

    def fake_save_job(job):
        memory[job["job_id"]] = deepcopy(job)

    def fake_get_job(job_id):
        job = memory.get(job_id)
        return deepcopy(job) if job else None

    monkeypatch.setattr(
        job_store,
        "save_job",
        fake_save_job,
    )
    monkeypatch.setattr(
        job_store,
        "get_job",
        fake_get_job,
    )

    job = job_store.create_job(
        "meeting_deduplicate_test"
    )

    job_store.update_job(
        job["job_id"],
        status="queued",
        progress=10,
        stage="等待后台处理",
    )

    job_store.update_job(
        job["job_id"],
        status="queued",
        progress=10,
        stage="等待后台处理",
    )

    saved_job = memory[job["job_id"]]

    assert len(saved_job["history"]) == 2


def test_job_progress_never_moves_backward(
    monkeypatch,
):
    memory = {}

    def fake_save_job(job):
        memory[job["job_id"]] = deepcopy(job)

    def fake_get_job(job_id):
        job = memory.get(job_id)
        return deepcopy(job) if job else None

    monkeypatch.setattr(
        job_store,
        "save_job",
        fake_save_job,
    )
    monkeypatch.setattr(
        job_store,
        "get_job",
        fake_get_job,
    )

    job = job_store.create_job(
        "meeting_monotonic_progress_test"
    )

    job_store.update_job(
        job["job_id"],
        status="agent_processing",
        progress=98,
        stage="ASR 与证据阶段完成",
    )
    job_store.update_job(
        job["job_id"],
        status="agent_processing",
        progress=65,
        stage="多 Agent 抽取会议内容",
    )

    saved_job = memory[job["job_id"]]

    assert saved_job["progress"] == 98
    assert saved_job["history"][-1]["progress"] == 98
    assert (
        saved_job["history"][-1]["stage"]
        == "多 Agent 抽取会议内容"
    )


def test_legacy_job_can_create_history(
    monkeypatch,
):
    memory = {
        "legacy_job": {
            "job_id": "legacy_job",
            "meeting_id": "legacy_meeting",
            "status": "completed",
            "progress": 100,
            "stage": "分析完成",
        }
    }

    def fake_save_job(job):
        memory[job["job_id"]] = deepcopy(job)

    def fake_get_job(job_id):
        job = memory.get(job_id)
        return deepcopy(job) if job else None

    monkeypatch.setattr(
        job_store,
        "save_job",
        fake_save_job,
    )
    monkeypatch.setattr(
        job_store,
        "get_job",
        fake_get_job,
    )

    job_store.update_job(
        "legacy_job",
        status="failed",
        progress=100,
        stage="任务失败",
    )

    saved_job = memory["legacy_job"]

    assert len(saved_job["history"]) == 2
    assert (
        saved_job["history"][0]["stage"]
        == "分析完成"
    )
    assert (
        saved_job["history"][1]["stage"]
        == "任务失败"
    )
