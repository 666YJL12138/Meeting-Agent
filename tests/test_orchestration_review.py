from orchestration.graph import (
    build_meeting_graph,
    human_review,
    review_gate,
    route_after_review_gate,
)


def test_review_gate_routes_overlap_to_human_review():
    state = {
        "meeting_id": "meeting_review_001",
        "evidence": [{
            "evidence_id": "ev_001",
            "speaker_id": "speaker_00",
            "overlap": True,
        }],
        "errors": [],
        "review_queue": [],
        "review_reasons": [],
        "orchestration_trace": [],
    }

    result = review_gate(state)

    assert result["review_required"] is True
    assert "存在多人重叠语音" in result["review_reasons"]
    assert route_after_review_gate(result) == "human_review"

    reviewed = human_review(result)
    assert reviewed["review_queue"][0]["source"] == "orchestrator"
    assert reviewed["orchestration_trace"][-1]["status"] == "queued"


def test_meeting_graph_contains_review_route():
    graph = build_meeting_graph().get_graph()
    assert "review_gate" in graph.nodes
    assert "human_review" in graph.nodes
