from __future__ import annotations


DEFAULT_MEETING_TITLE = "真实会议测试"
DEFAULT_MEETING_HOST = "主持人"
DEFAULT_MEETING_PARTICIPANTS = "张三\n李四\n王五"
DEFAULT_MEETING_LANGUAGE = "zh-CN"


def parse_participants(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = value.replace("，", ",").replace("、", ",").split(",")
        normalized = []
        for item in items:
            normalized.extend(item.splitlines())
        items = normalized

    return [
        str(item).strip()
        for item in items
        if str(item).strip()
    ]
