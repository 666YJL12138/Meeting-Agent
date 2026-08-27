from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measure_stage(state: dict, stage_name: str):
    """
    记录单个处理阶段的执行时间。
    无论阶段成功还是失败，都会写入 timings。
    """
    started_at = perf_counter()

    try:
        yield
    finally:
        elapsed_seconds = perf_counter() - started_at
        timings = state.setdefault("timings", {})
        timings[stage_name] = round(elapsed_seconds, 3)


def record_timing(
    state: dict,
    stage_name: str,
    elapsed_seconds: float,
) -> None:
    """Record a timing supplied by a worker thread."""
    timings = state.setdefault("timings", {})
    timings[stage_name] = round(elapsed_seconds, 3)
