"""Runpod handler input, shared reframe arguments, and upload payload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from handler import (
    EXECUTION_TIMEOUT_MS,
    JOB_TTL_MS,
    parse_input,
    run_job,
)
from verthor.reframe_job import ReframeResult, reframe_video


def test_documented_timeouts_are_a_day_and_a_week() -> None:
    assert EXECUTION_TIMEOUT_MS == 24 * 60 * 60 * 1000
    assert JOB_TTL_MS == 7 * 24 * 60 * 60 * 1000


def test_parse_uses_cog_defaults() -> None:
    parsed = parse_input({"video": "clip.mp4", "duration": 0})
    assert parsed["duration"] == 0.0
    assert parsed["preset"] == "talking_head"
    assert parsed["start"] == 0.0
    assert parsed["fast"] is False
    assert parsed["saliency_model"] == "default"
    assert parsed["output_width"] == 1080
    assert parsed["output_height"] == 1920
    assert parsed["save_debug_preview"] is False


def test_missing_video_is_an_error() -> None:
    result = run_job({"id": "job-1", "input": {}}, models={}, publish=_unused)
    assert result == {"error": "video is required"}


def test_input_must_be_an_object() -> None:
    result = run_job({"id": "job-1", "input": []}, models={}, publish=_unused)
    assert result == {"error": "input must be an object"}


def test_unknown_preset_is_an_error() -> None:
    result = run_job(
        {"id": "job-1", "input": {"video": "clip.mp4", "preset": "nope"}},
        models={},
        publish=_unused,
    )
    assert result["error"].startswith("preset must be one of:")


def test_missing_local_video_is_an_error() -> None:
    result = run_job(
        {"id": "job-1", "input": {"video": "/tmp/does-not-exist-verthor.mp4"}},
        models={},
        publish=_unused,
    )
    assert result == {"error": "video not found: /tmp/does-not-exist-verthor.mp4"}


def test_zero_duration_reframes_the_whole_clip(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_process(args, model=None):
        seen["duration"] = args.duration
        seen["encoder"] = args.video_encoder
        seen["device"] = args.device
        Path(args.output).write_bytes(b"mp4")
        return {"preset": args.preset, "frames_processed": 1}

    monkeypatch.setattr("verthor.reframe_job.pipeline.process_video", fake_process)
    monkeypatch.setattr("verthor.reframe_job.device", lambda: "cpu")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"src")

    result = reframe_video(
        {"yolo11n-seg.pt": object()},
        video,
        duration=0,
        output_path=tmp_path / "out.mp4",
        debug_path=tmp_path / "debug.mp4",
    )

    assert seen == {"duration": None, "encoder": "libx264", "device": "cpu"}
    assert result.summary["frames_processed"] == 1
    assert result.debug_video is None


def test_run_job_publishes_video_and_summary(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"input-bytes")
    published: list[tuple[str, str, bytes]] = []

    def fake_reframe(models, video_path, **kwargs):
        assert Path(video_path).read_bytes() == b"input-bytes"
        assert kwargs["duration"] == 0.0
        output = Path(kwargs["output_path"])
        output.write_bytes(b"mp4-bytes")
        return ReframeResult(
            video=output,
            summary={"preset": "talking_head", "frames_processed": 3},
            debug_video=None,
        )

    def publish(job_id: str, path: Path) -> str:
        published.append((job_id, path.name, path.read_bytes()))
        return f"https://example.com/{path.name}"

    result = run_job(
        {"id": "abc", "input": {"video": str(source), "duration": 0}},
        models={},
        publish=publish,
        reframe=fake_reframe,
    )

    summary = json.loads(result["summary"])
    assert result["video"] == "https://example.com/vertical.mp4"
    assert "debug_video" not in result
    assert summary["frames_processed"] == 3
    assert published == [("abc", "vertical.mp4", b"mp4-bytes")]


def test_debug_preview_is_published(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"input")

    def fake_reframe(models, video_path, **kwargs):
        assert kwargs["save_debug_preview"] is True
        output = Path(kwargs["output_path"])
        debug = Path(kwargs["debug_path"])
        output.write_bytes(b"v")
        debug.write_bytes(b"d")
        return ReframeResult(
            video=output,
            summary={"frames_processed": 1},
            debug_video=debug,
        )

    names: list[str] = []
    result = run_job(
        {
            "id": "abc",
            "input": {"video": str(source), "save_debug_preview": True},
        },
        models={},
        publish=lambda job_id, path: names.append(path.name) or f"url:{path.name}",
        reframe=fake_reframe,
    )

    assert result["debug_video"] == "url:debug.mp4"
    assert names == ["vertical.mp4", "debug.mp4"]


def test_pipeline_errors_propagate(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"input")

    def boom(*args, **kwargs):
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        run_job(
            {"id": "abc", "input": {"video": str(source)}},
            models={},
            publish=_unused,
            reframe=boom,
        )


def _unused(job_id: str, path: Path) -> str:
    raise AssertionError(f"publish should not be called for {job_id} {path}")
