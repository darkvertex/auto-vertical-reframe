"""Runpod handler input, shared reframe arguments, and upload payload."""

from __future__ import annotations

import json
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from handler import (
    DOWNLOAD_USER_AGENT,
    EXECUTION_TIMEOUT_MS,
    JOB_TTL_MS,
    acquire_video,
    bucket_object_prefix,
    parse_input,
    publish_video,
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


def test_http_download_avoids_the_blocked_python_user_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            return b""

    def fake_urlopen(request: urllib.request.Request):
        seen["ua"] = request.get_header("User-agent")
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    acquire_video("https://example.com/clip.mp4", tmp_path / "out.mp4")
    assert seen["ua"] == DOWNLOAD_USER_AGENT
    assert "python-urllib" not in seen["ua"].lower()


def test_upload_prefix_defaults_to_the_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BUCKET_UPLOAD_PREFIX", raising=False)
    assert bucket_object_prefix("job-1") == "job-1"


def test_upload_prefix_uses_the_configured_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BUCKET_UPLOAD_PREFIX", "/shows/ep64/")
    assert bucket_object_prefix("job-1") == "shows/ep64/job-1"


def test_upload_prefix_rejects_parent_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUCKET_UPLOAD_PREFIX", "shows/../secret")
    with pytest.raises(ValueError, match="BUCKET_UPLOAD_PREFIX"):
        bucket_object_prefix("job-1")


def test_bad_upload_prefix_fails_before_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BUCKET_UPLOAD_PREFIX", "..")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"input")

    result = run_job(
        {"id": "abc", "input": {"video": str(source)}},
        models={},
        publish=_unused,
    )

    assert result["error"].startswith("BUCKET_UPLOAD_PREFIX")


def test_publish_puts_the_folder_ahead_of_the_job_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BUCKET_UPLOAD_PREFIX", "shows/ep64")
    captured: dict[str, object] = {}

    def upload_file_to_bucket(**kwargs: object) -> str:
        captured.update(kwargs)
        return "https://example.com/out.mp4"

    module = types.ModuleType("runpod.serverless.utils.rp_upload")
    module.upload_file_to_bucket = upload_file_to_bucket
    for name in ("runpod", "runpod.serverless", "runpod.serverless.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils.rp_upload", module)

    video = tmp_path / "vertical.mp4"
    video.write_bytes(b"mp4")
    url = publish_video("job-abcdef12", video)

    assert url == "https://example.com/out.mp4"
    assert captured["prefix"] == "shows/ep64/job-abcdef12"
    assert captured["file_name"] == "vertical-job-abcd.mp4"
    assert captured["file_location"] == str(video)


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
