"""The Cog smoke-test validator accepts an MP4 embedded in prediction JSON."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from validate_cog_output import validate


def _mp4() -> bytes:
    return b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1200


def test_validate_data_url_video(tmp_path: Path) -> None:
    payload = {
        "video": "data:video/mp4;base64," + base64.b64encode(_mp4()).decode(),
        "summary": json.dumps({"frames_processed": 12, "preset": "talking_head"}),
        "debug_video": None,
    }
    path = tmp_path / "output.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    message = validate(path)
    assert "12 frame" in message
    assert "talking_head" in message


def test_validate_file_path_video(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(_mp4())
    payload = {
        "output": {
            "video": str(video),
            "summary": json.dumps({"frames_processed": 3, "preset": "movie"}),
        }
    }
    path = tmp_path / "prediction.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "movie" in validate(path)


def test_validate_rejects_empty_summary(tmp_path: Path) -> None:
    payload = {
        "video": "data:video/mp4;base64," + base64.b64encode(_mp4()).decode(),
        "summary": json.dumps({"frames_processed": 0}),
    }
    path = tmp_path / "output.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="frames_processed"):
        validate(path)
