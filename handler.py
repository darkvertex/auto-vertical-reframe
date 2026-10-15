"""Runpod Serverless handler for vertical auto-reframing.

The worker does not impose its own deadline. A clip runs until the pipeline
finishes or the endpoint stops it. Runpod's default execution timeout is 10
minutes, which is too short for a full video. On the endpoint, set the
execution timeout to 24 hours (86400000 ms) and the job TTL to 7 days
(604800000 ms). Send the same values in each request's ``policy`` so a caller
is covered even when the endpoint still has the default. Use ``/run`` and poll
``/status``. ``/runsync`` holds the HTTP connection open and only keeps its
result for a minute.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from verthor.reframe_job import (  # noqa: E402
    MAX_OUTPUT_PX,
    MIN_OUTPUT_PX,
    PRESETS,
    SALIENCY_MODELS,
    ReframeResult,
    load_models,
    reframe_video,
)

EXECUTION_TIMEOUT_MS = 86_400_000
JOB_TTL_MS = 604_800_000

_MODELS: dict[str, Any] | None = None


def get_models() -> dict[str, Any]:
    global _MODELS
    if _MODELS is None:
        _MODELS = load_models()
    return _MODELS


def _as_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _as_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _as_int(value: object, name: str, *, low: int, high: int) -> int:
    number = _as_number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    integer = int(number)
    if integer < low or integer > high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return integer


def parse_input(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a job input and fill the same defaults as the Cog predictor."""
    video = data.get("video")
    if not isinstance(video, str) or not video.strip():
        raise ValueError("video is required")
    video = video.strip()
    if video.startswith(("http://", "https://")):
        if urlparse(video).netloc == "":
            raise ValueError("video URL is missing a host")
    elif "://" in video:
        raise ValueError("video must be an http(s) URL or a local file path")

    preset = data.get("preset", "talking_head")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError(f"preset must be one of: {', '.join(PRESETS)}")

    saliency_model = data.get("saliency_model", "default")
    if not isinstance(saliency_model, str) or saliency_model not in SALIENCY_MODELS:
        raise ValueError(f"saliency_model must be one of: {', '.join(SALIENCY_MODELS)}")

    start = _as_number(data.get("start", 0), "start")
    if start < 0:
        raise ValueError("start must be greater than or equal to 0")
    duration = _as_number(data.get("duration", 0), "duration")
    if duration < 0:
        raise ValueError("duration must be greater than or equal to 0")

    return {
        "video": video,
        "preset": preset,
        "start": start,
        "duration": duration,
        "fast": _as_bool(data.get("fast", False), "fast"),
        "saliency_model": saliency_model,
        "output_width": _as_int(
            data.get("output_width", 1080),
            "output_width",
            low=MIN_OUTPUT_PX,
            high=MAX_OUTPUT_PX,
        ),
        "output_height": _as_int(
            data.get("output_height", 1920),
            "output_height",
            low=MIN_OUTPUT_PX,
            high=MAX_OUTPUT_PX,
        ),
        "save_debug_preview": _as_bool(
            data.get("save_debug_preview", False), "save_debug_preview"
        ),
    }


def acquire_video(source: str, dest: Path) -> None:
    """Copy a local file or download an http(s) URL into ``dest``."""
    if source.startswith(("http://", "https://")):
        urllib.request.urlretrieve(source, dest)
        return
    path = Path(source)
    if not path.is_file():
        raise ValueError(f"video not found: {source}")
    shutil.copy(path, dest)


def _safe_job_id(job_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", job_id)
    return cleaned or "local"


def _suffix_for(source: str) -> str:
    if source.startswith(("http://", "https://")):
        suffix = Path(urlparse(source).path).suffix
    else:
        suffix = Path(source).suffix
    if suffix and re.fullmatch(r"\.[A-Za-z0-9]{1,8}", suffix):
        return suffix
    return ".mp4"


def publish_video(job_id: str, path: Path) -> str:
    """Upload an MP4. Without bucket credentials this returns a local path."""
    from runpod.serverless.utils import rp_upload

    file_name = f"{path.stem}-{_safe_job_id(job_id)[:8]}{path.suffix or '.mp4'}"
    return rp_upload.upload_file_to_bucket(
        file_name=file_name,
        file_location=str(path),
        prefix=job_id,
        extra_args={"ContentType": "video/mp4"},
    )


def run_job(
    job: dict[str, Any],
    *,
    models: dict[str, Any],
    publish: Callable[[str, Path], str],
    reframe: Callable[..., ReframeResult] = reframe_video,
) -> dict[str, Any]:
    raw = job.get("input")
    if not isinstance(raw, dict):
        return {"error": "input must be an object"}
    try:
        request = parse_input(raw)
    except ValueError as exc:
        return {"error": str(exc)}

    job_id = str(job.get("id") or "local")
    work = Path("/tmp") / f"verthor-{_safe_job_id(job_id)}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        input_path = work / f"input{_suffix_for(request['video'])}"
        try:
            acquire_video(request["video"], input_path)
        except ValueError as exc:
            return {"error": str(exc)}

        output_path = work / "vertical.mp4"
        debug_path = work / "debug.mp4"
        result = reframe(
            models,
            input_path,
            preset=request["preset"],
            start=request["start"],
            duration=request["duration"],
            fast=request["fast"],
            saliency_model=request["saliency_model"],
            output_width=request["output_width"],
            output_height=request["output_height"],
            save_debug_preview=request["save_debug_preview"],
            output_path=output_path,
            debug_path=debug_path,
        )
        payload: dict[str, Any] = {
            "video": publish(job_id, result.video),
            "summary": json.dumps(result.summary),
        }
        if result.debug_video is not None:
            payload["debug_video"] = publish(job_id, result.debug_video)
        return payload
    finally:
        shutil.rmtree(work, ignore_errors=True)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    return run_job(job, models=get_models(), publish=publish_video)


def main() -> None:
    import runpod

    get_models()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
