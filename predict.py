"""Replicate Cog predictor for vertical auto-reframing."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path as LocalPath
from typing import Any

from cog import BaseModel, BasePredictor, Input, Path

_SRC = str(LocalPath(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from verthor.reframe_job import load_models, reframe_video  # noqa: E402


class Output(BaseModel):
    video: Path
    summary: str
    debug_video: Path | None = None


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load both YOLO checkpoints once per worker."""
        self.models: dict[str, Any] = load_models()

    def predict(
        self,
        video: Path = Input(description="Horizontal input video"),
        preset: str = Input(
            description="Framing preset",
            default="talking_head",
            choices=["talking_head", "sports", "pets", "cars", "movie", "comedian"],
        ),
        start: float = Input(
            description="Start time in seconds",
            default=0,
            ge=0,
        ),
        duration: float = Input(
            description="Seconds to reframe. 0 reframes the whole clip.",
            default=0,
            ge=0,
        ),
        fast: bool = Input(
            description="Smaller YOLO input; skip pose, masks, and saliency",
            default=False,
        ),
        saliency_model: str = Input(
            description=(
                "default follows the preset (off for movie and fast). "
                "DeepGaze is not available in this image."
            ),
            default="default",
            choices=["default", "handcrafted", "off"],
        ),
        output_width: int = Input(
            description="Output width in pixels",
            default=1080,
            ge=16,
            le=3840,
        ),
        output_height: int = Input(
            description="Output height in pixels",
            default=1920,
            ge=16,
            le=3840,
        ),
        save_debug_preview: bool = Input(
            description="Also return a preview with the crop and detections drawn",
            default=False,
        ),
        continuous: bool = Input(
            description=(
                "One continuous shot. Skip scene detection and its extra full decode."
            ),
            default=False,
        ),
    ) -> Output:
        """Reframe a horizontal video to vertical 9:16, keeping the subject in frame."""
        suffix = LocalPath(str(video)).suffix or ".mp4"
        input_path = LocalPath("/tmp/verthor-input").with_suffix(suffix)
        output_path = LocalPath("/tmp/verthor-vertical.mp4")
        debug_path = LocalPath("/tmp/verthor-debug.mp4")
        shutil.copy(str(video), input_path)

        result = reframe_video(
            self.models,
            input_path,
            preset=preset,
            start=start,
            duration=duration,
            fast=fast,
            saliency_model=saliency_model,
            output_width=output_width,
            output_height=output_height,
            save_debug_preview=save_debug_preview,
            continuous=continuous,
            output_path=output_path,
            debug_path=debug_path,
        )
        return Output(
            video=Path(result.video),
            summary=json.dumps(result.summary),
            debug_video=Path(result.debug_video) if result.debug_video else None,
        )
