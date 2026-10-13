"""Replicate Cog predictor for vertical auto-reframing."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path as LocalPath
from typing import Any

from cog import BaseModel, BasePredictor, Input, Path

SEG_WEIGHTS_NAME = "yolo11n-seg.pt"
DET_WEIGHTS_NAME = "yolo11n.pt"


class Output(BaseModel):
    video: Path
    summary: str
    debug_video: Path | None = None


def _weights_dir() -> LocalPath:
    baked = LocalPath("/src") / SEG_WEIGHTS_NAME
    if baked.is_file():
        return baked.parent
    return LocalPath(__file__).resolve().parent


def _pipeline():
    src = str(LocalPath(__file__).resolve().parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import verthor.auto_reframe as pipeline

    return pipeline


def _device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load both YOLO checkpoints once per worker."""
        from ultralytics import YOLO, settings

        settings.update({"sync": False})
        weights = _weights_dir()
        self.models: dict[str, Any] = {}
        for name in (SEG_WEIGHTS_NAME, DET_WEIGHTS_NAME):
            path = weights / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing weights at {path}. The image build downloads "
                    f"{DET_WEIGHTS_NAME}; {SEG_WEIGHTS_NAME} ships in the repo."
                )
            self.models[name] = YOLO(str(path))

    def predict(
        self,
        video: Path = Input(description="Horizontal input video"),
        preset: str = Input(
            description="Framing preset",
            default="talking_head",
            choices=["talking_head", "sports", "pets", "cars", "movie"],
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
    ) -> Output:
        """Reframe a horizontal video to vertical 9:16, keeping the subject in frame."""
        pipeline = _pipeline()
        pipeline.setup_logging("INFO")

        suffix = LocalPath(str(video)).suffix or ".mp4"
        input_path = LocalPath("/tmp/verthor-input").with_suffix(suffix)
        output_path = LocalPath("/tmp/verthor-vertical.mp4")
        debug_path = LocalPath("/tmp/verthor-debug.mp4")
        shutil.copy(str(video), input_path)

        weights = _weights_dir()
        args = pipeline.build_namespace(
            str(input_path),
            str(output_path),
            preset=preset,
            start=start,
            duration=None if duration == 0 else duration,
            fast=fast,
            saliency_model=None if saliency_model == "default" else saliency_model,
            output_width=output_width,
            output_height=output_height,
            save_debug_preview=save_debug_preview,
            debug_path=str(debug_path) if save_debug_preview else "",
            video_encoder="libx264",
            device=_device(),
            seg_model=str(weights / SEG_WEIGHTS_NAME),
        )
        model = self.models.get(LocalPath(args.seg_model).name)
        if model is None:
            raise FileNotFoundError(
                f"No baked checkpoint for {args.seg_model}. "
                f"Available: {', '.join(sorted(self.models))}"
            )

        summary = pipeline.process_video(args, model=model)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Reframe finished without an output video")

        debug_output: Path | None = None
        if save_debug_preview:
            if not debug_path.is_file() or debug_path.stat().st_size == 0:
                raise RuntimeError("Debug preview was requested but was not written")
            debug_output = Path(debug_path)

        return Output(
            video=Path(output_path),
            summary=json.dumps(summary),
            debug_video=debug_output,
        )
