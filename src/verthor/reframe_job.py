"""Shared reframe job used by the Cog predictor and the Runpod handler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verthor import auto_reframe as pipeline

SEG_WEIGHTS_NAME = "yolo11n-seg.pt"
DET_WEIGHTS_NAME = "yolo11n.pt"
PRESETS = ("talking_head", "sports", "pets", "cars", "movie", "comedian")
SALIENCY_MODELS = ("default", "handcrafted", "off")
MIN_OUTPUT_PX = 16
MAX_OUTPUT_PX = 3840


@dataclass(frozen=True)
class ReframeResult:
    video: Path
    summary: dict[str, Any]
    debug_video: Path | None


def weights_dir() -> Path:
    baked = Path("/src") / SEG_WEIGHTS_NAME
    if baked.is_file():
        return baked.parent
    # src/verthor/reframe_job.py -> repository root, where the seg weights ship.
    return Path(__file__).resolve().parents[2]


def device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def load_models(directory: Path | None = None) -> dict[str, Any]:
    """Load both YOLO checkpoints once per worker."""
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    weights = directory or weights_dir()
    models: dict[str, Any] = {}
    for name in (SEG_WEIGHTS_NAME, DET_WEIGHTS_NAME):
        path = weights / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing weights at {path}. The image build downloads "
                f"{DET_WEIGHTS_NAME}; {SEG_WEIGHTS_NAME} ships in the repo."
            )
        models[name] = YOLO(str(path))
    mic_path = weights / pipeline.MIC_WEIGHTS_NAME
    if mic_path.is_file():
        mic_model = pipeline.open_microphone_model(str(mic_path))
        if mic_model is not None:
            models[pipeline.MIC_WEIGHTS_NAME] = mic_model
    return models


def reframe_video(
    models: dict[str, Any],
    video_path: str | Path,
    *,
    preset: str = "talking_head",
    start: float = 0,
    duration: float = 0,
    fast: bool = False,
    saliency_model: str = "default",
    output_width: int = 1080,
    output_height: int = 1920,
    save_debug_preview: bool = False,
    output_path: str | Path,
    debug_path: str | Path,
) -> ReframeResult:
    """Reframe a local video. ``duration`` of 0 reframes the whole clip."""
    pipeline.setup_logging("INFO")
    output = Path(output_path)
    debug = Path(debug_path)
    weights = weights_dir()
    args = pipeline.build_namespace(
        str(video_path),
        str(output),
        preset=preset,
        start=start,
        duration=None if duration == 0 else duration,
        fast=fast,
        saliency_model=None if saliency_model == "default" else saliency_model,
        output_width=output_width,
        output_height=output_height,
        save_debug_preview=save_debug_preview,
        debug_path=str(debug) if save_debug_preview else "",
        video_encoder="libx264",
        device=device(),
        seg_model=str(weights / SEG_WEIGHTS_NAME),
    )
    model = models.get(Path(args.seg_model).name)
    if model is None:
        raise FileNotFoundError(
            f"No baked checkpoint for {args.seg_model}. "
            f"Available: {', '.join(sorted(models))}"
        )

    summary = pipeline.process_video(
        args,
        model=model,
        mic_model=models.get(pipeline.MIC_WEIGHTS_NAME),
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Reframe finished without an output video")

    debug_output: Path | None = None
    if save_debug_preview:
        if not debug.is_file() or debug.stat().st_size == 0:
            raise RuntimeError("Debug preview was requested but was not written")
        debug_output = debug

    return ReframeResult(video=output, summary=summary, debug_video=debug_output)
