"""Preset and namespace behavior that does not run the video pipeline."""

from __future__ import annotations

import pytest

from verthor.auto_reframe import (
    build_namespace,
    clamp,
    lerp,
    limit_step,
    process_video,
    resolve_scene_starts,
)


def test_talking_head_preset_defaults() -> None:
    args = build_namespace("in.mp4", "out.mp4", preset="talking_head")
    assert args.preset == "talking_head"
    assert args.classes == ["person"]
    assert args.min_zoom == 1.05
    assert args.max_zoom == 1.85
    assert args.camera_mode == "follow"
    assert args.seg_model == "yolo11n-seg.pt"
    assert args.saliency_model == "handcrafted"
    assert args.duration is None
    assert args.video_encoder == "h264_videotoolbox"


def test_movie_preset_swaps_to_detection_checkpoint() -> None:
    args = build_namespace("in.mp4", "out.mp4", preset="movie")
    assert args.camera_mode == "locked"
    assert args.seg_model == "yolo11n.pt"
    assert args.saliency_model == "off"
    assert args.skip_pose is True


def test_movie_preset_swaps_absolute_segmentation_checkpoint() -> None:
    args = build_namespace(
        "in.mp4",
        "out.mp4",
        preset="movie",
        seg_model="/src/yolo11n-seg.pt",
        video_encoder="libx264",
        device="cpu",
    )
    assert args.seg_model == "/src/yolo11n.pt"
    assert args.video_encoder == "libx264"
    assert args.device == "cpu"


def test_custom_checkpoint_is_not_swapped() -> None:
    args = build_namespace(
        "in.mp4",
        "out.mp4",
        preset="movie",
        seg_model="custom.pt",
    )
    assert args.seg_model == "custom.pt"


def test_fast_shrinks_analysis_and_skips_saliency() -> None:
    args = build_namespace("in.mp4", "out.mp4", fast=True)
    assert args.imgsz <= 512
    assert args.skip_pose is True
    assert args.saliency_model == "off"
    assert args.retina_masks is False


def test_explicit_saliency_survives_fast() -> None:
    args = build_namespace(
        "in.mp4",
        "out.mp4",
        fast=True,
        saliency_model="handcrafted",
    )
    assert args.saliency_model == "handcrafted"
    assert args.skip_saliency is False


def test_continuous_defaults_off() -> None:
    args = build_namespace("in.mp4", "out.mp4")
    assert args.continuous is False


def test_continuous_skips_scene_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    args = build_namespace("in.mp4", "out.mp4", continuous=True)
    assert args.continuous is True

    def boom(*_args: object, **_kwargs: object) -> list[int]:
        raise AssertionError("detect_scenes should not run")

    monkeypatch.setattr("verthor.auto_reframe.detect_scenes", boom)
    assert resolve_scene_starts("clip.mp4", 3.0, 15, continuous=True) == [1]


def test_scene_detection_runs_unless_continuous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "verthor.auto_reframe.detect_scenes",
        lambda *_args, **_kwargs: [1, 40],
    )
    assert resolve_scene_starts("clip.mp4", 3.0, 15, continuous=False) == [1, 40]


def test_camera_math_helpers() -> None:
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
    assert lerp(0, 10, 0.5) == 5
    assert limit_step(0, 10, 2) == 2
    assert limit_step(0, -10, 2) == -2
    assert limit_step(0, 1, 2) == 1


def test_non_positive_duration_is_rejected_before_opening_video() -> None:
    args = build_namespace("in.mp4", "out.mp4", duration=0)
    with pytest.raises(ValueError, match="duration"):
        process_video(args)
