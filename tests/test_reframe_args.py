"""Preset and namespace behavior that does not run the video pipeline."""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from verthor.auto_reframe import (
    FfmpegPipeWriter,
    MediaPipePoseHelper,
    PrefetchIterator,
    apply_preset,
    build_namespace,
    build_raw_encode_cmd,
    clamp,
    encoder_output_args,
    extract_saliency_region,
    has_ffmpeg_encoder,
    lerp,
    limit_step,
    parse_args,
    process_video,
    resolve_scene_starts,
    resolve_video_encoder,
    scaled_mask_stats,
    scene_frame_skip,
    should_extract_source_clip,
    yolo_half_precision,
    yolo_precision_kwargs,
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
    assert args.mic_action is False
    assert args.duration is None
    assert args.video_encoder == "auto"


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


def test_comedian_preset_follows_microphone_holders() -> None:
    args = build_namespace("in.mp4", "out.mp4", preset="comedian")
    assert args.classes == ["person"]
    assert args.camera_mode == "follow"
    assert args.min_zoom == 1.05
    assert args.max_zoom == 1.50
    assert args.mic_action is True
    assert args.mic_grace_sec == 1.0
    assert args.mic_conf == 0.15
    assert args.skip_pose is False
    assert args.saliency_model == "handcrafted"
    assert args.seg_model == "yolo11n-seg.pt"


def test_comedian_fast_keeps_mic_framing_and_skips_pose() -> None:
    args = build_namespace("in.mp4", "out.mp4", preset="comedian", fast=True)
    assert args.mic_action is True
    assert args.skip_pose is True
    assert args.saliency_model == "off"


def test_fast_shrinks_analysis_and_skips_saliency() -> None:
    args = build_namespace("in.mp4", "out.mp4", fast=True)
    assert args.imgsz <= 512
    assert args.skip_pose is True
    assert args.saliency_model == "off"


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


def test_pose_stride_defaults_to_every_other_frame() -> None:
    args = build_namespace("in.mp4", "out.mp4")
    assert args.pose_stride == 2
    args = apply_preset(parse_args(["in.mp4", "out.mp4", "--pose-stride", "1"]))
    assert args.pose_stride == 1


def test_scene_sampling_targets_about_fifteen_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert scene_frame_skip(12) == 0
    assert scene_frame_skip(30) == 1
    assert scene_frame_skip(60) == 3
    seen: dict[str, object] = {}

    def fake(path, threshold, min_len, *, frame_skip=0, end_frame=None):
        seen["frame_skip"] = frame_skip
        seen["end_frame"] = end_frame
        return [1]

    monkeypatch.setattr("verthor.auto_reframe.detect_scenes", fake)
    assert resolve_scene_starts(
        "clip.mp4",
        3.0,
        15,
        continuous=False,
        frame_skip=scene_frame_skip(30),
        end_frame=90,
    ) == [1]
    assert seen == {"frame_skip": 1, "end_frame": 90}


def test_zero_start_does_not_reencode_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert should_extract_source_clip(0) is False
    assert should_extract_source_clip(1.5) is True
    monkeypatch.setattr(
        "verthor.auto_reframe.cv2.VideoCapture", lambda *_a, **_k: _FakeCapture()
    )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("extract_source_clip should not run")

    monkeypatch.setattr("verthor.auto_reframe.extract_source_clip", boom)
    monkeypatch.setattr(
        "verthor.auto_reframe.resolve_scene_starts",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("scene")),
    )
    args = build_namespace("in.mp4", "out.mp4", start=0, duration=1)
    with pytest.raises(RuntimeError, match="scene"):
        process_video(args, model=_FakeModel())


def test_positive_start_reencodes_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "verthor.auto_reframe.cv2.VideoCapture", lambda *_a, **_k: _FakeCapture()
    )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("clipped")

    monkeypatch.setattr("verthor.auto_reframe.extract_source_clip", boom)
    args = build_namespace("in.mp4", "out.mp4", start=1.5)
    with pytest.raises(RuntimeError, match="clipped"):
        process_video(args, model=_FakeModel())


def test_proxy_saliency_maps_back_to_the_frame() -> None:
    saliency = np.zeros((2, 2), np.float32)
    saliency[0, 0] = 1.0
    region = extract_saliency_region(saliency, (0, 0, 4, 4), frame_size=(4, 4))
    assert region is not None
    assert region[0] < 2
    assert region[1] < 2


def test_mask_stats_scale_to_the_frame() -> None:
    small = np.ones((2, 2), np.float32)
    full = np.ones((4, 4), np.float32)
    area, cx, cy, top = scaled_mask_stats(small, 4, 4)
    area_full, cx_full, cy_full, top_full = scaled_mask_stats(full, 4, 4)
    assert area == 16
    assert top == 0
    assert cx == pytest.approx(cx_full)
    assert cy == pytest.approx(cy_full)
    assert top_full == 0
    assert area_full == 16


def _letterboxed(full: np.ndarray, imgsz: int, stride: int = 32) -> np.ndarray:
    """Resize and pad like ultralytics LetterBox(auto=True) for a stream frame."""
    frame_h, frame_w = full.shape
    gain = min(imgsz / frame_h, imgsz / frame_w)
    new_w, new_h = round(frame_w * gain), round(frame_h * gain)
    pad_w = (imgsz - new_w) % stride
    pad_h = (imgsz - new_h) % stride
    resized = cv2.resize(full, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = round(pad_h / 2 - 0.1)
    left = round(pad_w / 2 - 0.1)
    return cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=0,
    )


@pytest.mark.parametrize(
    ("frame_w", "frame_h", "imgsz"),
    [(1920, 1080, 640), (1280, 720, 512), (1080, 1920, 640), (854, 480, 640)],
)
def test_letterboxed_mask_stats_match_full_res(
    frame_w: int, frame_h: int, imgsz: int
) -> None:
    full = np.zeros((frame_h, frame_w), np.float32)
    # A person-shaped blob off center, so padding errors would shift it.
    x1, x2 = int(frame_w * 0.62), int(frame_w * 0.81)
    y1, y2 = int(frame_h * 0.21), int(frame_h * 0.93)
    full[y1:y2, x1:x2] = 1.0
    small = _letterboxed(full, imgsz)
    assert small.shape != full.shape
    area, cx, cy, top = scaled_mask_stats(small, frame_w, frame_h)
    area_full, cx_full, cy_full, top_full = scaled_mask_stats(full, frame_w, frame_h)
    assert cx == pytest.approx(cx_full, abs=1.0)
    assert cy == pytest.approx(cy_full, abs=1.0)
    assert top == pytest.approx(top_full, abs=max(1.0, frame_h / small.shape[0]))
    assert area == pytest.approx(area_full, rel=0.02)


def test_pose_stride_reuses_the_last_pose() -> None:
    helper = MediaPipePoseHelper(stride=2)
    if helper.detector is None:
        pytest.skip("MediaPipe pose is unavailable")
    fake = SimpleNamespace(calls=0, pose_landmarks=None)

    def process(_rgb: np.ndarray) -> SimpleNamespace:
        fake.calls += 1
        return fake

    helper.detector = SimpleNamespace(process=process)
    frame = np.zeros((20, 20, 3), np.uint8)
    box = (0, 0, 10, 18)
    assert helper.detect_in_person_box(frame, box, track_id=7) is None
    assert helper.detect_in_person_box(frame, box, track_id=7) is None
    assert fake.calls == 1
    helper.reset_cache()
    helper.detect_in_person_box(frame, box, track_id=7)
    assert fake.calls == 2


def test_nvenc_args_and_single_pass_command() -> None:
    assert encoder_output_args("h264_nvenc", 18, "medium") == [
        "-preset",
        "p4",
        "-rc",
        "vbr",
        "-cq",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]
    assert encoder_output_args("hevc_nvenc", 18, "medium")[-2:] == ["-tag:v", "hvc1"]
    assert encoder_output_args("libx264", 18, "medium")[:4] == [
        "-crf",
        "18",
        "-preset",
        "medium",
    ]
    cmd = build_raw_encode_cmd(
        width=16,
        height=32,
        fps=30,
        audio_source="in.mp4",
        output_path="out.mp4",
        encoder="libx264",
        crf=18,
        preset="medium",
        vf=None,
    )
    assert cmd[cmd.index("-s") + 1] == "16x32"
    assert "pipe:0" in cmd
    assert "1:a?" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-c:a") + 1] == "copy"


def test_missing_nvenc_falls_back_to_libx264(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "verthor.auto_reframe.has_ffmpeg_encoder",
        lambda name: name == "libx264",
    )
    assert resolve_video_encoder("h264_nvenc") == "libx264"
    assert resolve_video_encoder("hevc_nvenc") == "libx264"
    monkeypatch.setattr("verthor.auto_reframe.has_ffmpeg_encoder", lambda _name: True)
    assert resolve_video_encoder("h264_nvenc") == "h264_nvenc"


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    """Stub the encoder probe: encoders in ``working`` encode, others fail."""
    working: set[str] = set()
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        encoder = cmd[cmd.index("-c:v") + 1]
        code = 0 if encoder in working else 1
        return subprocess.CompletedProcess(cmd, code, "", "Cannot load libnvidia")

    monkeypatch.setattr("verthor.auto_reframe.shutil.which", lambda _name: "ffmpeg")
    monkeypatch.setattr("verthor.auto_reframe.subprocess.run", fake_run)
    has_ffmpeg_encoder.cache_clear()
    yield SimpleNamespace(working=working, calls=calls)
    has_ffmpeg_encoder.cache_clear()


def test_listed_but_unloadable_nvenc_falls_back(fake_ffmpeg, monkeypatch) -> None:
    monkeypatch.setattr("verthor.auto_reframe.sys.platform", "linux")
    assert resolve_video_encoder("hevc_nvenc") == "libx264"
    assert resolve_video_encoder("auto") == "libx264"
    probe = fake_ffmpeg.calls[0]
    assert probe[probe.index("-f") + 1] == "lavfi"
    assert probe[-2:] == ["null", "-"]


def test_working_nvenc_is_probed_once(fake_ffmpeg, monkeypatch) -> None:
    monkeypatch.setattr("verthor.auto_reframe.sys.platform", "linux")
    fake_ffmpeg.working.add("hevc_nvenc")
    assert resolve_video_encoder("auto") == "hevc_nvenc"
    assert resolve_video_encoder("hevc_nvenc") == "hevc_nvenc"
    assert len(fake_ffmpeg.calls) == 1


def test_auto_encoder_uses_videotoolbox_on_macos(fake_ffmpeg, monkeypatch) -> None:
    monkeypatch.setattr("verthor.auto_reframe.sys.platform", "darwin")
    fake_ffmpeg.working.update({"h264_videotoolbox", "hevc_nvenc"})
    assert resolve_video_encoder("auto") == "h264_videotoolbox"
    fake_ffmpeg.working.clear()
    has_ffmpeg_encoder.cache_clear()
    assert resolve_video_encoder("auto") == "libx264"


def test_explicit_encoder_is_honored(fake_ffmpeg, monkeypatch) -> None:
    monkeypatch.setattr("verthor.auto_reframe.sys.platform", "darwin")
    fake_ffmpeg.working.update({"h264_videotoolbox", "hevc_nvenc"})
    assert resolve_video_encoder("hevc_nvenc") == "hevc_nvenc"
    assert resolve_video_encoder("libx264") == "libx264"


def test_encoder_probe_survives_missing_ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr("verthor.auto_reframe.shutil.which", lambda _name: None)
    has_ffmpeg_encoder.cache_clear()
    try:
        assert has_ffmpeg_encoder("hevc_nvenc") is False
    finally:
        has_ffmpeg_encoder.cache_clear()


def test_pipe_writer_muxes_one_encode(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    source = tmp_path / "src.mp4"
    made = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if made.returncode != 0:
        pytest.skip("ffmpeg has no lavfi source")
    output = tmp_path / "out.mp4"
    writer = FfmpegPipeWriter(
        final_output_path=output,
        width=16,
        height=16,
        fps=10,
        audio_source=source,
        video_encoder="libx264",
        crf=18,
        preset="ultrafast",
        vf=None,
    )
    frame = np.zeros((16, 16, 3), np.uint8)
    frame[:, :, 1] = 255
    writer.write(frame)
    writer.write(frame)
    writer.release()
    data = output.read_bytes()
    assert data[4:8] == b"ftyp"
    assert len(data) > 100


class _FakeCapture:
    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 360
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 30
        return 0

    def release(self) -> None:
        return None


class _FakeModel:
    names: dict[int, str] = {}
    predictor = None


@pytest.mark.parametrize(
    ("device", "cuda", "enabled", "expected"),
    [
        (None, True, True, True),
        ("0", True, True, True),
        ("0,1", True, True, True),
        ("cuda:0", True, True, True),
        ("cpu", True, True, False),
        ("mps", True, True, False),
        (None, False, True, False),
        ("0", False, True, False),
        ("0", True, False, False),
    ],
)
def test_yolo_half_only_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
    device: str | None,
    cuda: bool,
    enabled: bool,
    expected: bool,
) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda))
    monkeypatch.setattr("verthor.auto_reframe.torch", fake_torch)
    assert yolo_half_precision(device, enabled) is expected


def test_no_half_flag() -> None:
    assert build_namespace("in.mp4", "out.mp4").half is True
    args = apply_preset(parse_args(["in.mp4", "out.mp4", "--no-half"]))
    assert args.half is False


def test_fp16_uses_quantize_when_ultralytics_has_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ultralytics import cfg

    assert yolo_precision_kwargs(False) == {}
    monkeypatch.setitem(cfg.DEFAULT_CFG_DICT, "quantize", None)
    assert yolo_precision_kwargs(True) == {"quantize": 16}
    monkeypatch.delitem(cfg.DEFAULT_CFG_DICT, "quantize")
    assert yolo_precision_kwargs(True) == {"half": True}


def test_prefetch_keeps_order() -> None:
    with PrefetchIterator(iter(range(50)), maxsize=2) as items:
        assert list(items) == list(range(50))


def test_prefetch_reraises_producer_errors() -> None:
    def source():
        yield 1
        yield 2
        raise ValueError("decode failed")

    items = PrefetchIterator(source(), maxsize=2)
    assert next(items) == 1
    assert next(items) == 2
    with pytest.raises(ValueError, match="decode failed"):
        next(items)
    with pytest.raises(StopIteration):
        next(items)
    items.close()


def test_prefetch_close_stops_and_closes_source() -> None:
    import threading

    produced: list[int] = []
    closed = threading.Event()

    def source():
        try:
            for i in range(1_000_000):
                produced.append(i)
                yield i
        finally:
            closed.set()

    items = PrefetchIterator(source(), maxsize=2)
    for i in items:
        if i == 3:
            break
    items.close(timeout=5)
    assert not items._thread.is_alive()
    assert closed.is_set()
    # Bounded queue: the producer ran at most a few items ahead.
    assert len(produced) <= 4 + 3
    with pytest.raises(StopIteration):
        next(items)


def test_prefetch_close_while_producer_blocks_on_full_queue() -> None:
    items = PrefetchIterator(iter(range(100)), maxsize=1)
    assert next(items) == 0
    items.close(timeout=5)
    assert not items._thread.is_alive()
