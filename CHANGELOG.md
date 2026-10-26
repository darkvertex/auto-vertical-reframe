# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `comedian` preset. It frames the people holding a microphone, one face when there is a single mic and the whole group when several holders fit in the vertical crop.
- Runpod Serverless worker (`handler.py`, `Dockerfile`) for queue endpoints. It takes the same inputs as the Cog predictor, uploads the MP4 when bucket credentials are set, and is meant to run with a 24-hour execution timeout and a 7-day job TTL.
- Cog predictor and `cog.yaml` for [darkvertex/auto-vertical-reframe](https://replicate.com/darkvertex/auto-vertical-reframe). The image bakes YOLO weights and returns the vertical MP4 plus a JSON summary.
- `uv` project metadata, `uv.lock`, and a dev dependency group.
- GitHub Actions: pull requests lint, test, and smoke-test the Cog image; pushes to `main` release it to Replicate.
- `movie` preset: locked camera with one static crop per shot and hard cuts.
- `--start` / `--duration` to reframe only a window of the source.
- `--fast`, `--imgsz`, `--analyze-stride`, `--face-stride`, `--device`, and `--saliency-model off`.
- `--no-two-person-framing` to switch off two-person framing in presets that enable it.
- `--continuous` (and the matching Cog/Runpod `continuous` input) to skip scene detection on a one-shot clip.
- `--no-half` to keep YOLO in FP32. CUDA runs YOLO in FP16 by default.
- `--pose-stride` to reuse pose landmarks for N analyzed frames. The default is every frame.

### Fixed
- Encoder fallback tests a one-frame encode instead of reading `ffmpeg -encoders`, which lists `hevc_nvenc` even when the NVENC driver library is missing. Such jobs used to fail at the end; they now use `libx264` from the start.
- The Runpod image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`, so the NVIDIA runtime mounts the NVENC library.
- macOS uses `h264_videotoolbox` by default again instead of falling back to CPU `libx264`.

### Changed
- Output frames go straight to one ffmpeg encode with the source audio, instead of an intermediate `mp4v` file and a second encode.
- A start time of 0 no longer re-encodes the source just to apply `--duration`.
- Scene detection samples at about 15 fps, and stops at `--duration` instead of scanning the rest of the file.
- Locked mode with stride 1 renders from the frames already decoded for tracking.
- Handcrafted saliency and YOLO mask stats stay on their small maps and scale back to the frame.
- The default encoder is `hevc_nvenc` for Cog/Runpod. The CLI default is `auto`: `h264_videotoolbox` on macOS, `hevc_nvenc` elsewhere. HEVC output is tagged `hvc1`.
- Source audio is stream-copied (`-c:a copy`) instead of re-encoded to AAC.
- YOLO uses CUDA when a GPU is present.
- `process_video` returns the summary dict it already logs, and can reuse a loaded YOLO model.
- The `movie` preset swaps `yolo11n-seg.pt` for `yolo11n.pt` by filename, including absolute paths.
- `sports` preset follows the ball and the nearest player, with a dead zone, motion lead, and two-person framing on by default; `max_zoom` and step limits are lower.
- Follow presets hard-cut to the subject on scene changes instead of easing from frame center.

## [0.1.0] - 2026-04-19

### Added
- Initial release of the Auto Vertical Reframe CLI.
- Scene-aware vertical reframing pipeline built on YOLOv11 segmentation, MediaPipe face/pose, and PySceneDetect.
- Presets: `talking_head`, `sports`, `pets`, `cars`.
- Handcrafted and DeepGaze-MR saliency backends with CPU/CUDA/MPS selection.
- macOS `run_verthor.command` launcher with native dialogs.
- Debug preview export and JSON summary logging.
