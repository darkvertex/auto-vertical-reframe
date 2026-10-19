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

### Changed
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
