"""Check a `cog predict -o` JSON file for a vertical MP4 and a summary."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


def _mp4_bytes(value: object) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("data:"):
        _, _, payload = value.partition(",")
        if not payload:
            return None
        return base64.b64decode(payload)
    path = value[7:] if value.startswith("file://") else value
    file_path = Path(path)
    if file_path.is_file():
        return file_path.read_bytes()
    return None


def _output_object(payload: object) -> dict[str, object]:
    if isinstance(payload, dict) and "video" in payload:
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("output"), dict):
        output = payload["output"]
        if "video" in output:
            return output
    raise AssertionError("prediction JSON has no video field")


def validate(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = _output_object(payload)
    video = _mp4_bytes(output.get("video"))
    if video is None or len(video) < 1000 or b"ftyp" not in video[:64]:
        raise AssertionError("video output is not a non-empty MP4")
    summary_raw = output.get("summary")
    if not isinstance(summary_raw, str):
        raise AssertionError("summary output is not a JSON string")
    summary = json.loads(summary_raw)
    frames = summary.get("frames_processed")
    if not isinstance(frames, int) or frames < 1:
        raise AssertionError(f"expected frames_processed >= 1, got {frames!r}")
    return f"OK: {frames} frame(s), preset={summary.get('preset')}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_json", type=Path)
    args = parser.parse_args(argv)
    print(validate(args.prediction_json))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, json.JSONDecodeError, OSError) as exc:
        print(f"invalid cog output: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
