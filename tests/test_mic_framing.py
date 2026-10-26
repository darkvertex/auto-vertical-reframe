"""Microphone ownership and group crops, without running the video pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from verthor.auto_reframe import (
    CameraState,
    Candidate,
    MicDetection,
    assign_mics,
    choose_holder_group,
    claim_mic,
    reset_for_new_scene,
    resolve_mic_holders,
)

PROMPT = (
    Path(__file__).resolve().parents[1] / "src" / "verthor" / "microphone_prompt.npz"
)


def _person(**overrides: object) -> Candidate:
    fields: dict[str, object] = {
        "cls_id": 0,
        "cls_name": "person",
        "track_id": 1,
        "conf": 0.9,
        "x1": 100.0,
        "y1": 50.0,
        "x2": 220.0,
        "y2": 450.0,
        "cx": 160.0,
        "cy": 250.0,
        "width": 120.0,
        "height": 400.0,
        "area": 48000.0,
        "mask_area": 48000.0,
        "mask_cx": 160.0,
        "mask_cy": 250.0,
        "mask_top_y": 50.0,
        "framing_cx": 160.0,
        "framing_cy": 120.0,
        "face_box": None,
    }
    fields.update(overrides)
    return Candidate(**fields)  # type: ignore[arg-type]


def _state() -> CameraState:
    return CameraState(
        crop_center_x=0.0,
        crop_center_y=0.0,
        zoom=1.0,
        target_center_x=0.0,
        target_center_y=0.0,
        target_zoom=1.0,
    )


def _at(track_id: int, left: float, width: float = 40.0) -> Candidate:
    return _person(
        track_id=track_id,
        x1=left,
        x2=left + width,
        y1=0.0,
        y2=100.0,
        width=width,
        height=100.0,
        framing_cx=left + width / 2.0,
        framing_cy=30.0,
        mask_top_y=0.0,
        cx=left + width / 2.0,
        cy=50.0,
    )


def test_only_the_person_with_a_hand_on_the_mic_is_a_holder() -> None:
    mic = MicDetection(140.0, 190.0, 160.0, 220.0, 0.8)
    holder = _person(track_id=4, framing_cx=400.0, left_wrist=(150.0, 205.0))
    bystander = _at(9, 800.0)
    assert claim_mic(holder, mic) == ("wrist", 0.0)
    assert claim_mic(bystander, mic) is None
    assert [person.track_id for person in assign_mics([bystander, holder], [mic])] == [
        4
    ]


def test_stand_mic_in_front_of_the_chest_counts() -> None:
    person = _person(
        mouth_xy=(160.0, 120.0),
        shoulder_cx=160.0,
        shoulder_y=180.0,
        shoulder_span=80.0,
        hip_y=360.0,
    )
    mic = MicDetection(154.0, 200.0, 166.0, 230.0, 0.7)
    kind, _distance = claim_mic(person, mic) or ("", 0.0)
    assert kind == "stand"


def test_mic_on_a_stool_belongs_to_nobody() -> None:
    person = _person(
        left_wrist=(180.0, 200.0),
        mouth_xy=(160.0, 120.0),
        shoulder_cx=160.0,
        shoulder_y=180.0,
        shoulder_span=80.0,
        hip_y=360.0,
    )
    stool = MicDetection(20.0, 390.0, 40.0, 430.0, 0.8)
    assert claim_mic(person, stool) is None
    assert assign_mics([person], [stool]) == []


def test_wrist_claim_beats_a_nearer_stand_claim() -> None:
    mic = MicDetection(140.0, 190.0, 160.0, 220.0, 0.9)
    with_wrist = _person(track_id=1, framing_cx=300.0, left_wrist=(150.0, 205.0))
    at_stand = _person(
        track_id=2,
        framing_cx=150.0,
        mouth_xy=(150.0, 200.0),
        shoulder_cx=150.0,
        shoulder_y=220.0,
        shoulder_span=200.0,
        hip_y=400.0,
    )
    assert [
        person.track_id for person in assign_mics([at_stand, with_wrist], [mic])
    ] == [1]


def test_each_microphone_adds_its_holder() -> None:
    left = _person(track_id=2, framing_cx=80.0, left_wrist=(80.0, 100.0))
    right = _person(track_id=5, framing_cx=240.0, right_wrist=(240.0, 100.0))
    mics = [
        MicDetection(230.0, 90.0, 250.0, 110.0, 0.6),
        MicDetection(70.0, 90.0, 90.0, 110.0, 0.6),
    ]
    assert [person.track_id for person in assign_mics([right, left], mics)] == [2, 5]


def test_upper_body_box_claims_a_mic_when_pose_is_missing() -> None:
    person = _person()
    chest = MicDetection(150.0, 140.0, 170.0, 170.0, 0.5)
    feet = MicDetection(150.0, 390.0, 170.0, 420.0, 0.5)
    assert claim_mic(person, chest)[0] == "body"  # type: ignore[index]
    assert claim_mic(person, feet) is None


def test_wide_lineup_keeps_the_cluster_nearest_the_crop() -> None:
    near_left = _at(1, 0.0)
    near_right = _at(2, 50.0)
    far = _at(3, 2000.0)
    group = choose_holder_group([far, near_left, near_right], 200.0, 70.0)
    assert [person.track_id for person in group] == [1, 2]

    spread = choose_holder_group(
        [_at(1, 0.0), _at(2, 400.0), _at(3, 800.0)], 150.0, 420.0
    )
    assert [person.track_id for person in spread] == [2]


def test_grace_keeps_the_last_holder_for_one_second() -> None:
    state = _state()
    person = _person(track_id=7, left_wrist=(150.0, 205.0))
    mic = MicDetection(140.0, 190.0, 160.0, 220.0, 0.9)
    held = resolve_mic_holders([person], [mic], state, 30.0, 1.0)
    assert [person.track_id for person in held] == [7]
    assert state.mic_grace_frames == 0

    for _ in range(30):
        held = resolve_mic_holders([person], [], state, 30.0, 1.0)
        assert [person.track_id for person in held] == [7]
    assert resolve_mic_holders([person], [], state, 30.0, 1.0) == []
    assert state.mic_holder_keys == []


def test_scene_change_forgets_the_previous_holder() -> None:
    state = _state()
    state.mic_holder_keys = [(7, 0)]
    state.mic_grace_frames = 12
    reset_for_new_scene(state, 1920, 1080, 1.05, hard_cut=True)
    assert state.mic_holder_keys == []
    assert state.mic_grace_frames == 0


def test_microphone_prompt_is_bound_to_yoloe11s() -> None:
    with np.load(PROMPT, allow_pickle=False) as data:
        assert list(data.files) == ["embeddings", "names", "model"]
        assert str(data["names"][0]) == "microphone"
        assert str(data["model"].item()) == "yoloe-11s"
        assert data["embeddings"].shape[1] == 1
        assert data["embeddings"].dtype == np.float32
