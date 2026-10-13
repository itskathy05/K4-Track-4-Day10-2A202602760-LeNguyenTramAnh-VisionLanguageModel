import json

import pytest

from student.e5 import iou_reward


GT = [{"label": "pothole", "bbox_2d": [400, 400, 600, 600]}]
PERFECT = json.dumps(GT)


def test_e5_reward_perfect_prediction_is_one():
    assert iou_reward(PERFECT, GT) == pytest.approx(1.0)


def test_e5_reward_malformed_json_is_zero():
    assert iou_reward("not json", GT) == 0.0


def test_e5_reward_wrong_class_gets_valid_json_floor():
    wrong = json.dumps([{"label": "crack", "bbox_2d": [400, 400, 600, 600]}])
    assert iou_reward(wrong, GT) == pytest.approx(0.2)


def test_e5_reward_extra_box_is_penalized_by_denominator():
    extra = json.dumps(GT + [{"label": "crack", "bbox_2d": [100, 100, 200, 200]}])
    assert iou_reward(extra, GT) == pytest.approx(0.6)


def test_e5_reward_two_empty_lists_is_one():
    assert iou_reward("[]", []) == 1.0


def test_e5_reward_empty_prediction_with_ground_truth_is_floor():
    assert iou_reward("[]", GT) == pytest.approx(0.2)


def test_e5_reward_huge_box_does_not_match_ground_truth():
    huge = json.dumps([{"label": "pothole", "bbox_2d": [-10000, -10000, 10000, 10000]}])
    assert iou_reward(huge, GT) == pytest.approx(0.2)


def test_e5_reward_accepts_trl_conversational_completion():
    completion = [{"role": "assistant", "content": PERFECT}]
    assert iou_reward(completion, GT) == pytest.approx(1.0)
