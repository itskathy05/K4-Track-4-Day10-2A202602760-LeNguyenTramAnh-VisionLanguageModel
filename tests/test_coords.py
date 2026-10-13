import pytest

from student.coords import yolo_to_box


def test_tiny_in_bounds_box_survives_rounding():
    box = yolo_to_box("crack", [0.64, 0.786, 0.004, 0.0001])
    x1, y1, x2, y2 = box["bbox_2d"]
    assert x2 > x1 and y2 > y1
    assert 0 <= x1 < x2 <= 1000
    assert 0 <= y1 < y2 <= 1000


@pytest.mark.parametrize("coords", [
    [1.2, 0.5, 0.1, 0.1],  # entirely to the right
    [0.5, -0.2, 0.1, 0.1],  # entirely above
    [0.5, 0.5, 0.0, 0.1],  # zero width
])
def test_non_intersecting_or_zero_area_box_is_rejected(coords):
    with pytest.raises(ValueError):
        yolo_to_box("crack", coords)
