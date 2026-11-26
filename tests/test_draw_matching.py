from types import SimpleNamespace

import pytest

from draw_matching import match_draw


@pytest.mark.parametrize("target,expected", [
    ({"draw_id": "100"}, 0),
    ({"draw_id": "missing"}, None),
    ({"draw_time": "21:40", "draw_index": 1}, 1),
    ({"draw_time": "16:00", "draw_index": 1}, None),
    ({"draw_index": 2}, 1),
    ({"draw_index": 3}, None),
    ({"draw_index": 0}, None),
    ({"draw_index": -1}, None),
    ({"draw_index": True}, None),
    ({"draw_index": "2"}, None),
    ({}, 1),
])
def test_metadata_matching(target, expected):
    draws = [SimpleNamespace(draw_identifier="100", draw_time="14:00"),
             SimpleNamespace(draw_identifier="101", draw_time="21:40")]
    assert match_draw(draws, **target) is (None if expected is None else draws[expected])


def test_time_can_match_second_draw_even_when_first_is_missing():
    draw = SimpleNamespace(draw_identifier="101", draw_time="21:40")
    assert match_draw([draw], draw_time="21:40", draw_index=2) is draw
    assert match_draw([draw], draw_time="14:00", draw_index=1) is None


def test_untimed_history_uses_slot_and_waits_for_missing_slot():
    draws = [SimpleNamespace(draw_time=None), SimpleNamespace(draw_time=None)]
    assert match_draw(draws, draw_time="21:40", draw_index=2) is draws[1]
    assert match_draw(draws[:1], draw_time="21:40", draw_index=2) is None
    assert match_draw(draws[:1], draw_time="21:40") is None
    assert match_draw([], draw_index=1) is None
