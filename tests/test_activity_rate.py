# SPDX-License-Identifier: Apache-2.0
"""Rolling-window decode rate shown on the dashboard's generating rows."""

from omlx.admin.activity_rate import RecentRateTracker


def _tracker() -> RecentRateTracker:
    return RecentRateTracker(window_seconds=3.0, min_span_seconds=1.0)


def test_steady_stream_reports_windowed_rate():
    t = _tracker()
    assert t.observe("r", 100.0, 0) is None
    assert t.observe("r", 100.5, 15) is None  # half a second is too short a span
    assert t.observe("r", 101.0, 30) == 30.0
    assert t.observe("r", 103.0, 90) == 30.0  # 90 tokens over the full 3 s window


def test_rate_change_shows_within_one_window_not_lifetime():
    t = _tracker()
    for i in range(21):  # 60 tok/s for ten seconds
        t.observe("r", 100.0 + i * 0.5, i * 30)
    rate = None
    for i in range(1, 7):  # then 20 tok/s for three seconds
        rate = t.observe("r", 110.0 + i * 0.5, 600 + i * 10)
    assert rate == 20.0
    assert rate < 660 / 13.0  # the lifetime average would still read ~51


def test_gap_longer_than_window_restarts_the_window():
    t = _tracker()
    t.observe("r", 100.0, 0)
    t.observe("r", 101.0, 100)
    assert t.observe("r", 105.0, 110) is None  # only samples older than 3 s remain
    assert t.observe("r", 106.0, 120) == 10.0


def test_no_rate_before_two_samples_a_second_apart():
    t = _tracker()
    assert t.observe("r", 100.0, 5) is None
    assert t.observe("r", 100.9, 30) is None
    assert t.observe("r", 101.0, 35) == 30.0


def test_prune_forgets_finished_requests():
    t = _tracker()
    t.observe("a", 100.0, 0)
    t.observe("b", 100.0, 0)
    t.prune({"a"})
    assert t.observe("a", 101.0, 40) == 40.0
    assert t.observe("b", 101.0, 40) is None


def test_token_count_going_backwards_reports_zero():
    t = _tracker()
    t.observe("r", 100.0, 50)
    assert t.observe("r", 101.0, 40) == 0.0
