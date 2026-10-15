from collections import deque

import numpy as np

from src.api.streams import MediaStream
from src.engine.request import Media


def test_frames_closer_than_the_sample_interval_are_skipped():
    stream = MediaStream(fps=2, window=5)
    assert stream.admit(0.0) is None
    stream.keep("f0", "k0", [])
    assert stream.admit(0.3) == "sampled out"
    assert stream.admit(0.6) is None
    assert (stream.skipped, stream.dropped) == (1, 0)


def test_frames_are_dropped_while_the_previous_one_is_still_encoding():
    stream = MediaStream(fps=100, window=5)
    pending = Media(positions=[0], data=np.zeros((1, 1)))
    assert stream.admit(0.0) is None
    stream.keep("f0", "k0", [pending])
    assert stream.admit(1.0) == "encoder busy"
    pending.encoded = 1
    assert stream.admit(2.0) is None
    assert stream.dropped == 1


def test_window_keeps_only_the_latest_frames():
    stream = MediaStream(fps=100, window=2)
    for i in range(3):
        assert stream.admit(float(i)) is None
        stream.keep(f"f{i}", f"k{i}", [])
    assert [m["data"] for m in stream.media()] == ["f1", "f2"]
    assert stream.keys() == ["k1", "k2"]
    assert stream.kept == 3


def test_media_items_compare_by_identity():
    a = Media(positions=[0, 1], data=np.zeros((1, 1)))
    b = Media(positions=[0, 1], data=np.zeros((1, 1)))
    items = deque([a, b])
    items.remove(b)
    assert list(items) == [a] and a != b


def _busy_stream(fps=2.0):
    stream = MediaStream(fps=fps, window=5)
    pending = Media(positions=[0], data=np.zeros((1, 1)))
    assert stream.admit(0.0) is None
    stream.keep("f0", "k0", [pending])
    return stream, pending


def test_a_drop_halves_the_sampling_rate_down_to_a_floor():
    stream = MediaStream(fps=1.0, window=5, min_fps=0.25)
    for t in (0.0, 2.0, 6.0, 20.0):
        stream.reject(t)
    assert stream.fps == 0.25 and stream.dropped == 4


def test_drops_in_quick_succession_halve_only_once():
    stream, _ = _busy_stream(fps=2.0)
    assert stream.admit(0.6) == "encoder busy"
    stream.reject(0.7)
    assert stream.fps == 1.0 and stream.dropped == 2


def test_rate_recovers_in_steps_after_a_calm_period():
    stream = MediaStream(fps=1.0, window=5, step=0.25, calm=5.0)
    stream.reject(0.0)
    assert stream.fps == 0.5
    stream.admit(4.0)
    assert stream.fps == 0.5
    stream.admit(5.0)
    assert stream.fps == 0.75
    stream.admit(10.0)
    stream.admit(15.0)
    assert stream.fps == 1.0


def test_sampling_follows_the_current_rate():
    stream, pending = _busy_stream(fps=2.0)
    assert stream.admit(0.6) == "encoder busy"
    pending.encoded = 1
    assert stream.admit(0.9) == "sampled out"
    assert stream.admit(1.1) is None
