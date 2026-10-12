from collections import deque

import numpy as np

from src.api.streams import MediaStream
from src.engine.request import Media


def test_frames_closer_than_the_sample_interval_are_skipped():
    stream = MediaStream(fps=2, window=5)
    assert stream.admit(0.0) is None
    stream.keep("f0", [])
    assert stream.admit(0.3) == "sampled out"
    assert stream.admit(0.6) is None
    assert (stream.skipped, stream.dropped) == (1, 0)


def test_frames_are_dropped_while_the_previous_one_is_still_encoding():
    stream = MediaStream(fps=100, window=5)
    pending = Media(positions=[0], data=np.zeros((1, 1)))
    assert stream.admit(0.0) is None
    stream.keep("f0", [pending])
    assert stream.admit(1.0) == "encoder busy"
    pending.encoded = 1
    assert stream.admit(2.0) is None
    assert stream.dropped == 1


def test_window_keeps_only_the_latest_frames():
    stream = MediaStream(fps=100, window=2)
    for i in range(3):
        assert stream.admit(float(i)) is None
        stream.keep(f"f{i}", [])
    assert [m["data"] for m in stream.media()] == ["f1", "f2"]
    assert stream.kept == 3


def test_media_items_compare_by_identity():
    a = Media(positions=[0, 1], data=np.zeros((1, 1)))
    b = Media(positions=[0, 1], data=np.zeros((1, 1)))
    items = deque([a, b])
    items.remove(b)
    assert list(items) == [a] and a != b
