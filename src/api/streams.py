import itertools
from collections import deque

_ids = itertools.count(1)


class MediaStream:
    def __init__(self, fps=1.0, window=30):
        self.id = f"s{next(_ids)}"
        self.fps = fps
        self.frames = deque(maxlen=window)
        self.pending = None
        self.last_kept = None
        self.kept = self.skipped = self.dropped = 0

    def admit(self, now):
        if self.last_kept is not None and now - self.last_kept < 1 / self.fps:
            self.skipped += 1
            return "sampled out"
        if self.pending is not None and not self.pending.ready:
            self.dropped += 1
            return "encoder busy"
        self.last_kept = now
        return None

    def keep(self, frame, key, queued):
        self.frames.append((frame, key))
        self.pending = queued[0] if queued else None
        self.kept += 1

    def media(self):
        return [{"type": "frame", "data": frame} for frame, _ in self.frames]

    def keys(self):
        return [key for _, key in self.frames]
