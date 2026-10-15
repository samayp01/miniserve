import itertools
from collections import deque

_ids = itertools.count(1)


class MediaStream:
    def __init__(self, fps=1.0, window=30, min_fps=0.25, step=0.25, calm=5.0):
        self.id = f"s{next(_ids)}"
        self.target_fps = fps
        self.fps = fps
        self.min_fps, self.step, self.calm = min_fps, step, calm
        self.last_change = None
        self.frames = deque(maxlen=window)
        self.pending = None
        self.last_kept = None
        self.kept = self.skipped = self.dropped = 0

    def _back_off(self, now):
        self.dropped += 1
        if self.last_change is None or now - self.last_change >= 1 / self.fps:
            self.fps = max(self.min_fps, self.fps / 2)
            self.last_change = now

    def _recover(self, now):
        if self.fps < self.target_fps and (self.last_change is None or now - self.last_change >= self.calm):
            self.fps = min(self.target_fps, self.fps + self.step)
            self.last_change = now

    def admit(self, now):
        self._recover(now)
        if self.last_kept is not None and now - self.last_kept < 1 / self.fps:
            self.skipped += 1
            return "sampled out"
        if self.pending is not None and not self.pending.ready:
            self._back_off(now)
            return "encoder busy"
        self.last_kept = now
        return None

    def reject(self, now):
        self._back_off(now)

    def keep(self, frame, key, queued):
        self.frames.append((frame, key))
        self.pending = queued[0] if queued else None
        self.kept += 1

    def media(self):
        return [{"type": "frame", "data": frame} for frame, _ in self.frames]

    def keys(self):
        return [key for _, key in self.frames]
