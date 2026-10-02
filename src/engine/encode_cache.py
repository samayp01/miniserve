from collections import OrderedDict


class EncodeCache:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key):
        embeds = self.entries.get(key)
        if embeds is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return embeds

    def put(self, key, embeds):
        if key in self.entries or embeds.nbytes > self.max_bytes:
            return
        self.entries[key] = embeds
        self.bytes += embeds.nbytes
        while self.bytes > self.max_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.bytes -= evicted.nbytes
