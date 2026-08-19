import mlx.core as mx

class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0
        self.initmemory = mx.get_active_memory()

    def update_and_fetch(self, new_keys, new_values):
        if self.keys is None:
            self.keys = new_keys
            self.values = new_values
        else:
            self.keys = mx.concat([self.keys, new_keys], axis=2)
            self.values = mx.concat([self.values, new_values], axis=2)
        self.offset += new_keys.shape[2]
        return self.keys, self.values

def make_prompt_cache(model):
    num_layers = len(model.layers)
    return [KVCache() for _ in range(num_layers)]