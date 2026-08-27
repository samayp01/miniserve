import mlx.core as mx
from engine import _prefill_ids, tokenizer

def test_batched_logits_match_solo():
    a = [tokenizer.encode("the")[-1]] * 12
    b = [tokenizer.encode("cat")[-1]] * 12

    batched, _ = _prefill_ids([a, b])
    solo_a, _  = _prefill_ids([a])
    solo_b, _  = _prefill_ids([b])

    # each batched row's next-token logits match its solo run, up to fp noise
    assert mx.allclose(batched[0, -1], solo_a[0, -1], atol=0.1)
    assert mx.allclose(batched[1, -1], solo_b[0, -1], atol=0.1)
