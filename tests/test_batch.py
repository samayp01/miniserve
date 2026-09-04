import mlx.core as mx
from src.model_runner import _prefill_ids, tokenizer

def test_batched_logits_match_solo():
    a = [tokenizer.encode("the")[-1]] * 12
    b = [tokenizer.encode("cat")[-1]] * 12

    batched, _ = _prefill_ids([a, b])
    solo_a, _  = _prefill_ids([a])
    solo_b, _  = _prefill_ids([b])

    # each batched row's next-token logits match its solo run, up to fp noise
    assert mx.allclose(batched[0, -1], solo_a[0, -1], atol=0.1)
    assert mx.allclose(batched[1, -1], solo_b[0, -1], atol=0.1)

def test_ragged_batch_matches_solo():
    short = [tokenizer.encode("the")[-1]] * 5
    long  = [tokenizer.encode("cat")[-1]] * 20

    batched, _    = _prefill_ids([short, long])
    solo_short, _ = _prefill_ids([short])
    solo_long, _  = _prefill_ids([long])

    assert mx.allclose(batched[0, -1], solo_short[0, -1], atol=0.5)
    assert mx.allclose(batched[1, -1], solo_long[0, -1], atol=0.5)
