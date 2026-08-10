

### v0.1.1 - Naive prefill & decode

- Model: mlx-community/Llama-3.2-1B-Instruct-4bit
- Hardware: M4 Max, 36GB unified memory
- mlx-lm version: 0.31.3
- Method: 1 warmup, median of 5 reps, 128 tokens generated, greedy (argmax)

```
prompt_tok  prefill_ms   ms/tok  prefill/tok
        64        16.7     3.00        0.261
       256        59.2     3.04        0.231
      1024       233.2     3.13        0.228
      4096      1029.0     3.55        0.251
```

The prefill latency is dominated by overhead at shorter prompt lengths but is ~10% above a linear growth rate at higher prompt lengths which may be a sign of the quadratic attention term

The decode latency grew ~18% which can be attributed to the attention scan over the growing KV cache

Over a 64x increase in prompt length, prefill latency grew ~62x while decode only grew ~1.2x, so prefill is scaling with prompt length and decode is roughly flat per token