- Model: mlx-community/Llama-3.2-1B-Instruct-4bit
- Hardware: M4 Max, 36GB unified memory
- mlx-lm version: 0.31.3

### v0.1.5 Paged KV Cache Blocks

Method: increasing number of prompts passed in batch and measure the throughput vs latency

```
pool: 128 blocks x 16 = 2048 tokens/layer,  mix=[8, 16, 32, 128]
              max concurrent
       paged              43
  contiguous              16   (each reserves max_len=128)
-> paged holds 2.7x more sequences in the same memory
```

Each sequence's KV pairs now are broken into fixed size blocks allocated within a shared pool, so only ceil(len/16) blocks are used instead of reserving worst-case length. The scattered blocks are gathered back to contiguous for the attention kernel (still using builtin), so this still incurs a compute cost.


### v0.1.4 Static Batching and mask padded tokens

Method: mixed-length workload paged into a shared block pool via the real
cache until the allocator OOMs; contiguous baseline reserves max_len per sequence

```
 batch   agg_tok/s   ms/step
     1       320.4      3.12
     2       636.8      3.14
     4       917.2      4.36
     8      1045.0      7.66
    16      1442.1     11.10

            lengths   real    pad  waste%  prefill_ms  wasted_ms
  [64, 64, 64, 64]    256      0     0.0        61.3        0.0
  [16, 48, 80, 112]   256    192    42.9       108.0       46.3
    [8, 8, 8, 488]    512   1440    73.8       488.9      360.7
```

With static batching, the engine now processes the prefill/decode loop of one or many prompts in one pass. The first table indicates the number of prompts passed in a batch and how the aggregate token throughput scales linearly at small batch sizes but begins to taper as compute becomes the bottleneck.

Prefill computes the full `N * max_len` prompt rectangle, so every row's runtime is scaled to that of the longest prompt. This was implemented as more of an exercise to benchmark the later implementation against, but the second table indicates that batches with more uneven prompt lengths suffer from more wasted runtime. The lopsided batch (row 3) is 74% padding, with ~361ms of the total ~489ms prefill runtime wasted.


### v0.1.2 Custom KV Cache and memory benchmarks

Method: For memory, 2048 tokens sampled every 256, greedy

```
prompt_tok  prefill_ms   ttft_ms   ms/tok  decode_tok/s  prefill_tok/s
        64        16.5      16.6     3.05         328.1         3889.4
       256        59.2      59.4     3.12         320.7         4326.6
      1024       233.5     233.9     3.50         285.8         4385.2
      4096      1075.1    1150.2     4.59         217.9         3809.9

  tokens   active_MB  concat_gap_MB
     256       704.4            8.7
     512       712.8           16.0
     768       721.2           24.2
    1024       729.6           37.0
    1280       738.0           44.6
    1536       746.4           52.5
    1792       754.8           59.6
    2048       763.1           71.7

bytes/token: 32,768 (predicted 32,768)
ceiling: 898,908 tokens in 30.2GB working set
```

The numbers with the custom KV Cache seem in-line with the previous runtime benchmarks, which is expected since the mx arrays are now manually managed, but in a similar way, as previously done by mlx's native `make_prompt_cache`.

The active_MB column climbs linearly, where every 256-token step adds ~8.4 MB. At 2048 tokens the cache is ~67 MB, and the gap is 71.7 MB which points toward the KV cache concatenation temporarily holding onto a second full copy of the cache. Although the steady-state ceiling is noted to be ~898k tokens, the concat function's double buffer means that this would OOM when 2*cache_size fills the working set, so continuing naively would halve the amount of usable, working memory.


### v0.1.1 Naive prefill & decode

Method: 1 warmup, median of 5 reps, 128 tokens generated, greedy (argmax)

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