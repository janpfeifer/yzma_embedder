# `yzma_embedder` — MS MARCO GGUF Embedding Benchmark (`llama.cpp` via `yzma` vs GoMLX)

`yzma_embedder` benchmarks [`llama.cpp`](https://github.com/ggml-org/llama.cpp) sentence embedding throughput, CPU efficiency, and memory usage on the [MS MARCO v2.1](https://huggingface.co/datasets/microsoft/ms_marco) passage dataset using [`github.com/hybridgroup/yzma`](https://github.com/hybridgroup/yzma) (`purego` FFI bindings to `libllama`, no CGo required).

It serves as an apples-to-apples `llama.cpp` baseline for [`github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed`](https://github.com/gomlx/go-huggingface/tree/main/examples/msmarco/benchmark_embed):

- **Same dataset loop**: streams MS MARCO Parquet records (`microsoft/ms_marco`, `v2.1`, `train` split; ~10 passages per query).
- **Same tokenizer**: uses `go-huggingface`'s pure-Go embedded GGUF tokenizer (`hfModel.GetTokenizer()`).
- **Same model & post-processing**: [`second-state/All-MiniLM-L6-v2-Embedding-GGUF`](https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF) (`all-MiniLM-L6-v2-Q4_K_M.gguf`, 6-layer BERT encoder, `d_model=384`, mean pooling + L2 normalization).
- **Same `-limit`, `-warmup`, and summary metrics table**.

On first run, `yzma_embedder` automatically downloads prebuilt `llama.cpp` shared libraries into `~/.cache/yzma/lib` (customizable via `-lib` and `-processor`), plus the GGUF model and MS MARCO Parquet shard into `~/.cache/huggingface`.

---

## Execution Modes (`-bucket`)

`yzma_embedder` supports two execution strategies controlled by `-bucket`:

1. **`--bucket=0` — Unpadded Single-Sequence Parallel Mode (Fastest for `llama.cpp` on CPU)**  
   Bypasses bucketing, padding, and multi-sequence packing. Passages stream directly to worker goroutines; each worker tokenizes its passage and runs a single unpadded sequence (`n_seq_max=1`, `n_ctx=512`) via `llama.Encode`. Defaults `-num_embedders` to `runtime.GOMAXPROCS(0)` with `1` CPU thread per `llama.Context`, avoiding both cross-sequence attention waste and intra-context `ggml` threadpool barriers.
2. **`-bucket=8192` — Token-Budget Bucketed Mode (Default)**  
   Uses the same two-bit token-budget bucketer (`bucket.New(tokenizer).ByTwoBitBucketBudget(...)`) as GoMLX's `benchmark_embed`, stripping padding tokens per row and packing sequences into `n_ubatch`-sized `llama.cpp` micro-batches (default `-ubatch=512`).

---

## Benchmark Results

**Environment:**
- **CPU**: AMD Ryzen 9 9950X3D (16 physical cores / 32 threads, Zen 5, AVX-512 VNNI, 3D V-Cache)
- **OS / Go**: Linux `amd64`, Go `1.27.3` (`GOEXPERIMENT=simd` for GoMLX)
- **`llama.cpp` / `yzma`**: `llama.cpp` `0.9.0-dev` (build `b12110`) via `yzma` `v1.32.0` (CPU backend)
- **Workload**: `-limit 1000` queries (**10,000 passages**, **773,300 non-padding tokens**, ~77.3 tokens/passage) after `-warmup` on `second-state/All-MiniLM-L6-v2-Embedding-GGUF` (`all-MiniLM-L6-v2-Q4_K_M.gguf`)

| Engine & Configuration | Total Duration | Passages/s | Non-Pad Tokens/s | CPU Time / Passage | `~Cores` | Max RSS Mem |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GoMLX (`go:quantized_cache`, SIMD, `-bucket=8192`, `1` embedder)** | **`5.1s`** | **`1,969 passages/s`** | **`152.3K tokens/s`** | **`11.4ms`** | `22.5` | `431.3 MiB` |
| **`yzma` / `llama.cpp` (`--bucket=0`, `32` embedders × `1` thread, SMT)** | **`10.1s`** | **`990 passages/s`** | **`76.6K tokens/s`** | `31.8ms` | `31.4` | `381.4 MiB` |
| **`yzma` / `llama.cpp` (`--bucket=0`, `16` embedders × `1` thread, physical cores)** | **`11.1s`** | **`900 passages/s`** | **`69.7K tokens/s`** | **`17.8ms`** | `16.1` | `258.9 MiB` |
| `yzma` / `llama.cpp` (`-bucket=8192`, `4` embedders × `4` threads) | `14.8s` | `676 passages/s` | `52.3K tokens/s` | `20.4ms` | `13.8` | `360.5 MiB` |
| `yzma` / `llama.cpp` (`-bucket=8192`, `1` embedder × `16` threads) | `22.4s` | `447 passages/s` | `34.6K tokens/s` | `25.5ms` | `11.4` | `220.1 MiB` |

### Why `--bucket=0` Doubles `llama.cpp` Throughput, and Why GoMLX Is Still ~2× Faster

1. **3D Batched Attention vs `llama.cpp` 1D Concatenated Attention**:
   - In **GoMLX**, buckets have shape `[batchSize, seqLen, dModel]` (e.g. `[128, 64, 384]`): self-attention stays `[batchSize, heads, seqLen, seqLen]`, so quadratic attention cost is paid **only on `seqLen`** ($\text{batchSize} \times \text{seqLen}^2$), while dense/FFN projections flatten the leading axes to get full matrix reuse at $M = \text{batchSize} \times \text{seqLen} \approx 8192$.
   - In **`llama.cpp` (`ggml`)**, multi-sequence batches have no separate `batchSize` axis — sequences in a `ubatch` of length $U = \sum_i \text{seqLen}_i$ are concatenated end-to-end and masked via a dense `[U, U]` attention matrix ($U^2$ instead of $\sum_i \text{seqLen}_i^2$). Switching `llama.cpp` to `--bucket=0` (1 sequence per call across 16–32 independent single-threaded contexts) eliminates both cross-sequence $O(U^2)$ attention and `ggml` threadpool barrier overhead, jumping from `34.6K` to **`76.6K tokens/s`**.
2. **Large-Batch Pre-Packed AVX-512 VNNI GEMM vs Small-`M` Row Dot-Products**:
   - Even in `--bucket=0` mode, each `llama.cpp` call only sees a single passage ($M \approx 77$ rows) and unpacks `Q4_K` nibbles and 6-bit scales on-the-fly inside `ggml_vec_dot_q4_K_q8_K`.
   - GoMLX (`GOMLX_BACKEND=go:quantized_cache` + `GOEXPERIMENT=simd`) pre-packs constant GGUF weights once into K4-interleaved `uint8` panels (which fit comfortably in L3 cache for MiniLM) and executes 2D register-tiled AVX-512 VNNI (`VPDPBUSD`) GEMMs across $M \approx 8192$ rows with fused in-tile `Bias + GELU` epilogues — cutting CPU time/passage to **`11.4ms`** vs `17.8ms` (`152.3K` vs `76.6K tokens/s`).

---

## How to Run

### 1. Run `yzma_embedder` (`llama.cpp` via `yzma`)

Clone this repository (or run directly with `go run`):

```bash
git clone https://github.com/janpfeifer/yzma_embedder.git
cd yzma_embedder
```

- **Unpadded single-sequence mode across `GOMAXPROCS` embedders (`--bucket=0`, fastest on CPU)**:
  ```bash
  go run . \
    --bucket=0 \
    -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    -limit 1000 -warmup
  ```

- **Unpadded single-sequence mode pinned to physical core count (e.g. `16` embedders)**:
  ```bash
  go run . \
    --bucket=0 -num_embedders=16 \
    -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    -limit 1000 -warmup
  ```

- **Bucketed mode (`-bucket=8192`)**:
  ```bash
  # 1 embedder x default physical-core batch threads:
  go run . \
    -bucket=8192 -num_embedders=1 \
    -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    -limit 1000 -warmup

  # 4 parallel embedders (threads automatically partitioned across embedders):
  go run . \
    -bucket=8192 -num_embedders=4 \
    -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    -limit 1000 -warmup
  ```

### 2. Run the GoMLX Baseline (`go-huggingface`)

Requires Go 1.27+ with `GOEXPERIMENT=simd`:

```bash
GOEXPERIMENT=simd GOMLX_BACKEND=go:quantized_cache \
  go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed@main \
  -bucket=8192 \
  -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
  -limit 1000 -num_embedders=1 -warmup -dynamic=true
```

---

## CLI Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-model` | `second-state/All-MiniLM-L6-v2-Embedding-GGUF` | HuggingFace GGUF repository ID or path to a local `.gguf` file. |
| `-gguf_file` | `""` | Specific `.gguf` filename in multi-file GGUF repos (auto-selects `Q4_K_M`, etc. when empty). |
| `-limit` | `-1` | Number of MS MARCO queries to process (`<= 0` processes all ~808K queries). |
| `-msmarco_split` | `train` | MS MARCO v2.1 split (`train`, `validation`, `test`). |
| `-bucket` | `8192` | Token budget for bucketing; set `--bucket=0` for unpadded single-sequence mode. |
| `-num_embedders` | `0` (auto) | Number of parallel `llama.cpp` contexts (`GOMAXPROCS` when `--bucket=0`, `1` when `-bucket > 0`). |
| `-threads` | `0` (auto) | CPU threads per `llama.cpp` context (auto-divides physical cores across `-num_embedders`, min `1`). |
| `-ubatch` | `512` | Physical micro-batch size (`n_ubatch`) in bucketed mode (`-bucket > 0`). |
| `-flash_attn` | `false` | Enable Flash Attention in `llama.cpp`. |
| `-ngl` | `0` | Number of model layers to offload to GPU (`0` = CPU only). |
| `-processor` | `cpu` | `llama.cpp` shared library variant to download if `-lib` is empty (`cpu`, `cuda`, `vulkan`, `metal`). |
| `-lib` | `~/.cache/yzma/lib` | Directory where `llama.cpp` shared libraries are installed/loaded. |
| `-warmup` | `false` | Run an untimed warmup pass before the benchmark run. |
| `-shapes` | `false` | Print unique shapes encountered during warmup. |
| `-cpuprof` | `""` | Write Go CPU profile to the given file path. |
| `-profile` | `false` | Start `net/http/pprof` server on `localhost:6060`. |
| `-out` | `""` | Compute 4 reference prompts, write embeddings to file, and exit. |

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
