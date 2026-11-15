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

1. **`-bucket=0` — Unpadded Single-Sequence Parallel Mode (Default, Fastest on CPU)**  
   Uses `bucket.New(tokenizer).Unpadded()` (same streaming parallel tokenizer as GoMLX's `benchmark_embed` with `-bucket=0`), bypassing bucketing, padding, and multi-sequence packing. Each worker receives a single unpadded tokenized passage and runs it (`n_seq_max=1`, `n_ctx=512`) via `llama.Encode`. Defaults `-num_embedders` to `runtime.GOMAXPROCS(0)` with `1` CPU thread per `llama.Context`, avoiding both cross-sequence attention waste and intra-context `ggml` threadpool barriers.
2. **`-bucket=8192` — Token-Budget Bucketed Mode**  
   Uses the same two-bit token-budget bucketer (`bucket.New(tokenizer).ByTwoBitBucketBudget(...)`) as GoMLX's `benchmark_embed`, stripping padding tokens per row and packing sequences into `n_ubatch`-sized `llama.cpp` micro-batches (default `-ubatch=512`).

---

## Benchmark Results

**Shared Workload:**
- **Dataset & Model**: `-limit 1000` queries (**10,000 passages**, **773,300 non-padding tokens**, ~77.3 tokens/passage) after `-warmup` on `second-state/All-MiniLM-L6-v2-Embedding-GGUF` (`all-MiniLM-L6-v2-Q4_K_M.gguf`)
- **`llama.cpp` / `yzma`**: `llama.cpp` `0.9.0-dev` (build `b12110`) via `yzma` `v1.32.0` (CPU backend, `-ngl=0`)

### 1. AMD Ryzen 9 9950X3D (Linux `amd64`, 16 Cores / 32 Threads)

- **CPU**: AMD Ryzen 9 9950X3D (16 physical cores / 32 threads, Zen 5, AVX-512 VNNI, 3D V-Cache)
- **OS / Go**: Linux `amd64`, Go `1.27.3` (`GOEXPERIMENT=simd` for GoMLX)
- **Configuration**: `-bucket=0 -num_embedders=32 -limit 1000 -warmup` across all 13 GGUF quantizations of `second-state/All-MiniLM-L6-v2-Embedding-GGUF` (GoMLX with `GOMLX_BACKEND=go:quantized_cache -dynamic=true`)

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Quantization</th>
      <th colspan="3" align="center"><code>yzma</code> + <code>llama.cpp</code></th>
      <th colspan="3" align="center">GoMLX + Go backend (<code>go:quantized_cache</code>)</th>
    </tr>
    <tr>
      <th align="right">Passages/s</th>
      <th align="right">Tokens/s</th>
      <th align="right">Max RSS</th>
      <th align="right">Passages/s</th>
      <th align="right">Tokens/s</th>
      <th align="right">Max RSS</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>Q2_K</code></td><td align="right">1.5K</td><td align="right">113.1K</td><td align="right">435.6 MiB</td><td align="right">4.3K</td><td align="right">329.7K</td><td align="right">478.5 MiB</td></tr>
    <tr><td><code>Q3_K_S</code></td><td align="right">1.5K</td><td align="right">112.4K</td><td align="right">386.4 MiB</td><td align="right">4.3K</td><td align="right">329.1K</td><td align="right">482.0 MiB</td></tr>
    <tr><td><code>Q3_K_M</code></td><td align="right">1.3K</td><td align="right">97.1K</td><td align="right">370.4 MiB</td><td align="right">4.5K</td><td align="right">349.9K</td><td align="right">476.1 MiB</td></tr>
    <tr><td><code>Q3_K_L</code></td><td align="right">1K</td><td align="right">80.9K</td><td align="right">398.3 MiB</td><td align="right">4.5K</td><td align="right">349.4K</td><td align="right">486.8 MiB</td></tr>
    <tr><td><code>Q4_0</code></td><td align="right">1.5K</td><td align="right">117.6K</td><td align="right">460.5 MiB</td><td align="right">4.5K</td><td align="right">346.2K</td><td align="right">491.4 MiB</td></tr>
    <tr><td><code>Q4_K_S</code></td><td align="right">928</td><td align="right">71.8K</td><td align="right">403.5 MiB</td><td align="right">4.5K</td><td align="right">349.6K</td><td align="right">489.4 MiB</td></tr>
    <tr><td><code>Q4_K_M</code></td><td align="right">987</td><td align="right">76.4K</td><td align="right">415.8 MiB</td><td align="right">4.4K</td><td align="right">342K</td><td align="right">494.7 MiB</td></tr>
    <tr><td><code>Q5_0</code></td><td align="right">860</td><td align="right">66.6K</td><td align="right">377.9 MiB</td><td align="right">4.5K</td><td align="right">345.7K</td><td align="right">469.1 MiB</td></tr>
    <tr><td><code>Q5_K_S</code></td><td align="right">554</td><td align="right">42.9K</td><td align="right">395.9 MiB</td><td align="right">4.5K</td><td align="right">348.7K</td><td align="right">485.4 MiB</td></tr>
    <tr><td><code>Q5_K_M</code></td><td align="right">564</td><td align="right">43.7K</td><td align="right">412.2 MiB</td><td align="right">4.4K</td><td align="right">343.4K</td><td align="right">491.9 MiB</td></tr>
    <tr><td><code>Q6_K</code></td><td align="right">1.1K</td><td align="right">85.9K</td><td align="right">399.7 MiB</td><td align="right">4.2K</td><td align="right">328.2K</td><td align="right">499.0 MiB</td></tr>
    <tr><td><code>Q8_0</code></td><td align="right">1K</td><td align="right">79.4K</td><td align="right">387.7 MiB</td><td align="right">4.5K</td><td align="right">345.6K</td><td align="right">494.1 MiB</td></tr>
    <tr><td><code>F16</code></td><td align="right">736</td><td align="right">56.9K</td><td align="right">441.1 MiB</td><td align="right">1.9K</td><td align="right">145.4K</td><td align="right">648.9 MiB</td></tr>
  </tbody>
</table>

### 2. Apple Mac mini M6 (macOS `arm64`, 12 Cores)

- **CPU**: Apple M6 (12 cores total: 2 Super, 4 Performance, 6 Efficiency; ARMv9.2-A with NEON `USDOT` / `I8MM` and SME2)
- **OS / Go**: macOS `darwin/arm64`, Go `1.27.3` (`GOEXPERIMENT=simd` for GoMLX; activates NEON + SME2 kernels)
- **Configuration**: `-bucket=0 -num_embedders=12 -limit 1000 -warmup` across all 13 GGUF quantizations of `second-state/All-MiniLM-L6-v2-Embedding-GGUF` (GoMLX with `GOMLX_BACKEND=go:quantized_cache -dynamic=true`)

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Quantization</th>
      <th colspan="3" align="center"><code>yzma</code> + <code>llama.cpp</code></th>
      <th colspan="3" align="center">GoMLX + Go backend (<code>go:quantized_cache</code>)</th>
    </tr>
    <tr>
      <th align="right">Passages/s</th>
      <th align="right">Tokens/s</th>
      <th align="right">Max RSS</th>
      <th align="right">Passages/s</th>
      <th align="right">Tokens/s</th>
      <th align="right">Max RSS</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>Q2_K</code></td><td align="right">867</td><td align="right">67.1K</td><td align="right">388.6 MiB</td><td align="right">1K</td><td align="right">79.8K</td><td align="right">387.9 MiB</td></tr>
    <tr><td><code>Q3_K_S</code></td><td align="right">859</td><td align="right">66.5K</td><td align="right">384.3 MiB</td><td align="right">1K</td><td align="right">79.3K</td><td align="right">397.1 MiB</td></tr>
    <tr><td><code>Q3_K_M</code></td><td align="right">759</td><td align="right">58.8K</td><td align="right">427.0 MiB</td><td align="right">1.1K</td><td align="right">88K</td><td align="right">390.3 MiB</td></tr>
    <tr><td><code>Q3_K_L</code></td><td align="right">701</td><td align="right">54.3K</td><td align="right">361.6 MiB</td><td align="right">1.1K</td><td align="right">87.5K</td><td align="right">387.9 MiB</td></tr>
    <tr><td><code>Q4_0</code></td><td align="right">951</td><td align="right">73.6K</td><td align="right">372.6 MiB</td><td align="right">1.1K</td><td align="right">87.7K</td><td align="right">397.5 MiB</td></tr>
    <tr><td><code>Q4_K_S</code></td><td align="right">744</td><td align="right">57.6K</td><td align="right">391.3 MiB</td><td align="right">1.1K</td><td align="right">87.4K</td><td align="right">382.8 MiB</td></tr>
    <tr><td><code>Q4_K_M</code></td><td align="right">737</td><td align="right">57K</td><td align="right">389.9 MiB</td><td align="right">1.1K</td><td align="right">82K</td><td align="right">396.9 MiB</td></tr>
    <tr><td><code>Q5_0</code></td><td align="right">718</td><td align="right">55.6K</td><td align="right">410.9 MiB</td><td align="right">1.1K</td><td align="right">84.6K</td><td align="right">397.0 MiB</td></tr>
    <tr><td><code>Q5_K_S</code></td><td align="right">660</td><td align="right">51.1K</td><td align="right">366.7 MiB</td><td align="right">1.1K</td><td align="right">83.6K</td><td align="right">392.1 MiB</td></tr>
    <tr><td><code>Q5_K_M</code></td><td align="right">661</td><td align="right">51.2K</td><td align="right">369.8 MiB</td><td align="right">1K</td><td align="right">79.6K</td><td align="right">398.9 MiB</td></tr>
    <tr><td><code>Q6_K</code></td><td align="right">683</td><td align="right">52.9K</td><td align="right">372.8 MiB</td><td align="right">956</td><td align="right">74K</td><td align="right">396.8 MiB</td></tr>
    <tr><td><code>Q8_0</code></td><td align="right">836</td><td align="right">64.7K</td><td align="right">347.7 MiB</td><td align="right">1K</td><td align="right">81.1K</td><td align="right">390.4 MiB</td></tr>
    <tr><td><code>F16</code></td><td align="right">722</td><td align="right">55.8K</td><td align="right">406.1 MiB</td><td align="right">661</td><td align="right">51.2K</td><td align="right">569.6 MiB</td></tr>
  </tbody>
</table>

> **Note on Apple M6 3-tier core topology (`Super` / `Performance` / `Efficiency`) in `llama.cpp`:** On macOS, `llama.cpp`'s default thread detection queries `hw.perflevel0.physicalcpu` (which reports only the `2` Super cores on M6, ignoring the `4` Performance and `6` Efficiency cores). In single-sequence mode (`-bucket=0 -num_embedders=12`), each worker explicitly runs with `1` thread so all 12 cores are saturated (`~11.7` cores). In bucketed mode (`-bucket=8192`), pass `-threads=3` (for `-num_embedders=4`) or `-threads=6` / `-threads=12` (for `-num_embedders=1`) to use more than 2 threads.

### Why `-bucket=0 -num_embedders=GOMAXPROCS` Excels on CPU, and Why GoMLX Is Faster

1. **Unpadded Parallel Single-Sequence Execution (`-bucket=0 -num_embedders=GOMAXPROCS`)**:
   - In **`llama.cpp` (`ggml`)**, multi-sequence batches have no separate `batchSize` axis — sequences in a `ubatch` of length $U = \sum_i \text{seqLen}_i$ are concatenated end-to-end and masked via a dense `[U, U]` attention matrix ($U^2$ instead of $\sum_i \text{seqLen}_i^2$). Running 1 unpadded sequence per worker across independent single-threaded contexts eliminates both cross-sequence $O(U^2)$ attention and `ggml` threadpool barrier overhead, jumping from `34.6K` to **`76.3K tokens/s`** on Ryzen 9950X3D (`32` embedders) and from `27.2K` to **`60.0K tokens/s`** on Apple M6 (`12` embedders).
   - In **GoMLX**, switching from bucketed single-embedder execution (`-bucket=8192 -num_embedders=1`) to unpadded dynamic `seq_len` across `GOMAXPROCS` embedders (`-bucket=0`) eliminates all bucket-padding compute, removes intra-op worker synchronization (`OpsExecutionDynamic` runs each independent embedder sequentially on its own hardware thread), and keeps each passage's `[1, seqLen, 384]` activations hot inside that core's private L1/L2 cache — reaching **`364.7K tokens/s`** on Ryzen 9950X3D (`32` embedders) and **`86.4K tokens/s`** on Apple M6 (`12` embedders).
2. **2D Register-Tiled Quantized GEMM (`AVX-512 VNNI` on `amd64`, Hybrid `SME2 + NEON USDOT` on `arm64`) vs Row-by-Row Dot-Products**:
   - In the exact same `-bucket=0` configuration, each call processes a single passage of $M \approx 77$ token rows. `llama.cpp` unpacks `Q4_K` nibbles and 6-bit scales on-the-fly inside row-by-row `ggml_vec_dot_q4_K_q8_K` vector dots.
   - On **`amd64` (Ryzen 9950X3D)**, GoMLX executes 2D register-tiled AVX-512 VNNI (`VPDPBUSD`) GEMMs across multiple sequence rows simultaneously with fused in-tile `Bias + GELU` epilogues — reaching **`222.5K tokens/s`** (`2.9×` faster) with streaming `2 MB` panel chunks (`GOMLX_BACKEND=go`), and **`364.7K tokens/s`** (**`4.8×` faster**, `6.5ms` CPU/passage) when constant GGUF weights are pre-packed once into shared L3-resident K4-interleaved `uint8` panels (`GOMLX_BACKEND=go:quantized_cache`).
   - On **`arm64` (Apple M6)**, GoMLX combines per-cluster SME2 `SUMOPA` + `FMOPA` 16×16 outer-product matrix tiles with 3×16 NEON `USDOT` microkernels across the remaining cores and handwritten NEON `Q8_0`/`Q8_K` activation quantizers, `GELU`, and `LayerNorm` — reaching **`73.6K tokens/s`** (`1.23×` faster than `llama.cpp`) with `GOMLX_BACKEND=go`, and **`86.4K tokens/s`** (**`1.44×` faster**, `10.6ms` CPU/passage) with `GOMLX_BACKEND=go:quantized_cache`.

---

## How to Run

### 1. Run `yzma_embedder` (`llama.cpp` via `yzma`)

Clone this repository (or run directly with `go run`):

```bash
git clone https://github.com/janpfeifer/yzma_embedder.git
cd yzma_embedder
```

- **Default unpadded single-sequence mode across `GOMAXPROCS` embedders (`-bucket=0 -num_embedders=32`, fastest on CPU)**:
  ```bash
  go run . \
    -bucket=0 -num_embedders=32 \
    -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    -limit 1000 -warmup
  ```

- **Unpadded single-sequence mode pinned to physical core count (e.g. `16` embedders)**:
  ```bash
  go run . \
    -bucket=0 -num_embedders=16 \
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

- **Unpadded parallel single-sequence mode (`-bucket=0 -num_embedders=32`, fastest on CPU)**:
  ```bash
  GOEXPERIMENT=simd GOMLX_BACKEND=go:quantized_cache \
    go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed@main \
    -bucket=0 -num_embedders=32 \
    -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    -limit 1000 -warmup -dynamic=true
  ```

- **Token-budget bucketed mode (`-bucket=8192 -num_embedders=1`)**:
  ```bash
  GOEXPERIMENT=simd GOMLX_BACKEND=go:quantized_cache \
    go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed@main \
    -bucket=8192 -num_embedders=1 \
    -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    -limit 1000 -warmup -dynamic=true
  ```

---

## CLI Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-model` | `second-state/All-MiniLM-L6-v2-Embedding-GGUF` | HuggingFace GGUF repository ID or path to a local `.gguf` file. |
| `-gguf_file` | `""` | Specific `.gguf` filename in multi-file GGUF repos (auto-selects `Q4_K_M`, etc. when both `-gguf_file` and `-quant` are empty). Mutually exclusive with `-quant`. |
| `-quant` | `""` | GGUF quantization version/tag to select in multi-file GGUF repos (e.g., `Q4_0`, `Q4_K_M`, `Q8_0`, `F16`; case-insensitive). Mutually exclusive with `-gguf_file`. |
| `-limit` | `-1` | Number of MS MARCO queries to process (`<= 0` processes all ~808K queries). |
| `-msmarco_split` | `train` | MS MARCO v2.1 split (`train`, `validation`, `test`). |
| `-bucket` | `0` | Token budget (`<= 0` uses `bucket.Unpadded()` single-sequence mode; `> 0` uses two-bit token-budget bucketing). |
| `-num_embedders` | `GOMAXPROCS` | Number of parallel `llama.cpp` contexts (`<= 0` also defaults to `GOMAXPROCS`). |
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
