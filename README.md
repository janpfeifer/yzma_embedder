# `yzma_embedder` — MS MARCO GGUF Embedding Benchmark (`llama.cpp` via `yzma` vs GoMLX pure Go backend)

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
- **Dataset**: MS MARCO v2.1 (`train` split, ~77 tokens/passage) after `-warmup` (`-limit=1000` -> **10,000 passages**, **773,300 non-padding tokens** for MiniLM; `-limit=200` -> **2,000 passages**, **153,500 non-padding tokens** for `bge-base-en-v1.5`)
- **`llama.cpp` / `yzma`**: `llama.cpp` `0.9.0-dev` (build `b12110`) via `yzma` `v1.32.0` (CPU backend, `-ngl=0`)

### 1. AMD Ryzen 9 9950X3D (Linux `amd64`, 16 Cores / 32 Threads)

- **CPU**: AMD Ryzen 9 9950X3D (16 physical cores / 32 threads, Zen 5, AVX-512 VNNI, 3D V-Cache)
- **OS / Go**: Linux `amd64`, Go `1.27.3` (`GOEXPERIMENT=simd` for GoMLX)
- **Configuration**: `-bucket=0 -num_embedders=32 -warmup` (GoMLX with `GOMLX_BACKEND=go:quantized_cache -dynamic=true`)

#### 1.1 `second-state/All-MiniLM-L6-v2-Embedding-GGUF` (6 layers, `d_model=384`, `-limit=1000`)

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
    <tr><td><code>F16</code></td><td align="right">736</td><td align="right">56.9K</td><td align="right">441.1 MiB</td><td align="right">1.6K</td><td align="right">125.7K</td><td align="right">489.6 MiB</td></tr>
    <tr><td><code>F16</code> (<code>-force_f32</code>)</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">1.9K</td><td align="right">145.1K</td><td align="right">635.3 MiB</td></tr>
  </tbody>
</table>

#### 1.2 `ChristianAzinn/bge-base-en-v1.5-gguf` (12 layers, `d_model=768`, `-limit=200`)

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
    <tr><td><code>Q2_K</code></td><td align="right">268</td><td align="right">20.6K</td><td align="right">500.8 MiB</td><td align="right">389</td><td align="right">29.9K</td><td align="right">786.0 MiB</td></tr>
    <tr><td><code>Q3_K_S</code></td><td align="right">326</td><td align="right">25.1K</td><td align="right">477.5 MiB</td><td align="right">391</td><td align="right">30.1K</td><td align="right">788.4 MiB</td></tr>
    <tr><td><code>Q3_K_M</code></td><td align="right">303</td><td align="right">23.3K</td><td align="right">528.3 MiB</td><td align="right">433</td><td align="right">33.3K</td><td align="right">828.5 MiB</td></tr>
    <tr><td><code>Q3_K_L</code></td><td align="right">340</td><td align="right">26.2K</td><td align="right">519.9 MiB</td><td align="right">434</td><td align="right">33.4K</td><td align="right">792.0 MiB</td></tr>
    <tr><td><code>Q4_0</code></td><td align="right">311</td><td align="right">23.9K</td><td align="right">528.0 MiB</td><td align="right">471</td><td align="right">36.2K</td><td align="right">808.3 MiB</td></tr>
    <tr><td><code>Q4_K_S</code></td><td align="right">277</td><td align="right">21.3K</td><td align="right">552.9 MiB</td><td align="right">476</td><td align="right">36.6K</td><td align="right">803.8 MiB</td></tr>
    <tr><td><code>Q4_K_M</code></td><td align="right">277</td><td align="right">21.3K</td><td align="right">528.3 MiB</td><td align="right">454</td><td align="right">34.9K</td><td align="right">839.6 MiB</td></tr>
    <tr><td><code>Q5_0</code></td><td align="right">140</td><td align="right">10.8K</td><td align="right">482.3 MiB</td><td align="right">475</td><td align="right">36.6K</td><td align="right">844.2 MiB</td></tr>
    <tr><td><code>Q5_K_S</code></td><td align="right">354</td><td align="right">27.2K</td><td align="right">520.1 MiB</td><td align="right">463</td><td align="right">35.6K</td><td align="right">799.3 MiB</td></tr>
    <tr><td><code>Q5_K_M</code></td><td align="right">346</td><td align="right">26.6K</td><td align="right">527.1 MiB</td><td align="right">452</td><td align="right">34.7K</td><td align="right">807.3 MiB</td></tr>
    <tr><td><code>Q6_K</code></td><td align="right">316</td><td align="right">24.3K</td><td align="right">532.0 MiB</td><td align="right">395</td><td align="right">30.4K</td><td align="right">798.9 MiB</td></tr>
    <tr><td><code>Q8_0</code></td><td align="right">174</td><td align="right">13.4K</td><td align="right">517.7 MiB</td><td align="right">473</td><td align="right">36.3K</td><td align="right">833.3 MiB</td></tr>
    <tr><td><code>FP16</code></td><td align="right">112</td><td align="right">8.6K</td><td align="right">641.4 MiB</td><td align="right">196</td><td align="right">15.1K</td><td align="right">965.4 MiB</td></tr>
    <tr><td><code>FP16</code> (<code>-force_f32</code>)</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">113</td><td align="right">8.7K</td><td align="right">1.5 GiB</td></tr>
    <tr><td><code>FP32</code></td><td align="right">105</td><td align="right">8.1K</td><td align="right">791.8 MiB</td><td align="right">114</td><td align="right">8.8K</td><td align="right">1.5 GiB</td></tr>
  </tbody>
</table>

### 2. Apple Mac mini M6 (macOS `arm64`, 12 Cores)

- **CPU**: Apple M6 (12 cores total: 2 Super, 4 Performance, 6 Efficiency; ARMv9.2-A with NEON `USDOT` / `I8MM` and SME2)
- **OS / Go**: macOS `darwin/arm64`, Go `1.27.3` (`GOEXPERIMENT=simd` for GoMLX; activates NEON + SME2 kernels)
- **Configuration**: `-bucket=0 -num_embedders=12 -warmup` (GoMLX with `GOMLX_BACKEND=go:quantized_cache -dynamic=true`)

#### 2.1 `second-state/All-MiniLM-L6-v2-Embedding-GGUF` (6 layers, `d_model=384`, `-limit=1000`)

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
    <tr><td><code>Q2_K</code></td><td align="right">859</td><td align="right">66.5K</td><td align="right">385.8 MiB</td><td align="right">1K</td><td align="right">81.2K</td><td align="right">388.1 MiB</td></tr>
    <tr><td><code>Q3_K_S</code></td><td align="right">848</td><td align="right">65.7K</td><td align="right">372.3 MiB</td><td align="right">1K</td><td align="right">79.2K</td><td align="right">401.2 MiB</td></tr>
    <tr><td><code>Q3_K_M</code></td><td align="right">751</td><td align="right">58.2K</td><td align="right">374.7 MiB</td><td align="right">1.1K</td><td align="right">88.9K</td><td align="right">394.4 MiB</td></tr>
    <tr><td><code>Q3_K_L</code></td><td align="right">695</td><td align="right">53.8K</td><td align="right">355.4 MiB</td><td align="right">1.1K</td><td align="right">88.8K</td><td align="right">399.7 MiB</td></tr>
    <tr><td><code>Q4_0</code></td><td align="right">937</td><td align="right">72.5K</td><td align="right">346.7 MiB</td><td align="right">1.1K</td><td align="right">89K</td><td align="right">381.2 MiB</td></tr>
    <tr><td><code>Q4_K_S</code></td><td align="right">742</td><td align="right">57.4K</td><td align="right">364.5 MiB</td><td align="right">1.2K</td><td align="right">89.2K</td><td align="right">391.0 MiB</td></tr>
    <tr><td><code>Q4_K_M</code></td><td align="right">712</td><td align="right">55.1K</td><td align="right">400.0 MiB</td><td align="right">1.1K</td><td align="right">84.6K</td><td align="right">393.5 MiB</td></tr>
    <tr><td><code>Q5_0</code></td><td align="right">728</td><td align="right">56.3K</td><td align="right">370.2 MiB</td><td align="right">1.1K</td><td align="right">87.7K</td><td align="right">389.7 MiB</td></tr>
    <tr><td><code>Q5_K_S</code></td><td align="right">655</td><td align="right">50.7K</td><td align="right">452.3 MiB</td><td align="right">1.1K</td><td align="right">88.3K</td><td align="right">395.3 MiB</td></tr>
    <tr><td><code>Q5_K_M</code></td><td align="right">664</td><td align="right">51.4K</td><td align="right">382.6 MiB</td><td align="right">1.1K</td><td align="right">82.7K</td><td align="right">390.8 MiB</td></tr>
    <tr><td><code>Q6_K</code></td><td align="right">695</td><td align="right">53.8K</td><td align="right">360.4 MiB</td><td align="right">1K</td><td align="right">78.2K</td><td align="right">391.3 MiB</td></tr>
    <tr><td><code>Q8_0</code></td><td align="right">860</td><td align="right">66.5K</td><td align="right">376.3 MiB</td><td align="right">1.1K</td><td align="right">86.8K</td><td align="right">386.0 MiB</td></tr>
    <tr><td><code>F16</code></td><td align="right">725</td><td align="right">56.1K</td><td align="right">418.2 MiB</td><td align="right">800</td><td align="right">61.9K</td><td align="right">447.6 MiB</td></tr>
    <tr><td><code>F16</code> (<code>-force_f32</code>)</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">867</td><td align="right">67.1K</td><td align="right">527.9 MiB</td></tr>
  </tbody>
</table>

#### 2.2 `ChristianAzinn/bge-base-en-v1.5-gguf` (12 layers, `d_model=768`, `-limit=200`)

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
    <tr><td><code>Q2_K</code></td><td align="right">83</td><td align="right">6.4K</td><td align="right">502.6 MiB</td><td align="right">119</td><td align="right">9.2K</td><td align="right">681.4 MiB</td></tr>
    <tr><td><code>Q3_K_S</code></td><td align="right">82</td><td align="right">6.3K</td><td align="right">525.3 MiB</td><td align="right">115</td><td align="right">8.8K</td><td align="right">645.0 MiB</td></tr>
    <tr><td><code>Q3_K_M</code></td><td align="right">107</td><td align="right">8.3K</td><td align="right">531.0 MiB</td><td align="right">133</td><td align="right">10.2K</td><td align="right">691.3 MiB</td></tr>
    <tr><td><code>Q3_K_L</code></td><td align="right">93</td><td align="right">7.2K</td><td align="right">550.7 MiB</td><td align="right">131</td><td align="right">10.1K</td><td align="right">698.0 MiB</td></tr>
    <tr><td><code>Q4_0</code></td><td align="right">154</td><td align="right">11.8K</td><td align="right">431.8 MiB</td><td align="right">163</td><td align="right">12.6K</td><td align="right">686.1 MiB</td></tr>
    <tr><td><code>Q4_K_S</code></td><td align="right">106</td><td align="right">8.2K</td><td align="right">423.0 MiB</td><td align="right">159</td><td align="right">12.2K</td><td align="right">672.5 MiB</td></tr>
    <tr><td><code>Q4_K_M</code></td><td align="right">104</td><td align="right">8K</td><td align="right">418.2 MiB</td><td align="right">148</td><td align="right">11.4K</td><td align="right">655.2 MiB</td></tr>
    <tr><td><code>Q5_0</code></td><td align="right">86</td><td align="right">6.6K</td><td align="right">562.1 MiB</td><td align="right">157</td><td align="right">12K</td><td align="right">686.0 MiB</td></tr>
    <tr><td><code>Q5_K_S</code></td><td align="right">80</td><td align="right">6.2K</td><td align="right">433.0 MiB</td><td align="right">159</td><td align="right">12.2K</td><td align="right">684.6 MiB</td></tr>
    <tr><td><code>Q5_K_M</code></td><td align="right">88</td><td align="right">6.8K</td><td align="right">443.7 MiB</td><td align="right">153</td><td align="right">11.8K</td><td align="right">707.0 MiB</td></tr>
    <tr><td><code>Q6_K</code></td><td align="right">81</td><td align="right">6.2K</td><td align="right">458.1 MiB</td><td align="right">110</td><td align="right">8.4K</td><td align="right">683.9 MiB</td></tr>
    <tr><td><code>Q8_0</code></td><td align="right">130</td><td align="right">10K</td><td align="right">497.9 MiB</td><td align="right">161</td><td align="right">12.4K</td><td align="right">657.3 MiB</td></tr>
    <tr><td><code>FP16</code></td><td align="right">82</td><td align="right">6.3K</td><td align="right">695.5 MiB</td><td align="right">103</td><td align="right">7.9K</td><td align="right">891.4 MiB</td></tr>
    <tr><td><code>FP16</code> (<code>-force_f32</code>)</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">117</td><td align="right">9K</td><td align="right">1.0 GiB</td></tr>
    <tr><td><code>FP32</code></td><td align="right">97</td><td align="right">7.4K</td><td align="right">747.2 MiB</td><td align="right">109</td><td align="right">8.4K</td><td align="right">1.1 GiB</td></tr>
  </tbody>
</table>

> **Note on Apple M6 3-tier core topology (`Super` / `Performance` / `Efficiency`) in `llama.cpp`:** On macOS, `llama.cpp`'s default thread detection queries `hw.perflevel0.physicalcpu` (which reports only the `2` Super cores on M6, ignoring the `4` Performance and `6` Efficiency cores). In single-sequence mode (`-bucket=0 -num_embedders=12`), each worker explicitly runs with `1` thread so all 12 cores are saturated (`~11.7` cores). In bucketed mode (`-bucket=8192`), pass `-threads=3` (for `-num_embedders=4`) or `-threads=6` / `-threads=12` (for `-num_embedders=1`) to use more than 2 threads.

### Why `-bucket=0 -num_embedders=GOMAXPROCS` Excels on CPU, and Why GoMLX Is Faster

1. **Unpadded Parallel Single-Sequence Execution (`-bucket=0 -num_embedders=GOMAXPROCS`)**:
   - In **`llama.cpp` (`ggml`)**, multi-sequence batches have no separate `batchSize` axis — sequences in a `ubatch` of length $U = \sum_i \text{seqLen}_i$ are concatenated end-to-end and masked via a dense `[U, U]` attention matrix ($U^2$ instead of $\sum_i \text{seqLen}_i^2$). Running 1 unpadded sequence per worker across independent single-threaded contexts eliminates both cross-sequence $O(U^2)$ attention and `ggml` threadpool barrier overhead.
   - In **GoMLX**, running unpadded dynamic `seq_len` across `GOMAXPROCS` independent embedders (`-bucket=0`) eliminates all bucket-padding compute, removes intra-op worker synchronization (`OpsExecutionDynamic` runs each embedder sequentially on its own hardware thread), and keeps each passage's `[1, seqLen, d_model]` activations hot inside that core's private L1/L2 cache.

2. **`ggml` `CPU_REPACK` Coverage vs GoMLX's Universal Pre-Packed 2D GEMMs (`AVX-512 VNNI` on `amd64`, Hybrid `SME2 + NEON USDOT` on `arm64`)**:
   - **On `MiniLM-L6-v2` (`d_model = 384`, 6 layers, `22M` params)**: `384` is not a multiple of `256` (the K-quant super-block size), so K-quantization fell back heavily on 384-dim matrices (`-verbose_llama`):
     - `Q4_K_M` ended up with `28 × q5_0`, `2 × q8_0` encoder weights, `2 × q6_K`, and only `4 × q4_K` (`1.27 MiB` `CPU_REPACK`), leaving 97 of 101 tensors un-repacked (`76.4K tokens/s` on Ryzen, `55.1K` on M6).
     - `Q5_0`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, and `Q8_0` had `0` `CPU_REPACK` tensors (dropping as low as `42.9K tokens/s` on Ryzen and `50.7K` on M6).
     - Only `Q4_0` (block size `32`, repacking all 36 encoder weights = `5.70 MiB` `CPU_REPACK`) and `Q2_K`/`Q3_K_S` (`30 × iq4_nl` = `3.80 MiB`) got broad repacking, peaking at `117.6K tokens/s` (`Q4_0`) on Ryzen and `72.5K tokens/s` on M6.
   - **On `bge-base-en-v1.5` (`d_model = 768`, 12 layers, `109M` params)**: `768 = 3 × 256` **is** a multiple of `256`, so `llama.cpp` avoids the `q5_0` fallback (`Q4_K_M` repacks `60 × q4_K` = `36.07 MiB`, `Q4_K_S` repacks `65 × q4_K` = `40.50 MiB`, `Q4_0` repacks `72 × q4_0` = `45.56 MiB`).
     - Yet `llama.cpp`'s throughput on `bge-base` still varies sharply by format: on Ryzen, `Q5_K_S` (`27.2K tokens/s`) and `Q5_K_M` (`26.6K`) edge out repacked `Q4_0` (`23.9K`) and `Q4_K_M` (`21.3K`), while `Q5_0` (`10.8K`) and `Q8_0` (`13.4K`) drop by more than `2×`; on Apple M6, only `Q4_0` (`11.8K tokens/s`) and `Q8_0` (`10K`) reach double-digit K tokens/s, while K-quants drop to `6.2K–8.3K tokens/s`.
   - **GoMLX stays faster and uniform across all quantizations on both models and both CPUs**: With `GOMLX_BACKEND=go:quantized_cache`, GoMLX pre-unpacks **every** GGML quantized format (`Q2_K`–`Q8_0`) once into shared K4-interleaved `uint8` panels and executes 2D register-tiled GEMMs (`AVX-512 VNNI VPDPBUSD` on Ryzen; `SME2 + NEON USDOT` on Apple M6) with fused `Bias + GELU` epilogues:
     - **Ryzen 9950X3D — `MiniLM-L6-v2` (`22M`)**: GoMLX sustains **`328.2K–349.9K tokens/s`** across all quantized formats — **`2.9×` faster** than `llama.cpp` on fully repacked `Q4_0` (`346.2K` vs `117.6K`), **`4.5×` faster** on `Q4_K_M` (`342K` vs `76.4K`), **`8.1×` faster** on `Q5_K_S` (`348.7K` vs `42.9K`), and **`2.2×` faster** on native `F16` (`125.7K` vs `56.9K`, or **`2.6×` faster** at `145.1K` with `-force_f32`).
     - **Ryzen 9950X3D — `bge-base-en-v1.5` (`109M`)**: Even with `d_model = 768` allowing full `Q4_K`/`Q4_0` `CPU_REPACK` in `llama.cpp`, GoMLX wins on **every single format** (`29.9K–36.6K tokens/s` quantized): **`1.51×` faster** on `Q4_0` (`36.2K` vs `23.9K`), **`1.64×` faster** on `Q4_K_M` (`34.9K` vs `21.3K`), **`1.72×` faster** on `Q4_K_S` (`36.6K` vs `21.3K`), **`1.31×` faster** on `llama.cpp`'s fastest quant `Q5_K_S` (`35.6K` vs `27.2K`), **`2.71×` faster** on `Q8_0` (`36.3K` vs `13.4K`), **`3.39×` faster** on `Q5_0` (`36.6K` vs `10.8K`), **`1.76×` faster** on native `FP16` (`15.1K` vs `8.6K`), and **`1.09×` faster** on `FP32` (`8.8K` vs `8.1K`). With `-force_f32`, `FP16` drops back to `FP32` speed and memory (`8.7K tokens/s`, `1.5 GiB`), confirming the benefit of native `Float16` weights and GEMMs.
     - **Apple M6 — `MiniLM-L6-v2` (`22M`)**: GoMLX now beats `llama.cpp` across **every single quantization and float format**: `78.2K–89.2K tokens/s` vs `50.7K–72.5K` on quantized models (**`1.23×` faster** on `Q4_0` at `89K` vs `72.5K`, **`1.54×` faster** on `Q4_K_M` at `84.6K` vs `55.1K`, **`1.74×` faster** on `Q5_K_S` at `88.3K` vs `50.7K`), **`1.10×` faster** on native `F16` (`61.9K` vs `56.1K`, saving `80.3 MiB` RSS over `-force_f32`), and **`1.20×` faster** with `F16 -force_f32` (`67.1K` vs `56.1K`).
     - **Apple M6 — `bge-base-en-v1.5` (`109M`)**: GoMLX wins on **every single format** (`8.4K–12.6K tokens/s` quantized vs `6.2K–11.8K`): **`1.07×` faster** on `Q4_0` (`12.6K` vs `11.8K`), **`1.43×` faster** on `Q4_K_M` (`11.4K` vs `8K`), **`1.49×` faster** on `Q4_K_S` (`12.2K` vs `8.2K`), **`1.97×` faster** on `Q5_K_S` (`12.2K` vs `6.2K`), **`1.82×` faster** on `Q5_0` (`12K` vs `6.6K`), **`1.24×` faster** on `Q8_0` (`12.4K` vs `10K`), **`1.25×` faster** on native `FP16` (`7.9K` vs `6.3K`, or **`1.43×` faster** at `9K` with `-force_f32`), and **`1.14×` faster** on `FP32` (`8.4K` vs `7.4K`).

3. **Cache Footprint & Memory Trade-Off (`22M` vs `109M` Models)**:
   - On `MiniLM-L6-v2` (`22M`), the entire pre-packed weight set fits comfortably inside Ryzen 9 9950X3D's L3 / 3D V-Cache, allowing GoMLX to achieve **`2.9×–8.1×`** speedups with only ~`50–80 MiB` extra RSS (`476–499 MiB` vs `370–460 MiB`). Because even `FP32` weights fit in 3D V-Cache, `-force_f32` (`145.1K tokens/s`, `635.3 MiB`) is faster than native `F16` (`125.7K tokens/s`, `489.6 MiB`) by avoiding `Float16 <-> Float32` conversions in GEMM/LayerNorm/SDPA, while native `F16` saves `145.7 MiB` of RSS.
   - On `bge-base-en-v1.5` (`109M`, ~8× the GEMM FLOPs and ~5× the parameters), the pre-packed weights exceed the smaller `32 MiB` non-3D CCD L3 cache and become more memory-bandwidth-bound across CCDs/DRAM (`786–844 MiB` vs `478–553 MiB` RSS for quantized formats). Here, keeping weights in native `Float16` cuts GoMLX's `FP16` RSS from `1.5 GiB` (`-force_f32` / `FP32`) down to `965.4 MiB` and jumps throughput from `8.7K` to `15.1K tokens/s` (`1.76×` faster than `llama.cpp`'s `8.6K`), while quantized GoMLX still retains a consistent **`1.20×–3.39×`** lead over `llama.cpp`.

---

## How to Run

### Discovering Available Quantized Versions (`hubinfo`)

Multi-file GGUF repositories typically publish many quantizations (for example, `second-state/All-MiniLM-L6-v2-Embedding-GGUF` contains 13 `.gguf` files from `Q2_K` to `F16`). When no quantization or file is specified, both `yzma_embedder` and `benchmark_embed` default to `Q4_K_M` (`all-MiniLM-L6-v2-Q4_K_M.gguf`).

To list all files and quantization variants in a HuggingFace repository, install `hubinfo` from `go-huggingface`:

```bash
go install github.com/gomlx/go-huggingface/cmd/hubinfo@latest
hubinfo second-state/All-MiniLM-L6-v2-Embedding-GGUF
```

### 1. Run `yzma_embedder` (`llama.cpp` via `yzma`)

Clone this repository (or run directly with `go run`):

```bash
git clone https://github.com/janpfeifer/yzma_embedder.git
cd yzma_embedder
```

Use **`-quant`** to select a quantization tag case-insensitively (e.g. `-quant=Q4_0`, `-quant=Q4_K_M`, `-quant=Q8_0`, `-quant=F16`) without typing the full `.gguf` filename, or use **`-gguf_file`** to specify an exact `.gguf` filename (mutually exclusive with `-quant`).

- **Default unpadded single-sequence mode across `GOMAXPROCS` embedders (`-bucket=0 -num_embedders=32`, fastest on CPU)**:
  ```bash
  # Default quantization (Q4_K_M):
  go run . \
    -bucket=0 -num_embedders=32 \
    -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    -limit 1000 -warmup

  # Specific quantization via -quant (e.g. Q4_0):
  go run . \
    -bucket=0 -num_embedders=32 \
    -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    -quant=Q4_0 \
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

Requires Go 1.27+ with `GOEXPERIMENT=simd` (use `-gguf_file` to select a specific quantization file, such as `all-MiniLM-L6-v2-Q4_0.gguf`, and `-force_f32` to force `Float32` execution on `F16`/`BF16` GGUF models instead of native half-precision):

- **Unpadded parallel single-sequence mode (`-bucket=0 -num_embedders=32`, fastest on CPU)**:
  ```bash
  GOEXPERIMENT=simd GOMLX_BACKEND=go:quantized_cache \
    go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed@main \
    -bucket=0 -num_embedders=32 \
    -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    -gguf_file=all-MiniLM-L6-v2-Q4_0.gguf \
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
