/*
 *	Copyright 2026 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

// Benchmark GGUF sentence embeddings on the MS MARCO dataset using llama.cpp
// via github.com/hybridgroup/yzma, matching the dataset loop, go-huggingface
// tokenizer, -limit flag, and summary metrics table of
// github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed.
//
// Supports two execution modes controlled by -bucket:
//   - -bucket > 0 (default 8192): token-budget bucketing (same Bucketizer as benchmark_embed),
//     packing multiple unpadded sequences into llama.cpp micro-batches.
//   - -bucket=0: unpadded single-sequence calls (one passage per llama.cpp call, no bucketing/padding),
//     defaulting -num_embedders to GOMAXPROCS with 1 thread per llama.cpp context.
//
// Usage:
//
//	# Single-sequence mode across GOMAXPROCS embedders:
//	go run github.com/janpfeifer/yzma_embedder@latest --bucket=0 -model second-state/All-MiniLM-L6-v2-Embedding-GGUF -limit 1000 -warmup
//
//	# Bucketed mode:
//	go run github.com/janpfeifer/yzma_embedder@latest -bucket=8192 -model second-state/All-MiniLM-L6-v2-Embedding-GGUF -limit 1000 -num_embedders=1 -warmup
package main

import (
	"flag"
	"fmt"
	"math"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/datasets"
	minilmgguf "github.com/gomlx/go-huggingface/examples/minilm-l6-v2-gguf"
	"github.com/gomlx/go-huggingface/examples/msmarco"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/gguf"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers"
	tapi "github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/bucket"
	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
	"k8s.io/klog/v2"
)

var (
	flagModel = flag.String("model", minilmgguf.Repository,
		fmt.Sprintf("HuggingFace GGUF model repository or local .gguf path. Example: %q", minilmgguf.Repository))
	flagGGUFFile = flag.String("gguf_file", "",
		"Specific .gguf filename to load when -model is a multi-file GGUF repository. "+
			"When empty, selects by community preference order (Q4_K_M, Q4_K_S, ...).")
	flagLimit        = flag.Int("limit", -1, "Limit the number of queries indexed. Set <= 0 to use all.")
	flagMSMarcoSplit = flag.String("msmarco_split", msmarco.TrainSplit, "Split to read from MS MARCO dataset (e.g. 'train', 'validation', 'test')")
	flagBucketBudget = flag.Int("bucket", 8*1024,
		"Bucket budget in number of tokens (> 0 uses two-bit token-budget bucketing; 0 disables bucketing/packing and runs 1 unpadded sequence per llama.cpp call).")
	flagParallelEmbedders = flag.Int("num_embedders", 0,
		"Number of parallel llama.cpp embedder contexts (0 = auto: GOMAXPROCS when -bucket=0, 1 when -bucket > 0).")
	flagProfile = flag.Bool("profile", false, "Enable pprof HTTP server on localhost:6060.")
	flagCPUProf = flag.String("cpuprof", "", "Write CPU profile to file.")
	flagWarmup  = flag.Bool("warmup", false, "Do a warmup run over the dataset first.")
	flagShapes  = flag.Bool("shapes", false, "Report the shapes encountered during the warmup run.")
	flagDynamic = flag.Bool("dynamic", true, "Accepted for CLI flag compatibility with benchmark_embed.")

	// llama.cpp / yzma flags.
	flagLibDir    = flag.String("lib", defaultLibDir(), "Directory to install/load llama.cpp shared libraries.")
	flagProcessor = flag.String("processor", "cpu", "llama.cpp processor library variant to auto-download if missing (e.g. 'cpu', 'cuda', 'vulkan', 'metal').")
	flagNGL       = flag.Int("ngl", 0, "Number of GPU layers to offload in llama.cpp (0 = CPU only).")
	flagThreads   = flag.Int("threads", 0, "Number of CPU threads per llama.cpp context (0 = auto: divides physical cores across -num_embedders, minimum 1).")
	flagUBatch    = flag.Int("ubatch", 512, "Physical micro-batch size (n_ubatch) in bucketed mode (-bucket > 0).")
	flagFlashAttn = flag.Bool("flash_attn", false, "Enable Flash Attention in llama.cpp context.")
	flagVerbose   = flag.Bool("verbose_llama", false, "Enable llama.cpp internal logging.")
	flagOutRef    = flag.String("out", "", "If set, compute the 4 reference prompts, write similarity_embeddings.txt to this path, and exit.")
)

var referencePrompts = []string{
	"What is the capital of China?",
	"Explain gravity",
	"The capital of China is Beijing.",
	"Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
}

func defaultLibDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "/tmp/yzma_lib"
	}
	return filepath.Join(home, ".cache", "yzma", "lib")
}

func mapHas[K comparable, V any](m map[K]V, k K) bool {
	_, ok := m[k]
	return ok
}

type resolvedModel struct {
	ggufPath    string
	displayFile string
	tokenizer   tokenizers.Tokenizer
	poolingType llama.PoolingType
	normalize   bool
}

func resolveModelAndTokenizer(modelSpec, ggufFile string) *resolvedModel {
	if info, err := os.Stat(modelSpec); err == nil && !info.IsDir() && strings.HasSuffix(strings.ToLower(modelSpec), ".gguf") {
		f := mustRunWithElapsedTime("Parsing local GGUF header", func() (*gguf.File, error) {
			return gguf.Open(modelSpec)
		})
		tok := mustRunWithElapsedTime("Loading tokenizer", func() (tokenizers.Tokenizer, error) {
			return f.LoadTokenizer(nil)
		})
		pooling, normalize := poolingFromGGUFHeader(f)
		return &resolvedModel{
			ggufPath:    modelSpec,
			displayFile: filepath.Base(modelSpec),
			tokenizer:   tok,
			poolingType: pooling,
			normalize:   normalize,
		}
	}

	repo := hub.New(modelSpec)
	if err := repo.DownloadInfo(false); err != nil {
		klog.Fatalf("Failed to load repo info for %q: %v", modelSpec, err)
	}

	hfModel := mustRunWithElapsedTime("Loading model configurations", func() (*transformer.Model, error) {
		m, err := transformer.LoadModel(repo)
		if err != nil {
			return nil, err
		}
		if ggufFile != "" {
			m.WithGGUFFile(ggufFile)
		}
		return m, nil
	})
	if !hfModel.IsGGUF() {
		klog.Fatalf("Repository %q does not select a GGUF file (use a GGUF repo or specify -gguf_file)", modelSpec)
	}
	fmt.Printf("- GGUF file: %s\n", hfModel.GGUFFile())

	tok := mustRunWithElapsedTime("Loading tokenizer", func() (tokenizers.Tokenizer, error) {
		return hfModel.GetTokenizer()
	})

	ggufPath := mustRunWithElapsedTime("Resolving GGUF weights file", func() (string, error) {
		return repo.DownloadFile(hfModel.GGUFFile())
	})

	return &resolvedModel{
		ggufPath:    ggufPath,
		displayFile: hfModel.GGUFFile(),
		tokenizer:   tok,
		poolingType: poolingFromHFModel(hfModel),
		normalize:   shouldNormalizeHFModel(hfModel),
	}
}

func poolingFromGGUFHeader(f *gguf.File) (llama.PoolingType, bool) {
	arch := strings.ToLower(strings.TrimSpace(f.Architecture()))
	if arch != "" {
		if kv, ok := f.GetKeyValue(arch + ".pooling_type"); ok {
			switch kv.Int64() {
			case 1:
				return llama.PoolingTypeMean, true
			case 2:
				return llama.PoolingTypeCLS, true
			case 3:
				return llama.PoolingTypeLast, true
			}
		}
	}
	return llama.PoolingTypeUnspecified, true
}

func poolingFromHFModel(m *transformer.Model) llama.PoolingType {
	if m.PoolingConfig != nil {
		switch {
		case m.PoolingConfig.PoolingModeMeanTokens:
			return llama.PoolingTypeMean
		case m.PoolingConfig.PoolingModeClsToken:
			return llama.PoolingTypeCLS
		case m.PoolingConfig.PoolingModeLastToken:
			return llama.PoolingTypeLast
		}
	}
	return llama.PoolingTypeUnspecified
}

func shouldNormalizeHFModel(m *transformer.Model) bool {
	if len(m.Modules) == 0 {
		return true
	}
	for _, mod := range m.Modules {
		if mod.Type == "sentence_transformers.models.Normalize" {
			return true
		}
	}
	return false
}

// llamaWorker holds one llama.Context and pre-allocated buffers for a single embedder goroutine.
type llamaWorker struct {
	lctx           llama.Context
	mem            llama.Memory
	batch          llama.Batch
	tokenSlice     []llama.Token // used in single-sequence mode (-bucket=0)
	singleSeq      bool
	useEncode      bool
	normalize      bool
	nEmbd          int
	padID          int
	maxSeqTokens   int
	maxBatchTokens int
	maxSeqs        int
	outBuf         []float32
}

func newLlamaWorker(
	model llama.Model,
	bucketBudget int,
	numEmbedders int,
	threads int,
	requestedPooling llama.PoolingType,
	normalize bool,
	padID int,
) (*llamaWorker, error) {
	nEmbd := int(llama.ModelNEmbd(model))
	nCtxTrain := int(llama.ModelNCtxTrain(model))
	hasEncoder := llama.ModelHasEncoder(model)
	hasDecoder := llama.ModelHasDecoder(model)
	singleSeq := bucketBudget <= 0

	maxSeqTokens := nCtxTrain
	if maxSeqTokens <= 0 || (singleSeq && maxSeqTokens > 8192) {
		maxSeqTokens = max(bucketBudget, 512)
	}

	var maxBatchTokens, uBatchTokens, desiredSeqMax int
	if singleSeq {
		maxBatchTokens = maxSeqTokens
		uBatchTokens = maxSeqTokens
		desiredSeqMax = 1
	} else {
		maxBatchTokens = max(bucketBudget, maxSeqTokens, 512)
		uBatchTokens = maxBatchTokens
		if *flagUBatch > 0 {
			uBatchTokens = min(max(*flagUBatch, maxSeqTokens), maxBatchTokens)
		}
		desiredSeqMax = max(bucketBudget/8, 1)
	}

	buildParams := func(seqMax int, pooling llama.PoolingType) llama.ContextParams {
		p := llama.ContextDefaultParams()
		p.NCtx = uint32(maxBatchTokens)
		p.NBatch = uint32(maxBatchTokens)
		p.NUbatch = uint32(uBatchTokens)
		p.NSeqMax = uint32(seqMax)
		if seqMax > 1 {
			p.KVUnified = 1
		}
		p.PoolingType = pooling
		p.Embeddings = 1
		if *flagFlashAttn {
			p.FlashAttentionType = llama.FlashAttentionTypeEnabled
		}
		if threads > 0 {
			p.NThreads = int32(threads)
			p.NThreadsBatch = int32(threads)
		} else if numEmbedders > 1 && p.NThreadsBatch > 0 {
			divided := max(p.NThreadsBatch/int32(numEmbedders), 1)
			p.NThreads = divided
			p.NThreadsBatch = divided
		} else if singleSeq {
			// Ensure single-token sequences also use full batch thread count when numEmbedders == 1.
			p.NThreads = p.NThreadsBatch
		}
		return p
	}

	candidates := []int{desiredSeqMax}
	if !singleSeq {
		for _, fallback := range []int{512, 256, 128, 64, 32, 1} {
			if fallback < desiredSeqMax {
				candidates = append(candidates, fallback)
			}
		}
	}

	var lctx llama.Context
	var initErr error
	for _, seqMax := range candidates {
		lctx, initErr = llama.InitFromModel(model, buildParams(seqMax, requestedPooling))
		if initErr == nil && lctx != 0 {
			break
		}
	}
	if lctx == 0 {
		return nil, fmt.Errorf("llama.InitFromModel failed (tried n_seq_max=%v): %w", candidates, initErr)
	}

	// Ensure a valid sequence pooling mode is active so GetEmbeddingsSeq returns [n_embd].
	if llama.GetPoolingType(lctx) == llama.PoolingTypeNone {
		fallbackPooling := llama.PoolingTypeMean
		if hasDecoder && !hasEncoder {
			fallbackPooling = llama.PoolingTypeLast
		}
		seqMax := int(llama.NSeqMax(lctx))
		llama.Free(lctx)
		lctx, initErr = llama.InitFromModel(model, buildParams(seqMax, fallbackPooling))
		if initErr != nil || lctx == 0 {
			return nil, fmt.Errorf("llama.InitFromModel with fallback pooling failed: %w", initErr)
		}
	}

	useEncode := (hasEncoder && !hasDecoder) || !llama.GetCausalAttn(lctx)
	actualNBatch := int(llama.NBatch(lctx))
	actualNUBatch := int(llama.NUBatch(lctx))
	subBatchTokens := actualNBatch
	if useEncode && actualNUBatch > 0 && actualNUBatch < subBatchTokens {
		subBatchTokens = actualNUBatch
	}
	actualSeqMax := int(llama.NSeqMax(lctx))
	mem, _ := llama.GetMemory(lctx)
	batch := llama.BatchInit(int32(subBatchTokens), 0, 1)

	var tokenSlice []llama.Token
	if singleSeq {
		// Pre-populate Pos=[0..subBatchTokens-1], SeqId=0, Logits=true once so each
		// single-sequence call only writes token IDs and NTokens with zero FFI overhead.
		seqZero := []llama.SeqId{0}
		for pos := range subBatchTokens {
			if err := batch.Add(0, llama.Pos(pos), seqZero, true); err != nil {
				_ = llama.BatchFree(batch)
				_ = llama.Free(lctx)
				return nil, fmt.Errorf("batch pre-init failed at pos %d: %w", pos, err)
			}
		}
		batch.NTokens = 0
		tokenSlice = unsafe.Slice(batch.Token, subBatchTokens)
	}

	return &llamaWorker{
		lctx:           lctx,
		mem:            mem,
		batch:          batch,
		tokenSlice:     tokenSlice,
		singleSeq:      singleSeq,
		useEncode:      useEncode,
		normalize:      normalize,
		nEmbd:          nEmbd,
		padID:          padID,
		maxSeqTokens:   maxSeqTokens,
		maxBatchTokens: subBatchTokens,
		maxSeqs:        actualSeqMax,
		outBuf:         make([]float32, actualSeqMax*nEmbd),
	}, nil
}

func (w *llamaWorker) Close() {
	if w.batch.Token != nil {
		_ = llama.BatchFree(w.batch)
		w.batch = llama.Batch{}
	}
	if w.lctx != 0 {
		_ = llama.Free(w.lctx)
		w.lctx = 0
	}
}

func (w *llamaWorker) description() string {
	mode := "decode"
	if w.useEncode {
		mode = "encode"
	}
	seqMode := "bucketed"
	if w.singleSeq {
		seqMode = "single-seq"
	}
	return fmt.Sprintf(
		"llama.cpp via yzma (%s/%s, %s %s, threads=%d, batch_threads=%d, n_ctx=%d, n_batch=%d, n_ubatch=%d, n_seq_max=%d, pooling=%d, normalize=%t, ngl=%d)",
		runtime.GOOS, runtime.GOARCH, seqMode, mode,
		llama.NThreads(w.lctx), llama.NThreadsBatch(w.lctx),
		llama.NCtx(w.lctx), llama.NBatch(w.lctx), llama.NUBatch(w.lctx), llama.NSeqMax(w.lctx),
		llama.GetPoolingType(w.lctx), w.normalize, *flagNGL,
	)
}

func (w *llamaWorker) writeNormalized(seqID int, vec []float32) {
	dst := w.outBuf[seqID*w.nEmbd : (seqID+1)*w.nEmbd]
	if w.normalize {
		var normSq float32
		for _, v := range vec {
			normSq += v * v
		}
		norm := float32(math.Sqrt(float64(normSq)))
		if norm < 1e-12 {
			norm = 1e-12
		}
		invNorm := 1.0 / norm
		for j, v := range vec {
			dst[j] = v * invNorm
		}
	} else {
		copy(dst, vec)
	}
}

// EmbedTokens embeds a single unpadded sequence (used when -bucket=0).
func (w *llamaWorker) EmbedTokens(ids []int) (int, error) {
	tokLen := len(ids)
	if tokLen == 0 {
		return 0, nil
	}
	if tokLen > w.maxBatchTokens {
		tokLen = w.maxBatchTokens
	}

	for i := range tokLen {
		w.tokenSlice[i] = llama.Token(ids[i])
	}
	w.batch.NTokens = int32(tokLen)

	if !w.useEncode && w.mem != 0 {
		_ = llama.MemoryClear(w.mem, false)
	}

	var ret int32
	var err error
	if w.useEncode {
		ret, err = llama.Encode(w.lctx, w.batch)
	} else {
		ret, err = llama.Decode(w.lctx, w.batch)
	}
	if err != nil || ret != 0 {
		return 0, fmt.Errorf("llama %s failed (n_tokens=%d): ret=%d err=%w",
			map[bool]string{true: "Encode", false: "Decode"}[w.useEncode],
			tokLen, ret, err)
	}

	vec, err := llama.GetEmbeddingsSeq(w.lctx, 0, int32(w.nEmbd))
	if err != nil || len(vec) == 0 {
		return 0, fmt.Errorf("GetEmbeddingsSeq(0) failed: %w", err)
	}
	w.writeNormalized(0, vec)
	return tokLen, nil
}

func (w *llamaWorker) flushSubBatch(seqCount int) error {
	if seqCount == 0 || w.batch.NTokens == 0 {
		return nil
	}
	if !w.useEncode && w.mem != 0 {
		_ = llama.MemoryClear(w.mem, false)
	}

	var ret int32
	var err error
	if w.useEncode {
		ret, err = llama.Encode(w.lctx, w.batch)
	} else {
		ret, err = llama.Decode(w.lctx, w.batch)
	}
	if err != nil || ret != 0 {
		return fmt.Errorf("llama %s failed (n_tokens=%d, seqs=%d): ret=%d err=%w",
			map[bool]string{true: "Encode", false: "Decode"}[w.useEncode],
			w.batch.NTokens, seqCount, ret, err)
	}

	for seqID := range seqCount {
		vec, err := llama.GetEmbeddingsSeq(w.lctx, llama.SeqId(seqID), int32(w.nEmbd))
		if err != nil || len(vec) == 0 {
			return fmt.Errorf("GetEmbeddingsSeq(seq=%d) failed: %w", seqID, err)
		}
		w.writeNormalized(seqID, vec)
	}
	return nil
}

// EmbedBucket embeds all non-padding sentences in bk (used when -bucket > 0).
func (w *llamaWorker) EmbedBucket(bk *bucket.Bucket) (int64, error) {
	batchSize := bk.Shape.BatchSize
	seqLen := bk.Shape.SentenceLength

	_ = w.batch.Clear()
	seqCount := 0
	var processedTokens int64
	var seqIDBuf [1]llama.SeqId
	seqIDSlice := seqIDBuf[:]

	for rowIdx := range batchSize {
		if bk.References[rowIdx] == nil {
			continue
		}
		row := bk.Batch[rowIdx*seqLen : (rowIdx+1)*seqLen]
		tokLen := len(row)
		for tokLen > 0 && row[tokLen-1] == w.padID {
			tokLen--
		}
		if tokLen == 0 {
			continue
		}
		if w.maxSeqTokens > 0 && tokLen > w.maxSeqTokens {
			tokLen = w.maxSeqTokens
		}

		if seqCount >= w.maxSeqs || int(w.batch.NTokens)+tokLen > w.maxBatchTokens {
			if err := w.flushSubBatch(seqCount); err != nil {
				return processedTokens, err
			}
			_ = w.batch.Clear()
			seqCount = 0
		}

		seqIDBuf[0] = llama.SeqId(seqCount)
		for pos := range tokLen {
			if err := w.batch.Add(llama.Token(row[pos]), llama.Pos(pos), seqIDSlice, true); err != nil {
				return processedTokens, err
			}
		}
		processedTokens += int64(tokLen)
		seqCount++
	}

	if seqCount > 0 {
		if err := w.flushSubBatch(seqCount); err != nil {
			return processedTokens, err
		}
	}
	return processedTokens, nil
}

func ensureLlamaLoaded(libDir, processor string, verbose bool) {
	if err := os.MkdirAll(libDir, 0755); err != nil {
		klog.Fatalf("Failed to create lib dir %q: %v", libDir, err)
	}
	if !download.AlreadyInstalled(libDir) {
		fmt.Fprintf(os.Stderr, "Downloading llama.cpp (%s, %s) for %s/%s into %s...\n",
			download.DefaultVersion, processor, runtime.GOOS, runtime.GOARCH, libDir)
		if err := download.Get(runtime.GOARCH, runtime.GOOS, processor, download.DefaultVersion, libDir); err != nil {
			klog.Fatalf("download.Get failed: %v", err)
		}
	}
	if err := llama.Load(libDir); err != nil {
		klog.Fatalf("llama.Load(%q) failed: %v", libDir, err)
	}
	if !verbose {
		llama.LogSet(llama.LogSilent())
	}
	llama.Init()
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	numEmbedders := *flagParallelEmbedders
	if numEmbedders <= 0 {
		if *flagBucketBudget <= 0 {
			numEmbedders = runtime.GOMAXPROCS(0)
		} else {
			numEmbedders = 1
		}
	}

	fmt.Printf("- Model %q, bucket %d, parallel embedders %d, dynamic shapes %t, warmup %t, shapes %t\n",
		*flagModel, *flagBucketBudget, numEmbedders, *flagDynamic, *flagWarmup, *flagShapes)

	if *flagCPUProf != "" {
		f, err := os.Create(*flagCPUProf)
		if err != nil {
			klog.Fatalf("Could not create CPU profile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			klog.Fatalf("Could not start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	if *flagProfile {
		go func() {
			klog.Info("Starting pprof server on :6060")
			klog.Fatal(http.ListenAndServe("localhost:6060", nil))
		}()
	}

	resolved := resolveModelAndTokenizer(*flagModel, *flagGGUFFile)

	mustRunWithElapsedTime("Initializing llama.cpp runtime", func() (any, error) {
		ensureLlamaLoaded(*flagLibDir, *flagProcessor, *flagVerbose)
		return nil, nil
	})
	defer llama.Close()

	llamaModel := mustRunWithElapsedTime(fmt.Sprintf("Loading GGUF model into llama.cpp (%s)", resolved.displayFile), func() (llama.Model, error) {
		mparams := llama.ModelDefaultParams()
		mparams.NGpuLayers = int32(*flagNGL)
		m, err := llama.ModelLoadFromFile(resolved.ggufPath, mparams)
		if err != nil || m == 0 {
			return 0, fmt.Errorf("ModelLoadFromFile(%q): %w", resolved.ggufPath, err)
		}
		return m, nil
	})
	defer llama.ModelFree(llamaModel)

	padID := 0
	if id, err := resolved.tokenizer.SpecialTokenID(tapi.TokPad); err == nil {
		padID = id
	}

	if *flagOutRef != "" {
		exportReferenceEmbeddings(llamaModel, resolved, padID, *flagOutRef)
		return
	}

	workers := mustRunWithElapsedTime(fmt.Sprintf("Creating %d llama.cpp embedder contexts", numEmbedders), func() ([]*llamaWorker, error) {
		ws := make([]*llamaWorker, numEmbedders)
		for i := range ws {
			w, err := newLlamaWorker(llamaModel, *flagBucketBudget, numEmbedders, *flagThreads, resolved.poolingType, resolved.normalize, padID)
			if err != nil {
				return nil, err
			}
			ws[i] = w
		}
		return ws, nil
	})
	defer func() {
		for _, w := range workers {
			w.Close()
		}
	}()
	fmt.Printf("- Backend: %s\n", workers[0].description())

	// Dataset preparation and stats.
	ds := datasets.New(msmarco.ID)
	limit := *flagLimit
	dsInfo, err := ds.Info()
	if err != nil {
		klog.Fatalf("Failed to get dataset info: %v", err)
	}
	if !mapHas(dsInfo.DatasetInfo, msmarco.Config) || !mapHas(dsInfo.DatasetInfo[msmarco.Config].Splits, *flagMSMarcoSplit) {
		klog.Fatalf("Dataset %q doesn't have contents for config=%q / split=%q", ds.ID, msmarco.Config, *flagMSMarcoSplit)
	}
	splitInfo := dsInfo.DatasetInfo[msmarco.Config].Splits[*flagMSMarcoSplit]
	totalQueries := splitInfo.NumExamples
	fmt.Printf("- Dataset %q, split %q: %d queries in total\n", ds.ID, *flagMSMarcoSplit, totalQueries)
	if limit > 0 {
		limit = min(limit, int(totalQueries))
	}

	if *flagWarmup {
		fmt.Printf("\n--- Warmup Run ---\n")
		if *flagBucketBudget <= 0 {
			runSingleSeqBenchmark(workers, resolved.tokenizer, ds, limit, int(totalQueries), true)
		} else {
			runBucketBenchmark(workers, resolved.tokenizer, ds, limit, int(totalQueries), true)
		}
	}

	fmt.Printf("\n--- Benchmark Run ---\n")
	if *flagBucketBudget <= 0 {
		runSingleSeqBenchmark(workers, resolved.tokenizer, ds, limit, int(totalQueries), false)
	} else {
		runBucketBenchmark(workers, resolved.tokenizer, ds, limit, int(totalQueries), false)
	}
}

func exportReferenceEmbeddings(llamaModel llama.Model, resolved *resolvedModel, padID int, outPath string) {
	w, err := newLlamaWorker(llamaModel, 0, 1, *flagThreads, resolved.poolingType, resolved.normalize, padID)
	if err != nil {
		klog.Fatalf("Failed to create worker: %v", err)
	}
	defer w.Close()

	var outBuf strings.Builder
	for i, p := range referencePrompts {
		toks := resolved.tokenizer.Encode(p)
		fmt.Printf("Prompt %d tokens (%d): %v\n", i, len(toks), toks)
		if _, err := w.EmbedTokens(toks); err != nil {
			klog.Fatalf("EmbedTokens prompt %d failed: %v", i, err)
		}
		outBuf.WriteString(fmt.Sprintf("# %s\n", p))
		for _, v := range w.outBuf[:w.nEmbd] {
			outBuf.WriteString(fmt.Sprintf("%.9g\n", v))
		}
	}
	if err := os.WriteFile(outPath, []byte(outBuf.String()), 0644); err != nil {
		klog.Fatalf("Failed writing %q: %v", outPath, err)
	}
	fmt.Printf("Wrote %d x %d embeddings to %s\n", len(referencePrompts), w.nEmbd, outPath)
}

type shapeKey struct {
	batchSize, seqLen int
}

// SpeedStats holds counters for the benchmark speed report.
type SpeedStats struct {
	NumSentencesProcessed    int64
	NumTokensProcessed       int64
	NumNonPadTokensProcessed int64
	Elapsed                  time.Duration
	CPUElapsed               time.Duration
	MaxRSSBytes              int64
	IsWarmup                 bool
	UniqueShapes             int
}

// runSingleSeqBenchmark streams MS MARCO passages without bucketing or padding (-bucket=0),
// executing one sequence per llama.cpp call across len(workers) parallel contexts.
func runSingleSeqBenchmark(workers []*llamaWorker, tokenizer tokenizers.Tokenizer, ds *datasets.Dataset, limit, totalQueries int, isWarmup bool) {
	var wg sync.WaitGroup
	passagesChan := make(chan string, len(workers)*8)

	wg.Go(func() {
		defer close(passagesChan)

		var numQueriesRead int
		for record, err := range datasets.IterParquetFromDatasetAt[msmarco.MsMarcoRecord](ds, msmarco.Config, *flagMSMarcoSplit, 0) {
			if err != nil {
				klog.Fatalf("Dataset iterator error: %v", err)
			}
			numQueriesRead++
			for _, text := range record.Passages.PassageText {
				if text == "" {
					continue
				}
				passagesChan <- text
			}
			if limit > 0 && numQueriesRead >= limit {
				break
			}
		}
	})

	startTime, startCPUTime := time.Now(), cpuTime()
	var speedStats SpeedStats
	speedStats.IsWarmup = isWarmup

	expectedNumQueries := limit
	if expectedNumQueries <= 0 {
		expectedNumQueries = totalQueries
	}
	expectedTotalSentences := int64(10 * expectedNumQueries)

	fmt.Printf("- Starting processing:\n")
	doneProgressChan := make(chan struct{})

	wg.Go(func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()

		var prevTokens int64
		prevTime := startTime
		var emaSpeed float64
		var emaInitialized bool

		updateDisplay := func(final bool) {
			now := time.Now()
			processed := atomic.LoadInt64(&speedStats.NumSentencesProcessed)
			nonPadTokens := atomic.LoadInt64(&speedStats.NumNonPadTokensProcessed)
			elapsed := now.Sub(startTime).Seconds()
			var sentencesPerSecond float64
			if elapsed > 0 {
				sentencesPerSecond = float64(processed) / elapsed
			}

			dt := now.Sub(prevTime).Seconds()
			if dt > 0 {
				instTokSpeed := float64(nonPadTokens-prevTokens) / dt
				if !emaInitialized {
					emaSpeed = instTokSpeed
					emaInitialized = true
				} else {
					emaSpeed = 0.3*instTokSpeed + 0.7*emaSpeed
				}
				prevTokens = nonPadTokens
				prevTime = now
			}

			if final {
				var avgTokSpeed float64
				if elapsed > 0 {
					avgTokSpeed = float64(nonPadTokens) / elapsed
				}
				fmt.Printf("\r  ✅ Processed %s / %s passages (%s, %s non-padding) -- done.%s\n",
					humanize.Count(processed), humanize.Count(expectedTotalSentences),
					humanize.Speed(sentencesPerSecond, "passages"),
					humanize.Speed(avgTokSpeed, "tokens"), humanize.EraseToEndOfLine)
				return
			}

			eta := "Unknown"
			if processed > 0 && sentencesPerSecond > 0 {
				remainingSeconds := float64(expectedTotalSentences-processed) / sentencesPerSecond
				if remainingSeconds > 0 {
					eta = humanize.Duration(time.Duration(int64(remainingSeconds*1e9)) * time.Nanosecond)
				} else {
					eta = "done"
				}
			}
			fmt.Printf("\r   - Processed %s / %s passages (%s, %s non-padding) -- ETA %s ...%s",
				humanize.Count(processed), humanize.Count(expectedTotalSentences),
				humanize.Speed(sentencesPerSecond, "passages"),
				humanize.Speed(emaSpeed, "tokens"), eta, humanize.EraseToEndOfLine)
		}

		for {
			select {
			case <-ticker.C:
				updateDisplay(false)
			case <-doneProgressChan:
				updateDisplay(true)
				return
			}
		}
	})

	var seenLengths sync.Map
	var embeddersWg sync.WaitGroup
	for _, worker := range workers {
		w := worker
		embeddersWg.Go(func() {
			warmedWorker := false
			for text := range passagesChan {
				ids := tokenizer.Encode(text)
				tokLen := min(len(ids), w.maxBatchTokens)

				if isWarmup {
					_, alreadySeenLen := seenLengths.LoadOrStore(tokLen, true)
					if warmedWorker && alreadySeenLen {
						atomic.AddInt64(&speedStats.NumSentencesProcessed, 1)
						continue
					}
					if *flagShapes && !alreadySeenLen {
						fmt.Printf("\n  - Shape [1, %d], %d tokens\n", tokLen, tokLen)
					}
				}

				processedTokens, err := w.EmbedTokens(ids)
				if err != nil {
					fmt.Println()
					klog.Fatalf("Failed to execute llama.cpp single-sequence embedding (len=%d): %+v", len(ids), err)
				}
				warmedWorker = true

				atomic.AddInt64(&speedStats.NumTokensProcessed, int64(processedTokens))
				atomic.AddInt64(&speedStats.NumNonPadTokensProcessed, int64(processedTokens))
				atomic.AddInt64(&speedStats.NumSentencesProcessed, 1)
			}
		})
	}

	embeddersWg.Wait()
	close(doneProgressChan)
	wg.Wait()

	speedStats.Elapsed = time.Since(startTime)
	speedStats.CPUElapsed = cpuTime() - startCPUTime
	speedStats.MaxRSSBytes = maxRSSBytes()
	if isWarmup {
		speedStats.UniqueShapes = mapLen(&seenLengths)
	}
	fmt.Printf("Total duration: %v\n", humanize.Duration(speedStats.Elapsed))
	ReportSpeed(&speedStats)
}

// runBucketBenchmark uses the two-bit token-budget Bucketizer (-bucket > 0).
func runBucketBenchmark(workers []*llamaWorker, tokenizer tokenizers.Tokenizer, ds *datasets.Dataset, limit, totalQueries int, isWarmup bool) {
	var wg sync.WaitGroup

	bucketsInputChan := make(chan bucket.SentenceRef, 5)
	bucketsOutputChan := make(chan bucket.Bucket, 10)
	bkt := bucket.New(tokenizer).
		ByTwoBitBucketBudget(*flagBucketBudget, 8).
		WithMaxParallelization(-1).
		WithBatchPadding(true)

	wg.Go(func() {
		bkt.Run(bucketsInputChan, bucketsOutputChan)
	})

	var numSentencesRead int32

	wg.Go(func() {
		defer close(bucketsInputChan)

		var numQueriesRead int
		for record, err := range datasets.IterParquetFromDatasetAt[msmarco.MsMarcoRecord](ds, msmarco.Config, *flagMSMarcoSplit, 0) {
			if err != nil {
				klog.Fatalf("Dataset iterator error: %v", err)
			}

			numQueriesRead++

			pLens := len(record.Passages.PassageText)
			for queryPassageIdx := range pLens {
				text := record.Passages.PassageText[queryPassageIdx]
				if text == "" {
					continue
				}

				bucketsInputChan <- bucket.SentenceRef{
					Sentence:  text,
					Reference: atomic.AddInt32(&numSentencesRead, 1),
				}
			}

			if limit > 0 && numQueriesRead >= limit {
				break
			}
		}
	})

	startTime, startCPUTime := time.Now(), cpuTime()
	var speedStats SpeedStats
	speedStats.IsWarmup = isWarmup

	expectedNumQueries := limit
	if expectedNumQueries <= 0 {
		expectedNumQueries = totalQueries
	}
	var emaSpeed float64
	var emaInitialized bool

	fmt.Printf("- Starting processing:\n")
	numSentencesProcessedChan := make(chan int)
	var emaMu sync.RWMutex

	wg.Go(func() {
		lastReportTime := time.Now()
		var sentencesPerSecond float64
		for count := range numSentencesProcessedChan {
			speedStats.NumSentencesProcessed += int64(count)

			if time.Since(lastReportTime) > time.Second {
				lastReportTime = time.Now()

				sentencesPerSecond = float64(speedStats.NumSentencesProcessed) / time.Since(startTime).Seconds()
				eta := "Unknown"
				expectedTotalSentences := int64(10 * expectedNumQueries)
				if speedStats.NumSentencesProcessed > 0 {
					remainingSeconds := float64(expectedTotalSentences-speedStats.NumSentencesProcessed) / sentencesPerSecond
					if remainingSeconds > 0 {
						eta = humanize.Duration(time.Duration(int64(remainingSeconds*1e9)) * time.Nanosecond)
					} else {
						eta = "done"
					}
				}
				emaMu.RLock()
				speed := emaSpeed
				emaMu.RUnlock()
				fmt.Printf("\r   - Processed %s / %s passages (%s, %s non-padding) -- ETA %s ...%s",
					humanize.Count(speedStats.NumSentencesProcessed), humanize.Count(expectedTotalSentences), humanize.Speed(sentencesPerSecond, "passages"),
					humanize.Speed(speed, "tokens"), eta, humanize.EraseToEndOfLine)
			}
		}
		expectedTotalSentences := int64(10 * expectedNumQueries)
		emaMu.RLock()
		speed := emaSpeed
		emaMu.RUnlock()
		fmt.Printf("\r  ✅ Processed %s / %s passages (%s, %s non-padding) -- done.%s\n",
			humanize.Count(speedStats.NumSentencesProcessed), humanize.Count(expectedTotalSentences), humanize.Speed(sentencesPerSecond, "passages"),
			humanize.Speed(speed, "tokens"), humanize.EraseToEndOfLine)
	})

	var seenShapes sync.Map
	var embeddersWg sync.WaitGroup
	for _, worker := range workers {
		w := worker
		embeddersWg.Go(func() {
			for bk := range bucketsOutputChan {
				if bk.Error != nil {
					klog.Fatalf("Tokenization error: %v", bk.Error)
				}

				batchSize := bk.Shape.BatchSize
				seqLen := bk.Shape.SentenceLength

				if isWarmup {
					key := shapeKey{batchSize, seqLen}
					if _, loaded := seenShapes.LoadOrStore(key, true); loaded {
						count := 0
						for _, ref := range bk.References {
							if ref != nil {
								count++
							}
						}
						numSentencesProcessedChan <- count
						continue
					}
					if *flagShapes {
						fmt.Printf("\n  - Shape [%d, %d], %d total tokens\n", batchSize, seqLen, len(bk.Batch))
					}
				}

				batchStartTime := time.Now()
				processedTokens, err := w.EmbedBucket(&bk)
				if err != nil {
					fmt.Println()
					klog.Fatalf("Failed to execute llama.cpp embeddings for [%d, %d]: %+v", batchSize, seqLen, err)
				}

				atomic.AddInt64(&speedStats.NumTokensProcessed, processedTokens)
				atomic.AddInt64(&speedStats.NumNonPadTokensProcessed, int64(bk.NonPadTokens))

				batchDuration := time.Since(batchStartTime).Seconds()
				if batchDuration > 0 {
					currentSpeed := float64(bk.NonPadTokens) / batchDuration
					emaMu.Lock()
					if !emaInitialized {
						emaSpeed = currentSpeed
						emaInitialized = true
					} else {
						emaSpeed = 0.1*currentSpeed + 0.9*emaSpeed
					}
					emaMu.Unlock()
				}

				count := 0
				for _, ref := range bk.References {
					if ref != nil {
						count++
					}
				}
				numSentencesProcessedChan <- count
			}
		})
	}

	wg.Go(func() {
		embeddersWg.Wait()
		close(numSentencesProcessedChan)
	})

	wg.Wait()
	speedStats.Elapsed = time.Since(startTime)
	speedStats.CPUElapsed = cpuTime() - startCPUTime
	speedStats.MaxRSSBytes = maxRSSBytes()
	if isWarmup {
		speedStats.UniqueShapes = mapLen(&seenShapes)
	}
	fmt.Printf("Total duration: %v\n", humanize.Duration(speedStats.Elapsed))
	ReportSpeed(&speedStats)
}

// ReportSpeed prints a summary table with counts and processing speeds (matching benchmark_embed).
func ReportSpeed(s *SpeedStats) {
	cpuTimePerPassage := "N/A"
	if s.NumSentencesProcessed > 0 {
		cpuTimePerPassage = fmt.Sprintf("%s/passage", humanize.Duration(s.CPUElapsed/time.Duration(s.NumSentencesProcessed)))
	}
	names := []string{"Passages", "Tokens", "Non-Pad Tokens", "CPU Time", "~Cores", "Max RSS Mem"}
	totals := []string{
		humanize.Count(s.NumSentencesProcessed),
		humanize.Count(s.NumTokensProcessed),
		humanize.Count(s.NumNonPadTokensProcessed),
		humanize.Duration(s.CPUElapsed),
		fmt.Sprintf("%.1f", s.CPUElapsed.Seconds()/s.Elapsed.Seconds()),
		humanize.Bytes(s.MaxRSSBytes),
	}
	speeds := []string{
		humanize.Speed(float64(s.NumSentencesProcessed)/s.Elapsed.Seconds(), "passages"),
		humanize.Speed(float64(s.NumTokensProcessed)/s.Elapsed.Seconds(), "tokens"),
		humanize.Speed(float64(s.NumNonPadTokensProcessed)/s.Elapsed.Seconds(), "tokens"),
		cpuTimePerPassage,
		"N/A",
		"N/A",
	}
	if s.IsWarmup {
		names = append(names, "Unique Shapes")
		totals = append(totals, humanize.Count(s.UniqueShapes))
		speeds = append(speeds, "N/A")
	}
	baseStyle := lipgloss.NewStyle().Padding(0, 1)
	t := table.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("238"))).
		Headers("Metric", "Total", "Speed").
		StyleFunc(func(row, col int) lipgloss.Style {
			st := baseStyle
			if col > 0 && row != table.HeaderRow {
				st = st.Align(lipgloss.Right)
			}
			if row == table.HeaderRow {
				headerStyle := st.Foreground(lipgloss.Color("252")).Bold(true)
				if col > 0 {
					headerStyle = headerStyle.Align(lipgloss.Center)
				}
				return headerStyle
			}
			return st
		})
	for i := range names {
		t.Row(names[i], totals[i], speeds[i])
	}
	fmt.Println(t)
}

func cpuTime() time.Duration {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

func maxRSSBytes() int64 {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	if runtime.GOOS == "darwin" {
		return int64(ru.Maxrss) // bytes on macOS
	}
	return int64(ru.Maxrss) * 1024 // KiB on Linux/BSD
}

func mustRunWithElapsedTime[T any](name string, f func() (T, error)) T {
	fmt.Printf("%s...", name)
	start := time.Now()
	ret, err := f()
	if err != nil {
		klog.Fatalf("failed: %v\n", err)
	}
	fmt.Printf("done (%s)\n", humanize.Duration(time.Since(start)))
	return ret
}

func mapLen(m *sync.Map) int {
	length := 0
	m.Range(func(_, _ any) bool {
		length++
		return true
	})
	return length
}
