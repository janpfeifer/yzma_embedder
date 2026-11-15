package main

import "testing"

func TestSelectGGUFFileByQuant(t *testing.T) {
	miniLMFiles := []string{
		"all-MiniLM-L6-v2-Q2_K.gguf",
		"all-MiniLM-L6-v2-Q3_K_L.gguf",
		"all-MiniLM-L6-v2-Q3_K_M.gguf",
		"all-MiniLM-L6-v2-Q3_K_S.gguf",
		"all-MiniLM-L6-v2-Q4_0.gguf",
		"all-MiniLM-L6-v2-Q4_K_M.gguf",
		"all-MiniLM-L6-v2-Q4_K_S.gguf",
		"all-MiniLM-L6-v2-Q5_0.gguf",
		"all-MiniLM-L6-v2-Q5_K_M.gguf",
		"all-MiniLM-L6-v2-Q5_K_S.gguf",
		"all-MiniLM-L6-v2-Q6_K.gguf",
		"all-MiniLM-L6-v2-Q8_0.gguf",
		"all-MiniLM-L6-v2-ggml-model-f16.gguf",
	}

	cases := []struct {
		name    string
		files   []string
		quant   string
		want    string
		wantErr bool
	}{
		{
			name:  "Q4_0 exact",
			files: miniLMFiles,
			quant: "Q4_0",
			want:  "all-MiniLM-L6-v2-Q4_0.gguf",
		},
		{
			name:  "q4_0 lowercase",
			files: miniLMFiles,
			quant: "q4_0",
			want:  "all-MiniLM-L6-v2-Q4_0.gguf",
		},
		{
			name:  "Q4_0 with .gguf suffix",
			files: miniLMFiles,
			quant: "Q4_0.gguf",
			want:  "all-MiniLM-L6-v2-Q4_0.gguf",
		},
		{
			name:  "Q4_K_M",
			files: miniLMFiles,
			quant: "Q4_K_M",
			want:  "all-MiniLM-L6-v2-Q4_K_M.gguf",
		},
		{
			name:  "F16 case-insensitive",
			files: miniLMFiles,
			quant: "F16",
			want:  "all-MiniLM-L6-v2-ggml-model-f16.gguf",
		},
		{
			name:  "Q8_0",
			files: miniLMFiles,
			quant: "Q8_0",
			want:  "all-MiniLM-L6-v2-Q8_0.gguf",
		},
		{
			name:  "prefers non-mmproj",
			files: []string{"mmproj-Q4_0.gguf", "model-Q4_0.gguf"},
			quant: "Q4_0",
			want:  "model-Q4_0.gguf",
		},
		{
			name:  "subdirectory sharded GGUF picks first shard",
			files: []string{"Q4_0/model-00002-of-00002.gguf", "Q4_0/model-00001-of-00002.gguf", "Q8_0/model.gguf"},
			quant: "Q4_0",
			want:  "Q4_0/model-00001-of-00002.gguf",
		},
		{
			name:    "unknown quant returns error",
			files:   miniLMFiles,
			quant:   "IQ4_NL",
			wantErr: true,
		},
		{
			name:    "empty files returns error",
			files:   nil,
			quant:   "Q4_0",
			wantErr: true,
		},
		{
			name:    "empty quant returns error",
			files:   miniLMFiles,
			quant:   "   ",
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := selectGGUFFileByQuant(tc.files, tc.quant)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}
