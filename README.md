# Flint on one RTX PRO 6000, fast

One command to serve [Recursive Flint](https://huggingface.co/RecursiveSI/flint) (40B MoE, ~2.5B active, coding agent model) from its official [NVFP4 GGUF](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf) on llama.cpp, plus a small CUDA patch that makes decoding ~40% faster without changing the output.

```bash
git clone https://github.com/pqhaz3925/flint-rtxpro && bash flint-rtxpro/setup.sh
tail -f ~/flint/setup.log        # ends with "FLINT READY on http://127.0.0.1:8080"
```

The script installs CUDA 13 if needed, builds llama.cpp at a pinned commit with the patch, downloads the 22.9 GB model and starts the server. It detaches itself, so a dropped SSH session doesn't stop it. ~12 minutes on a 128-core box, most of it compiling. Re-running is safe.

Options (environment variables): `WORK=~/flint`, `PATCH=0` (stock llama.cpp), `START=0`, `PORT=8080`, `HOST=127.0.0.1`, `CTX=262144`, `CUDA_ARCH=120a`, `HF_TOKEN`.

## Numbers

Single stream, one RTX PRO 6000 Blackwell Server Edition (96 GB, driver 580, CUDA 13.0), full 262k context configured, 29.9 GB VRAM used.

| | stock llama.cpp | this repo |
|---|---|---|
| decode, short context | 192 tok/s | **267 tok/s** |
| decode, "rewrite this file" task (n-gram speculation) | ~245 tok/s | **450–540 tok/s** |
| prefill, 16k / 62k prompt¹ | 6.1k / 5.9k tok/s | **11.0k / 9.5k tok/s** |
| decode at 62k depth | not measured | 190 tok/s |

¹ Prefill gain comes from `-ub 2048` (default 512), measured on the patched build with both settings.

Greedy output of the patched build is byte-identical to stock (`bench/same_output.py`), and `test-backend-ops` passes for `MUL_MAT`, `MUL_MAT_ID` and `POOL_1D`.

## What the patch does

`flint-speed.patch` touches only the CUDA backend (5 files, +174 lines). Profiled with Nsight Systems: a decode step was ~2000 tiny kernels, latency-bound (GPU "100% busy" at 200 W, 29% memory bandwidth).

1. **amax rescale fusion.** For Flint (and Mistral 4 / Mellum) llama.cpp scales the input of the expert down-projection by a power of two derived from `max|x|` per row (`build_amax_src1_f16`). On CUDA that `max` ran through `pool_1d`, whose kernel gives each output to one thread that loops over the row: 6 threads on the whole GPU, 23 µs per MoE layer, ~20% of decode time. Followed by 8 more one-element kernels (`log`, `ceil`, `clamp`, `exp`, ...). The patch adds a warp-per-row path to `pool_1d` for whole-row windows and fuses the whole `abs → pool → scale → log → scale → ceil → clamp → scale → exp → div` chain into one kernel (same float ops in the same order, so results are bit-identical). The chain's `div` is allocated in place over its input, which the generic fusion overlap check rejects; the kernel reads each row fully before writing it, so exact aliasing is allowed explicitly. **192 → 257 tok/s.**
2. **Reuse quantized activations in MMVQ.** Every quantized mat-vec re-quantized its input to q8_1, even when consecutive mat-vecs share it (Q/K/V). The patch keeps the last q8_1 activations in a small per-device buffer and skips the quantize kernel when the next MMVQ call has the same `src1` tensor; the cache is invalidated at the start of every graph evaluation and only used on the main stream for decode-sized batches. **257 → 267 tok/s.**

Server flags on top (in `serve.sh`): `-b 2048 -ub 2048` (prefill +60–75%), `--spec-type ngram-mod --spec-ngram-mod-n-min 8` (no draft model; big win when the output repeats the context, no cost otherwise).

Tried and dropped: q8_0 KV cache (decode −15–20%), forcing fewer flash-attention splits, PDL launches for two more kernels, `ngram-cache` speculation (hurts plain text).

What's left: every mat-vec launch costs ≥3 µs even for a 0.6 MB matrix and there are 8 per layer, so the next step would be fusing Q/K/V into one launch and routed + shared experts into another (llama.cpp's existing shared-expert fusion doesn't fit Flint: shared expert is 1024 wide vs 512 per routed expert, and NVFP4 scale tensors). Probably +7–10%.

## Why not the BF16 checkpoint

The official SGLang kit only supports FA4 (sm_100, B200/B300) and FA3 (sm_90, H200) attention and asks for ≥141 GB. RTX PRO 6000 is sm_120 with 96 GB: the 80 GB of BF16 weights fit, but with ~10 GB left for KV cache and no supported attention backend. The NVFP4 GGUF is what Recursive tested on this card.

## Using it

The server listens on localhost. From your machine:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@gpu-box
```

Then `http://localhost:8080/v1` (OpenAI chat completions) or `http://localhost:8080/v1/messages` (Anthropic messages), model name `flint`. Reasoning comes back as `reasoning_content` / `thinking` blocks; send it back in the history. No API key unless you add `--api-key` to `serve.sh`.

Recommended sampling is baked into the server (temperature 1.0, top-p 0.95, top-k 20). Give hard tasks `max_tokens` ≥ 32768: reasoning counts toward it.

Agent configs from Recursive (pi, OpenCode) are downloaded next to the model in `~/flint/model/`.

## Benchmarks

```bash
cd bench
python3 decode.py 8080            # short-context decode
python3 agentic.py 8080           # free text vs. "edit this file"
python3 longctx.py 8080 220000    # prefill + decode at ~62k tokens
python3 same_output.py 8080 8081  # patched vs stock, greedy output must match
```

To compare against stock, build a second copy with `WORK=~/flint-stock PATCH=0 START=0 bash setup.sh` and start it with `~/flint-stock/serve.sh --port 8081`.
