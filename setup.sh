#!/usr/bin/env bash
# One-shot setup for Recursive Flint (NVFP4 GGUF) on llama.cpp with the speed patch.
#
#   curl -fsSL https://raw.githubusercontent.com/pqhaz3925/flint-rtxpro/main/setup.sh | bash
#   # or: git clone https://github.com/pqhaz3925/flint-rtxpro && bash flint-rtxpro/setup.sh
#
# Tested on Ubuntu 24.04, RTX PRO 6000 Blackwell (sm_120), driver 580. Needs sudo for apt.
# Re-running is safe: finished steps are skipped.
#
# Environment overrides:
#   WORK=~/flint            where llama.cpp, the model and logs go
#   PATCH=1                 0 = stock llama.cpp (no speed patch)
#   START=1                 0 = build and download only, do not start the server
#   PORT=8080 HOST=127.0.0.1 CTX=262144
#   CUDA_ARCH=120a          default: detected from nvidia-smi
#   HF_TOKEN=...            optional, faster/limit-free downloads
#
# The whole script detaches itself (setsid + nohup) so a dropped SSH session does not kill
# the build. Follow it with: tail -f $WORK/setup.log
set -euo pipefail

WORK="${WORK:-$HOME/flint}"
PATCH="${PATCH:-1}"
START="${START:-1}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-262144}"

LLAMA_REPO="https://github.com/ggml-org/llama.cpp"
LLAMA_COMMIT="5f5acfe2504b490386cd03020202f59e36861add"   # master 2026-10-22, has Flint (a37cc46104)
MODEL_REPO="RecursiveSI/flint-nvfp4-gguf"
MODEL_FILE="flint-nvfp4.gguf"
RAW="https://raw.githubusercontent.com/pqhaz3925/flint-rtxpro/main"

mkdir -p "$WORK"
LOG="$WORK/setup.log"

# Detach once so an SSH drop cannot kill the long build/download.
if [[ -z "${FLINT_SETUP_DETACHED:-}" ]]; then
  SELF="$WORK/setup.sh"
  if [[ -f "${BASH_SOURCE[0]:-}" ]]; then
    cp -f "${BASH_SOURCE[0]}" "$SELF" 2>/dev/null || true
    SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  else
    curl -fsSL "$RAW/setup.sh" -o "$SELF"
    SRC_DIR=""
  fi
  export FLINT_SETUP_DETACHED=1 FLINT_SRC_DIR="$SRC_DIR"
  export WORK PATCH START PORT HOST CTX CUDA_ARCH="${CUDA_ARCH:-}" HF_TOKEN="${HF_TOKEN:-}"
  setsid nohup bash "$SELF" > "$LOG" 2>&1 < /dev/null &
  echo "setup running in background (pid $!). Follow: tail -f $LOG"
  if [[ "$START" == "1" ]]; then
    echo "Done when the log ends with: FLINT READY on http://$HOST:$PORT"
  else
    echo "Done when the log ends with: FLINT BUILT"
  fi
  exit 0
fi

step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step "packages"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache libssl-dev \
  python3-venv python3-pip curl wget ca-certificates >/dev/null

step "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
if [[ -z "${CUDA_ARCH:-}" ]]; then
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
  case "$cap" in
    90|100|101|103|110|120|121) CUDA_ARCH="${cap}a" ;;   # arch-specific features (FP4 MMA etc.)
    *) CUDA_ARCH="$cap" ;;
  esac
fi
echo "CUDA_ARCH=$CUDA_ARCH"

step "CUDA toolkit (need >= 12.8 for Blackwell)"
NVCC=""
for c in /usr/local/cuda-13.0/bin/nvcc /usr/local/cuda/bin/nvcc $(command -v nvcc || true); do
  [[ -x "$c" ]] || continue
  v=$("$c" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p')
  if (( v >= 128 )); then NVCC="$c"; break; fi
done
if [[ -z "$NVCC" ]]; then
  if ! apt-cache show cuda-nvcc-13-0 >/dev/null 2>&1; then
    . /etc/os-release
    distro="ubuntu${VERSION_ID//./}"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update -qq
  fi
  # CUDA 13.0 matches driver 580; only the parts the build needs.
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 \
    libcublas-dev-13-0 cuda-nvml-dev-13-0 cuda-cccl-13-0 >/dev/null
  NVCC=/usr/local/cuda-13.0/bin/nvcc
fi
"$NVCC" --version | tail -1
export PATH="$(dirname "$NVCC"):$PATH"

step "llama.cpp @ ${LLAMA_COMMIT:0:10}"
if [[ ! -d "$WORK/llama.cpp/.git" ]]; then
  git clone -q "$LLAMA_REPO" "$WORK/llama.cpp"
fi
cd "$WORK/llama.cpp"
git fetch -q origin "$LLAMA_COMMIT" 2>/dev/null || git fetch -q origin
git checkout -q -f "$LLAMA_COMMIT"
git checkout -q -- .
if [[ "$PATCH" == "1" ]]; then
  if [[ -n "${FLINT_SRC_DIR:-}" && -f "$FLINT_SRC_DIR/flint-speed.patch" ]]; then
    cp "$FLINT_SRC_DIR/flint-speed.patch" "$WORK/flint-speed.patch"
  else
    curl -fsSL "$RAW/flint-speed.patch" -o "$WORK/flint-speed.patch"
  fi
  git apply "$WORK/flint-speed.patch"
  echo "speed patch applied"
fi

step "build (about 5 min on 32+ cores)"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > "$WORK/cmake.log" 2>&1
cmake --build build -j "$(nproc)" --target llama-server > "$WORK/build.log" 2>&1 || { tail -30 "$WORK/build.log"; exit 1; }
ls -la build/bin/llama-server

step "model $MODEL_REPO (22.9 GB)"
if [[ ! -x "$WORK/venv/bin/hf" ]]; then
  python3 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q -U pip huggingface_hub hf_transfer
fi
if [[ ! -s "$WORK/model/$MODEL_FILE" ]] || (( $(stat -c %s "$WORK/model/$MODEL_FILE") < 22000000000 )); then
  HF_HUB_ENABLE_HF_TRANSFER=1 "$WORK/venv/bin/hf" download "$MODEL_REPO" "$MODEL_FILE" pi-models.json opencode.json \
    --local-dir "$WORK/model"
fi
ls -la "$WORK/model/$MODEL_FILE"

step "serve.sh"
cat > "$WORK/serve.sh" <<EOF
#!/bin/bash
# Flint NVFP4 on llama.cpp. Extra args are passed to llama-server (later flags win).
exec "$WORK/llama.cpp/build/bin/llama-server" -m "$WORK/model/$MODEL_FILE" \\
  --alias flint -ngl all -c $CTX -np 1 \\
  -fa on -ctk f16 -ctv f16 --jinja \\
  -b 2048 -ub 2048 \\
  --spec-type ngram-mod --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 64 --spec-ngram-mod-n-match 16 \\
  --temp 1 --top-p 0.95 --top-k 20 --min-p 0 \\
  --host $HOST --port $PORT "\$@"
EOF
chmod +x "$WORK/serve.sh"

if [[ "$START" == "1" ]]; then
  step "start server"
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "something already listens on :$PORT, not starting a second server"
  else
    setsid nohup "$WORK/serve.sh" > "$WORK/server.log" 2>&1 < /dev/null &
    echo $! > "$WORK/server.pid"
    for _ in $(seq 1 120); do
      curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
      kill -0 "$(cat "$WORK/server.pid")" 2>/dev/null || { tail -30 "$WORK/server.log"; exit 1; }
      sleep 2
    done
  fi
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"model":"flint","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":512}' \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print("smoke test:", repr(r["choices"][0]["message"]["content"]), "|", round(r["timings"]["predicted_per_second"]), "tok/s")'
  echo
  echo "FLINT READY on http://$HOST:$PORT  (OpenAI /v1/chat/completions, Anthropic /v1/messages; model name: flint)"
  echo "logs: $WORK/server.log   restart: kill \$(cat $WORK/server.pid); $WORK/serve.sh"
else
  echo "FLINT BUILT. Start with: $WORK/serve.sh"
fi
