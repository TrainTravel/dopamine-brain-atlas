#!/usr/bin/env bash
# Downloads the pinned embedding model into ./model-src (about 24 MB) and verifies its checksums.
# Source: Xenova/all-MiniLM-L6-v2 on Hugging Face, commit 751bff37 (Apache-2.0).
set -euo pipefail
dir="${1:-model-src}"; mkdir -p "$dir"
base=https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9
get() { [ -s "$dir/$2" ] || curl -fsSL -o "$dir/$2" "$base/$1"; }
get onnx/model_quantized.onnx model_quantized.onnx
get tokenizer.json tokenizer.json
( cd "$dir" && sha256sum -c - <<'SUMS'
afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1  model_quantized.onnx
da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0  tokenizer.json
SUMS
)
echo "Model ready in ./$dir"
