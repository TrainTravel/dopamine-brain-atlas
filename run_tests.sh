#!/usr/bin/env bash
# Full check: unit tests -> build from raw data -> data and search tests -> browser tests on the built site.
# Usage: ./run_tests.sh [neurosynth_dir] [model_dir]   (defaults ./neurosynth and ./model-src, downloaded if absent)
set -euo pipefail
cd "$(dirname "$0")"
NS="${1:-neurosynth}"; MODEL="${2:-model-src}"
[ -d "$NS" ] || ./scripts/fetch_neurosynth.sh
[ -s "$MODEL/model_quantized.onnx" ] || ./scripts/fetch_model.sh "$MODEL"
[ -d node_modules/three ] && [ -d node_modules/onnxruntime-web ] || npm ci
export ATLAS_DATA_JSON=build/data.json ATLAS_BUILD=build ATLAS_MODEL_DIR="$MODEL" NEUROSYNTH_DIR="$NS" ATLAS_FIXTURES=cache/fixtures
echo "== 1/6 pipeline unit tests";        python3 -m pytest tests/test_pipeline.py -q
echo "== 2/6 build data, embeddings, site"; python3 build_data.py "$NS" build --model-dir "$MODEL" && python3 assemble.py build
echo "== 3/6 search benchmark + fixtures"; mkdir -p cache
python3 - "$NS" "$MODEL" <<'PY'
import json, sys, numpy as np
m = json.load(open("build/search/meta.json")); E = np.fromfile("build/search/emb.bin", np.int8).reshape(m["n"], m["dim"]) * m["scale"]
np.savez_compressed("cache/emb_all.npz", ids=np.array(m["ids"]), E=E.astype(np.float32))
PY
python3 tools/benchmark.py "$NS" "$MODEL" cache/emb_all.npz cache/benchmark.json > cache/benchmark.txt && sed -n 1,8p cache/benchmark.txt
python3 tools/make_search_fixture.py "$NS" "$MODEL" build cache/benchmark.json cache/fixtures
echo "== 4/6 data and search integration tests"; python3 -m pytest tests/test_data.py tests/test_search.py -q
echo "== 5/6 search unit tests (Node)";    node --test tests/search-core.test.js
echo "== 6/6 browser tests";               node --test tests/site.test.js
echo "ALL TESTS PASSED - deploy the dist/ folder"
