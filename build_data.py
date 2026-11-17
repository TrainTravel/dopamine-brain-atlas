"""Usage: python build_data.py <neurosynth_dir> <out_dir> [--model-dir DIR]

Writes <out_dir>/data.json (the atlas) and, when --model-dir is given, the search files
(<out_dir>/search/*, <out_dir>/model/*). The model directory must hold the pinned
model_quantized.onnx and tokenizer.json (see scripts/fetch_model.sh).
"""
import argparse, json, os
import pipeline as P

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("neurosynth_dir"); ap.add_argument("out_dir"); ap.add_argument("--model-dir")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tables = P.load(a.neurosynth_dir)
    data = P.build_from_tables(*tables); data["mesh"] = P.pack_mesh()
    with open(os.path.join(a.out_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)  # strict JSON: NaN is an error
    print(f"data.json: {len(data['studies'])} studies, {sum(len(s[7]) // 3 for s in data['studies'])} peaks")
    if a.model_dir:
        from search_build import build_search
        meta, _ = build_search(tables[0], tables[1], a.model_dir, a.out_dir)
        print(f"search: {meta['n']} studies embedded, {sum(meta['peakCounts'])} peaks")
