"""Build the search files shipped with the site.

  search/meta.json   titles, PMIDs, years, journals, first authors, peak counts for every usable study
  search/peaks.bin   int8 x,y,z for every peak, concatenated in study order
  search/emb.bin     int8 title embeddings (n x 384), one shared scale stored in meta.json
  model/vocab.json   WordPiece vocabulary (token strings by id) for the browser tokenizer
"""
import json, os, shutil
import numpy as np
import pipeline as P
from embed import Embedder, quantize, DIM, MAX_TOKENS, MODEL_FILES


def corpus(coords, meta):
    """Usable studies (at least one in-bounds peak), in metadata order."""
    c = P.filter_bounds(coords)
    keep = np.isin(meta.id.values, c.id.values)
    return meta[keep].reset_index(drop=True), c


def first_author(a):
    a = P.clean_text(a)
    return (a.split(",")[0] + (" et al." if "," in a else "")) if a else "Unknown authors"


def build_search(coords, meta, model_dir, out_dir, embeddings=None):
    m, c = corpus(coords, meta)
    titles = [P.clean_text(t, "Untitled study") for t in m.title]
    by = c.groupby("id")
    peaks, counts = [], []
    for sid in m.id.values:
        p = by.get_group(sid)[["x", "y", "z"]].values.round().astype(int)
        if np.abs(p).max() > 127:
            raise ValueError(f"study {sid}: coordinate does not fit in int8")
        peaks.append(p.astype(np.int8)); counts.append(len(p))
    emb = Embedder(model_dir)
    E = embeddings if embeddings is not None else emb.embed(titles)
    if E.shape != (len(titles), DIM):
        raise ValueError(f"embedding matrix {E.shape} does not match {len(titles)} studies")
    q, scale = quantize(E)
    journals = [P.clean_text(j).split(" : ")[0].split(" (")[0] for j in m.journal]
    jlist = sorted(set(journals)); jpos = {j: i for i, j in enumerate(jlist)}
    os.makedirs(os.path.join(out_dir, "search"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "model"), exist_ok=True)
    meta_out = {"n": len(titles), "dim": DIM, "scale": scale, "maxTokens": MAX_TOKENS,
                "model": "Xenova/all-MiniLM-L6-v2 (int8 ONNX)", "modelSha256": MODEL_FILES["model_quantized.onnx"],
                "ids": [int(i) for i in m.id.values], "titles": titles,
                "years": [int(y) if not np.isnan(y) else 0 for y in m.year.astype(float)],
                "journals": jlist, "journalIdx": [jpos[j] for j in journals],
                "authors": [first_author(a) for a in m.authors], "peakCounts": counts}
    with open(os.path.join(out_dir, "search", "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    np.concatenate(peaks).tofile(os.path.join(out_dir, "search", "peaks.bin"))
    q.tofile(os.path.join(out_dir, "search", "emb.bin"))
    tok = json.load(open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8"))
    vocab = sorted(tok["model"]["vocab"].items(), key=lambda kv: kv[1])
    assert [i for _, i in vocab] == list(range(len(vocab))), "vocabulary ids are not contiguous"
    with open(os.path.join(out_dir, "model", "vocab.json"), "w", encoding="utf-8") as f:
        json.dump([t for t, _ in vocab], f, separators=(",", ":"), ensure_ascii=False)
    shutil.copy(os.path.join(model_dir, "model_quantized.onnx"), os.path.join(out_dir, "model", "model_quantized.onnx"))
    return meta_out, E
