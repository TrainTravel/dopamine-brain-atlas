"""Tests for the embedding step and the search files. Need ATLAS_MODEL_DIR (pinned model files) and
ATLAS_BUILD (output of build_data.py --model-dir); skipped when absent."""
import json, os, shutil
import numpy as np, pytest

import embed
from embed import Embedder, quantize, verify_model_dir, DIM

MODEL = os.environ.get("ATLAS_MODEL_DIR", "")
BUILD = os.environ.get("ATLAS_BUILD", "build")
NS = os.environ.get("NEUROSYNTH_DIR", "")
need_model = pytest.mark.skipif(not os.path.exists(os.path.join(MODEL, "model_quantized.onnx")), reason="model files not available")
need_build = pytest.mark.skipif(not os.path.exists(os.path.join(BUILD, "search", "meta.json")), reason="search build not present")


@pytest.fixture(scope="module")
def emb():
    return Embedder(MODEL)


@pytest.fixture(scope="module")
def S():
    m = json.load(open(os.path.join(BUILD, "search", "meta.json"), encoding="utf-8"))
    return m, np.fromfile(os.path.join(BUILD, "search", "emb.bin"), np.int8).reshape(m["n"], m["dim"]), \
        np.fromfile(os.path.join(BUILD, "search", "peaks.bin"), np.int8)


def test_quantize_roundtrip():
    rng = np.random.default_rng(1); E = rng.normal(size=(50, DIM)).astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
    q, scale = quantize(E)
    assert q.dtype == np.int8 and np.abs(q).max() == 127
    assert ((q * scale) * E).sum(1).min() > 0.995          # cosine with the original


@need_model
def test_tampered_or_missing_model_is_refused(tmp_path):
    for f in embed.MODEL_FILES: shutil.copy(os.path.join(MODEL, f), tmp_path / f)
    verify_model_dir(str(tmp_path))
    with open(tmp_path / "tokenizer.json", "ab") as f: f.write(b" ")
    with pytest.raises(ValueError, match="sha256"): verify_model_dir(str(tmp_path))
    os.remove(tmp_path / "tokenizer.json")
    with pytest.raises(FileNotFoundError): verify_model_dir(str(tmp_path))


@need_model
def test_no_padding_enters_the_embedding(emb):
    """Regression: the shipped tokenizer pads to a fixed length; padding once got averaged into every vector."""
    ids = emb.tokenize("mindful practices for ADHD kids")
    assert ids[0] == 101 and ids[-1] == 102 and 0 not in ids and len(ids) < 20
    assert len(emb.tokenize("word " * 500)) == embed.MAX_TOKENS
    assert emb.tokenize("") == [101, 102]


@need_model
def test_embeddings_are_unit_length_deterministic_and_meaningful(emb):
    T = ["kids", "children", "hippocampus", "staying awake all night", "sleep deprivation", "levodopa in Parkinson's disease"]
    E = emb.embed(T)
    assert E.shape == (6, DIM) and np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-5)
    assert np.array_equal(E, emb.embed(T))
    S_ = E @ E.T
    assert S_[0, 1] > 0.7 > S_[0, 2]                       # kids ~ children, not hippocampus
    assert S_[3, 4] > 0.45 > S_[3, 5]                      # paraphrase beats unrelated topic


@need_build
def test_search_files_are_consistent(S):
    m, E, peaks = S
    n = m["n"]
    assert n > 14000 and m["dim"] == DIM
    for k in ("ids", "titles", "years", "journalIdx", "authors", "peakCounts"): assert len(m[k]) == n, k
    assert len(set(m["ids"])) == n and all(isinstance(i, int) and i > 0 for i in m["ids"])
    assert all(isinstance(t, str) and t.strip() and t.lower() != "nan" for t in m["titles"])
    assert all(a and a.lower() != "nan" for a in m["authors"]) and all(0 <= j < len(m["journals"]) for j in m["journalIdx"])
    assert min(m["peakCounts"]) >= 1 and sum(m["peakCounts"]) * 3 == len(peaks)
    P3 = peaks.reshape(-1, 3)
    assert np.abs(P3[:, 0]).max() <= 80 and P3[:, 1].min() >= -115 and P3[:, 1].max() <= 85 and P3[:, 2].min() >= -70 and P3[:, 2].max() <= 90
    norms = np.linalg.norm(E * m["scale"], axis=1)
    assert norms.min() > 0.99 and norms.max() < 1.01
    json.load(open(os.path.join(BUILD, "search", "meta.json"), encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


@need_build
def test_atlas_studies_are_all_searchable(S):
    m = S[0]; pos = {p: i for i, p in enumerate(m["ids"])}
    D = json.load(open(os.path.join(BUILD, "data.json")))
    off = np.concatenate([[0], np.cumsum(m["peakCounts"])]) * 3
    for s in D["studies"]:
        i = pos[s[0]]                                      # KeyError here = an atlas study missing from search
        assert S[2][off[i]:off[i + 1]].tolist() == s[7]    # same peaks in both files


@need_build
@need_model
def test_stored_embeddings_match_a_fresh_embedding(S, emb):
    m, E, _ = S
    idx = list(range(0, m["n"], 479))
    fresh = emb.embed([m["titles"][i] for i in idx])
    assert ((E[idx] * m["scale"]) * fresh).sum(1).min() > 0.995


@need_build
def test_vocab_file_matches_the_tokenizer(S):
    v = json.load(open(os.path.join(BUILD, "model", "vocab.json"), encoding="utf-8"))
    assert len(v) == 30522 and len(set(v)) == len(v) and v[101] == "[CLS]" and v[102] == "[SEP]" and v[100] == "[UNK]" and v[0] == "[PAD]"
    if MODEL and os.path.exists(os.path.join(MODEL, "tokenizer.json")):
        ref = json.load(open(os.path.join(MODEL, "tokenizer.json"), encoding="utf-8"))["model"]["vocab"]
        assert all(ref[t] == i for i, t in enumerate(v))


@need_build
@pytest.mark.skipif(not NS or not os.path.isdir(NS), reason="raw Neurosynth files not available")
def test_search_peaks_trace_back_to_source(S):
    import pipeline as P
    m, _, peaks = S
    coords, meta, _, _ = P.load(NS); c = P.filter_bounds(coords); g = c.groupby("id")
    off = np.concatenate([[0], np.cumsum(m["peakCounts"])]) * 3
    assert m["n"] == c.id.nunique()
    for i in range(0, m["n"], 211):
        src = g.get_group(m["ids"][i])[["x", "y", "z"]].values.round().astype(int).flatten().tolist()
        assert peaks[off[i]:off[i + 1]].tolist() == src
        assert m["titles"][i] == P.clean_text(meta.title[meta.id == m["ids"][i]].iloc[0], "Untitled study")
