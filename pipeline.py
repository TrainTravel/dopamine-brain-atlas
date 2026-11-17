"""Dopamine Disorder Atlas data pipeline.

Pure functions (testable on synthetic data) plus build() which runs them on the
Neurosynth v7 files and returns the dict that becomes data.json.
"""
import base64, math, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact

from nodes import NODES, PATHS

TFIDF_THRESHOLD = 0.001   # Neurosynth's default term-frequency threshold
RADIUS_MM = 10.0
BOUNDS = {"x": (-80, 80), "y": (-115, 85), "z": (-70, 90)}  # generous MNI brain box
GROUPS = {
    "ASD": ("Autism", ["autism", "asd", "autism spectrum"]),
    "AD": ("Alzheimer's", ["alzheimer", "alzheimer disease", "disease ad"]),
    "PD": ("Parkinson's", ["parkinson", "parkinson disease", "disease pd"]),
    "ADHD": ("ADHD", ["adhd", "attention deficit", "deficit hyperactivity", "hyperactivity disorder"]),
    "DA": ("Dopamine studies", ["dopamine", "dopaminergic"]),
}
# FreeSurfer's published MNI305 -> MNI152 affine
MNI305_TO_152 = np.array([[0.9975, -0.0073, 0.0176, -0.0429],
                          [0.0146, 1.0009, -0.0024, 1.5496],
                          [-0.0130, -0.0093, 0.9971, 1.1840]])
FILES = {
    "coords": "data-neurosynth_version-7_coordinates.tsv.gz",
    "meta": "data-neurosynth_version-7_metadata.tsv.gz",
    "tfidf": "data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz",
    "vocab": "data-neurosynth_version-7_vocab-terms_vocabulary.txt",
}


def load(data_dir):
    for f in FILES.values():
        if not os.path.exists(os.path.join(data_dir, f)):
            raise FileNotFoundError(f"Missing Neurosynth file: {f} (expected in {data_dir})")
    coords = pd.read_csv(os.path.join(data_dir, FILES["coords"]), sep="\t")
    meta = pd.read_csv(os.path.join(data_dir, FILES["meta"]), sep="\t")
    vocab = [l.rstrip("\n") for l in open(os.path.join(data_dir, FILES["vocab"]), encoding="utf-8")]
    X = sp.load_npz(os.path.join(data_dir, FILES["tfidf"])).tocsc()
    if X.shape != (len(meta), len(vocab)):
        raise ValueError(f"tf-idf matrix {X.shape} does not match {len(meta)} studies x {len(vocab)} terms")
    return coords, meta, vocab, X


def filter_bounds(coords):
    """Drop peaks with missing values or outside the brain bounding box."""
    c = coords.dropna(subset=["x", "y", "z"])
    keep = np.ones(len(c), bool)
    for ax, (lo, hi) in BOUNDS.items():
        keep &= c[ax].between(lo, hi).values
    return c[keep]


def select_mask(X, vocab, terms, threshold=TFIDF_THRESHOLD):
    """Boolean mask of studies whose abstract uses any of `terms` above threshold."""
    mask = np.zeros(X.shape[0], bool)
    for t in terms:
        if t not in vocab:
            raise KeyError(f"term not in Neurosynth vocabulary: {t!r}")
        mask |= X[:, vocab.index(t)].toarray().ravel() > threshold
    return mask


def node_points(xyz, mid):
    x, y, z = xyz
    return [(x, y, z)] if mid else [(x, y, z), (-x, y, z)]


def node_hit_mask(study_ids, peak_xyz, peak_ids, xyz, mid, radius=RADIUS_MM, tree=None):
    """Studies with at least one peak within `radius` mm of the node (either hemisphere)."""
    tree = tree if tree is not None else cKDTree(peak_xyz)
    hit = set()
    for p in node_points(xyz, mid):
        hit.update(peak_ids[tree.query_ball_point(p, radius)])
    return np.isin(study_ids, list(hit))


def enrichment(hit, group, universe):
    """Hit rate of `group` vs all studies in `universe`, with a Fisher exact test
    of group vs the rest. Returns plain Python numbers (JSON safe, never NaN)."""
    g = group & universe
    rest = universe & ~group
    a, n = int((hit & g).sum()), int(g.sum())
    b, nb = int((hit & rest).sum()), int(rest.sum())
    base = float(hit[universe].mean()) if universe.any() else 0.0
    pct = 100 * a / n if n else 0.0
    ratio = (a / n) / base if n and base > 0 else 0.0
    p = float(fisher_exact([[a, n - a], [b, nb - b]])[1]) if n and nb else 1.0
    return {"hit": a, "n": n, "pct": round(pct, 1), "ratio": round(ratio, 2), "p": float(f"{p:.2g}")}


def clean_text(v, fallback=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return fallback
    s = " ".join(str(v).split())
    return s if s and s.lower() != "nan" else fallback


def study_record(row, peaks, group_bits, node_bits):
    title = clean_text(row.title, "Untitled study")
    if len(title) > 150:
        title = title[:147] + "..."
    journal = clean_text(row.journal).split(" : ")[0].split(" (")[0]
    authors = clean_text(row.authors)
    first = authors.split(",")[0] + (" et al." if "," in authors else "") if authors else "Unknown authors"
    year = int(row.year) if not pd.isna(row.year) else 0
    return [int(row.id), title, year, journal, first, int(group_bits), int(node_bits),
            np.asarray(peaks).round().astype(int).flatten().tolist()]


def mni305_to_152(V):
    return V @ MNI305_TO_152[:, :3].T + MNI305_TO_152[:, 3]


def pack_mesh():
    import nibabel as nib, nilearn
    p = os.path.join(os.path.dirname(nilearn.__file__), "datasets", "data", "fsaverage5")
    mesh = {}
    for h, side in (("L", "left"), ("R", "right")):
        g = nib.load(os.path.join(p, f"pial_{side}.gii.gz"))
        V, F = mni305_to_152(g.darrays[0].data), g.darrays[1].data
        s = nib.load(os.path.join(p, f"sulc_{side}.gii.gz")).darrays[0].data
        s = np.clip((s - s.min()) / (s.max() - s.min()), 0, 1)
        if len(V) > 65535:
            raise ValueError("mesh too large for 16-bit indices")
        mesh[h] = {"v": base64.b64encode(np.round(V * 10).astype("<i2").tobytes()).decode(),
                   "f": base64.b64encode(F.astype("<u2").tobytes()).decode(),
                   "s": base64.b64encode(np.round(s * 255).astype("u1").tobytes()).decode()}
    return mesh


def build_from_tables(coords, meta, vocab, X, groups=GROUPS, nodes=NODES, paths=PATHS, radius=RADIUS_MM):
    """Everything except the mesh. Works on synthetic tables for testing."""
    if len(nodes) > 30 or len(groups) > 30:
        raise ValueError("bitmask fields support at most 30 nodes/groups")
    c = filter_bounds(coords)
    ids = meta.id.values
    masks = {k: select_mask(X, vocab, t) for k, (_, t) in groups.items()}
    peak_xyz, peak_ids = c[["x", "y", "z"]].values, c.id.values
    tree = cKDTree(peak_xyz)
    universe = np.isin(ids, peak_ids)  # studies with at least one usable peak
    out_nodes, hits = [], []
    for nid, name, xyz, mid, role, expr in nodes:
        hm = node_hit_mask(ids, peak_xyz, peak_ids, xyz, mid, radius, tree)
        hits.append(hm)
        out_nodes.append({"id": nid, "name": name, "xyz": list(xyz), "mid": mid, "role": role, "expr": list(expr),
                          "base": round(100 * float(hm[universe].mean()), 1) if universe.any() else 0.0,
                          "enr": {k: enrichment(hm, masks[k], universe) for k in groups}})
    keys = list(groups)
    selected = np.zeros(len(meta), bool)
    for k in keys:
        selected |= masks[k]
    selected &= universe
    by_study = c.groupby("id")
    studies = []
    for i in np.where(selected)[0]:
        row = meta.iloc[i]
        gb = sum(1 << j for j, k in enumerate(keys) if masks[k][i])
        nb = sum(1 << j for j, hm in enumerate(hits) if hm[i])
        studies.append(study_record(row, by_study.get_group(row.id)[["x", "y", "z"]].values, gb, nb))
    return {"groups": {k: {"name": groups[k][0], "terms": groups[k][1], "n": int((masks[k] & universe).sum())} for k in keys},
            "keys": keys, "total": int(universe.sum()), "radius": radius, "nodes": out_nodes,
            "paths": [{"id": a, "name": b, "desc": d, "edges": [list(e) for e in edges]} for a, b, d, edges in paths],
            "studies": studies}


def build(data_dir):  # kept for callers that only need the atlas dict
    out = build_from_tables(*load(data_dir))
    out["mesh"] = pack_mesh()
    return out
