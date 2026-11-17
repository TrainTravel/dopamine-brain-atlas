"""Unit tests on synthetic data: every expected number below is worked out by hand."""
import json, math
import numpy as np, pandas as pd, pytest, scipy.sparse as sp
from scipy.stats import hypergeom

import pipeline as P
from nodes import NODES, PATHS
from assemble import js_safe

NODES_T = [("A", "Node A", (20, 0, 0), False, "target", [1, 2, 3, 0, 1]),
           ("M", "Midline", (0, 0, -30), True, "target", [0, 0, 0, 0, 0])]
PATHS_T = [("p", "Path", "desc", [("A", "M")])]
GROUPS_T = {"G1": ("Group one", ["alpha"]), "G2": ("Group two", ["beta", "gamma"])}
VOCAB = ["alpha", "beta", "gamma", "delta"]


@pytest.fixture
def tables():
    meta = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7],
        "title": ["One", "Two", "Three", "Four", "Five", "Six", float("nan")],
        "authors": ["Ab C, De F", "Solo A", "X Y, Z W", "Q R", "S T", "U V", float("nan")],
        "year": [2001, 2002, 2003, 2004, 2005, 2006, 2007],
        "journal": ["J : the long name", "K (Print)", "L", "M", "N", "O", float("nan")]})
    peaks = [(1, 20, 0, 0), (1, 50, 50, 50),        # exactly on A
             (2, -20, 0, 9),                        # 9 mm from mirrored A -> hit
             (3, 20, 0, 10.5),                      # 10.5 mm -> miss
             (4, 0, 0, -30), (4, -20, 10, 0),       # on M; exactly 10 mm from mirrored A -> hit (inclusive)
             (5, 20, 0, 0),                         # hits A but in no group
             (6, 200, 0, 0),                        # out of bounds -> study has no usable peak
             (7, 60, -60, 20), (7, float("nan"), 0, 0)]
    coords = pd.DataFrame(peaks, columns=["id", "x", "y", "z"])
    X = np.zeros((7, 4))
    X[[0, 1, 2, 5], 0] = 0.2            # alpha: studies 1,2,3,6
    X[[3, 5], 1] = 0.3                  # beta: studies 4,6
    X[4, 2] = 0.0005                    # gamma below threshold for study 5
    X[[4, 6], 3] = 0.5                  # delta (unused term)
    return coords, meta, VOCAB, sp.csc_matrix(X)


@pytest.fixture
def built(tables):
    return P.build_from_tables(*tables, groups=GROUPS_T, nodes=NODES_T, paths=PATHS_T)


def test_universe_excludes_studies_without_usable_peaks(built):
    assert built["total"] == 6                       # study 6 dropped


def test_group_counts(built):
    assert built["groups"]["G1"]["n"] == 3           # 1,2,3 (6 has no peaks)
    assert built["groups"]["G2"]["n"] == 1           # 4 only; 5 is below threshold


def test_node_base_rates(built):
    a, m = built["nodes"]
    assert a["base"] == 66.7                         # studies 1,2,4,5 of 6
    assert m["base"] == 16.7                         # study 4 of 6


def test_enrichment_values(built):
    a = built["nodes"][0]["enr"]
    assert (a["G1"]["hit"], a["G1"]["n"], a["G1"]["pct"], a["G1"]["ratio"]) == (2, 3, 66.7, 1.0)
    assert (a["G2"]["hit"], a["G2"]["n"], a["G2"]["pct"], a["G2"]["ratio"]) == (1, 1, 100.0, 1.5)


def test_selected_studies_and_bitmasks(built):
    s = {r[0]: r for r in built["studies"]}
    assert sorted(s) == [1, 2, 3, 4]                 # 5 and 7 in no group, 6 has no peaks
    assert s[1][5] == 0b01 and s[4][5] == 0b10       # group bits
    assert s[1][6] == 0b01 and s[2][6] == 0b01       # node A
    assert s[3][6] == 0                              # 10.5 mm away
    assert s[4][6] == 0b11                           # A (boundary) and M
    assert s[1][7] == [20, 0, 0, 50, 50, 50]


def test_mirroring_only_for_lateral_nodes():
    assert P.node_points((20, 0, 0), False) == [(20, 0, 0), (-20, 0, 0)]
    assert P.node_points((0, 0, -30), True) == [(0, 0, -30)]


def test_filter_bounds_drops_nan_and_outliers(tables):
    c = P.filter_bounds(tables[0])
    assert len(c) == 8 and not c[["x", "y", "z"]].isna().any().any()
    assert 6 not in set(c.id)


def test_select_mask_threshold_and_unknown_term(tables):
    _, _, vocab, X = tables
    assert P.select_mask(X, vocab, ["gamma"]).sum() == 0
    assert list(np.where(P.select_mask(X, vocab, ["beta", "gamma"]))[0]) == [3, 5]
    with pytest.raises(KeyError):
        P.select_mask(X, vocab, ["not-a-term"])


def test_enrichment_matches_independent_fisher():
    rng = np.random.default_rng(0)
    hit, grp, uni = rng.random(500) < 0.2, rng.random(500) < 0.3, np.ones(500, bool)
    e = P.enrichment(hit, grp, uni)
    a, n, K, N = e["hit"], e["n"], int(hit.sum()), 500
    probs = hypergeom.pmf(np.arange(0, n + 1), N, K, n)
    p = probs[probs <= probs[a] * (1 + 1e-7)].sum()   # two-sided Fisher, by definition
    assert e["p"] == pytest.approx(p, rel=0.02)
    assert e["ratio"] == pytest.approx((a / n) / (K / N), abs=0.006)


def test_enrichment_degenerate_cases_are_finite():
    z = np.zeros(10, bool); o = np.ones(10, bool)
    for hit, grp, uni in [(z, z, o), (z, o, o), (o, z, o), (o, o, o), (z, z, z)]:
        e = P.enrichment(hit, grp, uni)
        assert all(math.isfinite(v) for v in e.values())
        json.dumps(e, allow_nan=False)


def test_missing_metadata_is_cleaned(tables):
    coords, meta, vocab, X = tables
    r = P.study_record(meta.iloc[6], [[1, 2, 3]], 1, 0)
    assert r[1] == "Untitled study" and r[3] == "" and r[4] == "Unknown authors"
    assert "nan" not in json.dumps(r).lower()


def test_study_record_formatting(tables):
    meta = tables[1]
    r = P.study_record(meta.iloc[0], [[1.4, 2.6, -3.5]], 3, 5)
    assert r[3] == "J" and r[4] == "Ab C et al." and r[7] == [1, 3, -4] and r[5:7] == [3, 5]
    assert P.study_record(meta.iloc[1], [[0, 0, 0]], 0, 0)[3:5] == ["K", "Solo A"]
    long = meta.iloc[0].copy(); long["title"] = "word " * 60
    t = P.study_record(long, [[0, 0, 0]], 0, 0)[1]
    assert len(t) == 150 and t.endswith("...")


def test_output_is_strict_json(built):
    json.loads(json.dumps(built, allow_nan=False))


def test_mni305_to_152_transform():
    out = P.mni305_to_152(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    assert out[0] == pytest.approx([-0.0429, 1.5496, 1.1840])
    assert out[1] - out[0] == pytest.approx([9.975, 0.146, -0.130])


def test_too_many_nodes_rejected(tables):
    many = [(f"N{i}", "n", (i + 1, 0, 0), False, "target", [0] * 5) for i in range(31)]
    with pytest.raises(ValueError):
        P.build_from_tables(*tables, groups=GROUPS_T, nodes=many, paths=[])


def test_load_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        P.load(str(tmp_path))


# ---- the real region and pathway definitions ----
def test_node_definitions_are_valid():
    ids = [n[0] for n in NODES]
    assert len(ids) == len(set(ids)) == 15
    for nid, name, (x, y, z), mid, role, expr in NODES:
        assert role in {"source", "target", "relay"} and name
        assert len(expr) == 5 and all(e in (0, 1, 2, 3) for e in expr)
        assert (x == 0) if mid else (x > 0), f"{nid}: lateral nodes need x>0, midline x==0"
        assert -80 <= x <= 80 and -115 <= y <= 85 and -70 <= z <= 90


def test_paths_reference_known_nodes():
    ids = {n[0] for n in NODES}
    assert len({p[0] for p in PATHS}) == len(PATHS) == 6
    for pid, name, desc, edges in PATHS:
        assert edges and all(a in ids and b in ids and a != b for a, b in edges)


def test_js_safe_escapes_script_breakers():
    s = js_safe(json.dumps({"t": "</script><!--    "}, ensure_ascii=False))
    assert "</" not in s and "<!--" not in s and " " not in s and " " not in s
