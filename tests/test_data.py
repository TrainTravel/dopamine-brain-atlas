"""Integration tests on the real built data.json (and raw Neurosynth files if present).
Set ATLAS_DATA_JSON / NEUROSYNTH_DIR to point at them."""
import base64, json, os
import numpy as np, pytest

from nodes import NODES, PATHS

PATH = os.environ.get("ATLAS_DATA_JSON", "data.json")
NS = os.environ.get("NEUROSYNTH_DIR", "")
pytestmark = pytest.mark.skipif(not os.path.exists(PATH), reason="data.json not built")


def _no_constants(c):
    raise ValueError(f"non-finite constant {c} in JSON")


@pytest.fixture(scope="module")
def D():
    return json.load(open(PATH, encoding="utf-8"), parse_constant=_no_constants)


def test_schema(D):
    assert set(D) == {"groups", "keys", "total", "radius", "nodes", "paths", "studies", "mesh"}
    assert D["keys"] == ["ASD", "AD", "PD", "ADHD", "DA"] and D["radius"] == 10.0
    assert [n["id"] for n in D["nodes"]] == [n[0] for n in NODES]
    assert [p["id"] for p in D["paths"]] == [p[0] for p in PATHS]
    for s in D["studies"]:
        assert len(s) == 8
        assert isinstance(s[0], int) and s[0] > 0
        assert isinstance(s[1], str) and s[1].strip() and len(s[1]) <= 150 and s[1].lower() != "nan"
        assert isinstance(s[2], int) and 1990 <= s[2] <= 2030
        assert isinstance(s[3], str) and isinstance(s[4], str) and s[4].lower() != "nan"
        assert 0 < s[5] < 2 ** len(D["keys"]) and 0 <= s[6] < 2 ** len(D["nodes"])
        assert len(s[7]) >= 3 and len(s[7]) % 3 == 0 and all(isinstance(v, int) for v in s[7])


def test_studies_unique(D):
    ids = [s[0] for s in D["studies"]]
    assert len(ids) == len(set(ids))


def test_peaks_inside_brain_box(D):
    xyz = np.concatenate([np.array(s[7]).reshape(-1, 3) for s in D["studies"]])
    assert np.abs(xyz[:, 0]).max() <= 80 and -115 <= xyz[:, 1].min() and xyz[:, 1].max() <= 85
    assert -70 <= xyz[:, 2].min() and xyz[:, 2].max() <= 90


def test_group_counts_match_study_bits(D):
    for j, k in enumerate(D["keys"]):
        assert D["groups"][k]["n"] == sum(1 for s in D["studies"] if s[5] >> j & 1)
        assert D["groups"][k]["n"] >= 100


def test_node_bits_recomputed_by_brute_force(D):
    """Independent check of the KD-tree search: plain distance arithmetic per study.
    Stored peaks are rounded to 1 mm, so allow a tolerance band around the radius."""
    R = D["radius"]
    for s in D["studies"]:
        xyz = np.array(s[7], float).reshape(-1, 3)
        for j, n in enumerate(D["nodes"]):
            x, y, z = n["xyz"]
            d = np.sqrt((np.abs(xyz[:, 0]) - x) ** 2 + (xyz[:, 1] - y) ** 2 + (xyz[:, 2] - z) ** 2).min() if not n["mid"] \
                else np.sqrt(((xyz - [x, y, z]) ** 2).sum(1)).min()
            bit = s[6] >> j & 1
            if d <= R - 0.9:
                assert bit, (s[0], n["id"], d)
            elif d > R + 0.9:
                assert not bit, (s[0], n["id"], d)


def test_enrichment_tables_consistent_with_studies(D):
    """hit counts come from the pipeline's masks; recount them from the per-study bits."""
    for j, n in enumerate(D["nodes"]):
        assert 0 <= n["base"] <= 100
        for gi, k in enumerate(D["keys"]):
            e = n["enr"][k]
            recount = sum(1 for s in D["studies"] if s[5] >> gi & 1 and s[6] >> j & 1)
            assert e["hit"] == recount, (n["id"], k)
            assert e["n"] == D["groups"][k]["n"] and 0 <= e["hit"] <= e["n"]
            assert e["pct"] == round(100 * e["hit"] / e["n"], 1)
            assert 0 <= e["p"] <= 1
            if n["base"] > 0:
                assert e["ratio"] == pytest.approx(e["pct"] / n["base"], abs=0.06)
            else:
                assert e["ratio"] == 0 and e["hit"] == 0


def test_known_neuroscience_is_recovered(D):
    """If these fail the pipeline is wrong, whatever else passes."""
    N = {n["id"]: n for n in D["nodes"]}
    for node, grp, min_ratio in [("PUT", "PD", 1.5), ("HIP", "AD", 1.7), ("AMY", "ASD", 1.4), ("VTA", "DA", 2.5), ("NAc", "DA", 2.0)]:
        e = N[node]["enr"][grp]
        assert e["ratio"] >= min_ratio and e["p"] < 0.001, (node, grp, e)
    assert N["PIT"]["base"] == 0            # outside the fMRI brain mask


def test_mesh_decodes_and_is_anatomically_placed(D):
    for h, sign in (("L", -1), ("R", 1)):
        m = D["mesh"][h]
        v = np.frombuffer(base64.b64decode(m["v"]), "<i2").reshape(-1, 3) / 10
        f = np.frombuffer(base64.b64decode(m["f"]), "<u2").reshape(-1, 3)
        s = np.frombuffer(base64.b64decode(m["s"]), "u1")
        assert v.shape == (10242, 3) and f.shape == (20480, 3) and s.shape == (10242,)
        assert f.max() == len(v) - 1 and len(np.unique(f)) == len(v)
        assert np.sign(v[:, 0].mean()) == sign
        assert np.abs(v[:, 0]).max() < 80 and -112 < v[:, 1].min() and v[:, 1].max() < 80
        assert -60 < v[:, 2].min() and v[:, 2].max() < 90
    # every lateral region centre must sit inside the cortical bounding box
    for n in D["nodes"]:
        assert abs(n["xyz"][0]) < 70 and -105 < n["xyz"][1] < 72


@pytest.fixture(scope="module")
def raw():
    import pipeline as P
    coords, meta, vocab, X = P.load(NS)
    return P, P.filter_bounds(coords), meta, vocab, X


@pytest.mark.skipif(not NS or not os.path.isdir(NS), reason="raw Neurosynth files not available")
class TestAgainstRawNeurosynth:
    def test_snapshot_counts(self, D, raw):
        assert D["total"] == 14360   # 14,371 papers, minus 11 with no peak inside the brain box
        assert {k: g["n"] for k, g in D["groups"].items()} == {"ASD": 255, "AD": 269, "PD": 180, "ADHD": 172, "DA": 375}
        assert len(D["studies"]) == 1145

    def test_every_selected_study_really_uses_its_terms(self, D, raw):
        P, c, meta, vocab, X = raw
        pos = {i: k for k, i in enumerate(meta.id.values)}
        for j, key in enumerate(D["keys"]):
            mask = P.select_mask(X, vocab, D["groups"][key]["terms"])
            for s in D["studies"]:
                assert bool(s[5] >> j & 1) == bool(mask[pos[s[0]]])

    def test_peaks_match_source_rows(self, D, raw):
        P, c, meta, vocab, X = raw
        g = c.groupby("id")
        for s in D["studies"][::25]:
            src = g.get_group(s[0])[["x", "y", "z"]].values.round().astype(int).flatten().tolist()
            assert src == s[7]

    def test_term_based_sanity_check(self, D, raw):
        """Studies about the amygdala must hit the amygdala node far above base rate."""
        P, c, meta, vocab, X = raw
        ids = meta.id.values
        amy = [n for n in NODES if n[0] == "AMY"][0]
        hit = P.node_hit_mask(ids, c[["x", "y", "z"]].values, c.id.values, amy[2], amy[3])
        uni = np.isin(ids, c.id.values)
        e = P.enrichment(hit, P.select_mask(X, vocab, ["amygdala"]), uni)
        assert e["pct"] > 55 and e["ratio"] > 3
        # negative control: an unrelated term must not be enriched there
        assert P.enrichment(hit, P.select_mask(X, vocab, ["reading"]), uni)["ratio"] < 1.0
