"""Reference values the JavaScript search code is tested against.
Usage: python -I tools/make_search_fixture.py <ns_dir> <model_dir> <build_dir> <benchmark.json> <out_dir>
Writes search_eval.json (tokenizer ids, reference embeddings, benchmark queries with relevant studies)
and title_tokens.json (token ids for every study title).
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pipeline as P
from embed import Embedder
sys.argv_keep = sys.argv; ns, model_dir, build_dir, bench_path, out_dir = sys.argv[1:6]
import importlib.util
spec = importlib.util.spec_from_file_location("benchmark", os.path.join(ROOT, "tools", "benchmark.py")); bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)

meta = json.load(open(os.path.join(build_dir, "search", "meta.json"), encoding="utf-8"))
emb = Embedder(model_dir); idpos = {p: i for i, p in enumerate(meta["ids"])}
TRICKY = ["mindful practices for ADHD kids", "", "   ", "Parkinson's disease: L-DOPA & D2/D3 agonists (n=24)", "naïve café Ångström résumé", "ADHD", "fMRI",
          "résumé-naïve—dash – en dash", "γ-aminobutyric acid (GABA) 5-HT2A", "what's the 'default-mode' network?", "COVID-19 anosmia", "日本語のテキスト and 中文",
          "emoji 🧠 brain", "tabs\tand\nnewlines\r\nhere", "zero​width nbsp", "UPPERCASE lowercase MiXeD", "a" * 150, "supercalifragilisticexpialidocious pneumonoultramicroscopic",
          "<script>alert(1)</script>", "100% of p<0.05 results @ 3T", "İstanbul ǅ ß straße", "attention-deficit/hyperactivity disorder", "word " * 200,
          "children w/ autism vs. controls", "don't can't won't", "β-amyloid & τ (tau) PET", "…ellipsis “curly quotes” ‘single’", "x²+y³=z", "１２３ fullwidth ＡＢＣ"]
tok = [{"text": t, "ids": emb.tokenize(t)} for t in TRICKY]
PAR = ["mindful practices for ADHD kids", "staying awake all night", "levodopa in Parkinson's disease", "kids who struggle to read", "hearing voices in schizophrenia",
       "naïve café Ångström", "attention-deficit/hyperactivity disorder in children and adolescents", "x", "word " * 200,
       "Working memory-related functional brain patterns in never medicated children with ADHD.", "日本語 fMRI", "the bonding hormone given as a nose spray"]
PE = emb.embed(PAR)
bench = json.load(open(bench_path))
coords, m, vocab, X = P.load(ns)
rows = np.array([{i: k for k, i in enumerate(m.id.values)}[i] for i in meta["ids"]]); Xs = X.tocsc()[rows]
ev = []
for r in bench["queries"]:
    if r["set"] == "MeSH":
        pm = [int(x) for x in open(os.path.join(ROOT, "tests", "fixtures", "mesh", r["topic"] + ".txt")).read().split()]
        rel = sorted(idpos[p] for p in pm if p in idpos)
    else:
        groups = dict(bm.TERMS)[r["query"]]; mask = np.ones(len(rows), bool)
        for g in groups:
            gm = np.zeros(len(rows), bool)
            for t in [t for t in g.split("|") if t in vocab]: gm |= Xs[:, vocab.index(t)].toarray().ravel() > 0.001
            mask &= gm
        rel = np.where(mask)[0].tolist()
    assert len(rel) == r["n_relevant"], (r["query"], len(rel), r["n_relevant"])
    ev.append({"set": r["set"], "topic": r["topic"], "style": r["style"], "query": r["query"], "relevant": rel,
               "py": {k: round(r[k]["ndcg10"], 4) for k in ("keyword+synonyms", "semantic")}})
os.makedirs(out_dir, exist_ok=True)
json.dump({"n": meta["n"], "tokenizer": tok, "parity": [{"text": t, "vec": [round(float(x), 6) for x in v]} for t, v in zip(PAR, PE)], "eval": ev,
           "pySummary": {k: {m_: bench["summary"][k]["ndcg10"][m_] for m_ in ("keyword+synonyms", "semantic")} for k in ("MeSH: all 56", "Terms: all 32")}},
          open(os.path.join(out_dir, "search_eval.json"), "w"), separators=(",", ":"))
json.dump([emb.tokenize(t) for t in meta["titles"]], open(os.path.join(out_dir, "title_tokens.json"), "w"), separators=(",", ":"))
print("fixture written:", len(tok), "tokenizer cases,", len(PAR), "parity vectors,", len(ev), "benchmark queries,", meta["n"], "title token lists")
