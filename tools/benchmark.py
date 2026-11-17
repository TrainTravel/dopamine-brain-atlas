"""Search benchmark: keyword vs semantic vs hybrid ranking over all usable Neurosynth studies.

Every method sees ONLY study titles. Two independent relevance judges, neither visible to any method:
  A. MeSH   - PubMed's human-assigned subject headings (tests/fixtures/mesh/*.txt), 14 topics x 4 phrasings.
  B. Terms  - Neurosynth abstract tf-idf terms, 32 lay-language queries.
The hybrid's two constants (0.2, power 1) were chosen on set B, so set A is a held-out test of it.

Usage: python -I tools/benchmark.py <ns_dir> <model_dir> <emb_all.npz> [out.json]
"""
import json, math, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, snowballstemmer, pipeline as P
from embed import Embedder

STOP = set("a an the of in on for with and or to by from at as is are was were be this that how your their others other "
           "people when who what do does they them after before".split())
# General lay-to-technical synonyms, fixed before the MeSH benchmark was run.
SYN = {"kids": ["children", "child"], "teenagers": ["adolescents", "adolescent"], "teens": ["adolescents"],
       "old": ["older", "aging", "elderly"], "sad": ["depression", "depressed"], "smokers": ["smoking", "nicotine"],
       "worry": ["anxiety"], "scared": ["fear"], "hyperactivity": ["adhd"], "autistic": ["autism"],
       "forgetting": ["memory"], "drinking": ["alcohol"], "mindful": ["mindfulness", "meditation"], "mindfulness": ["meditation"]}

MESH = {  # topic: (lay, researcher, topic+population, bare keyword)
 "meditation": ("calming the mind by focusing on the breath", "mindfulness meditation training", "mindful practices for ADHD kids", "meditation"),
 "hypnosis": ("being put into a trance by suggestion", "hypnotic suggestion and analgesia", "hypnosis for pain in adults", "hypnosis"),
 "tinnitus": ("constant ringing in the ears", "tinnitus auditory cortex resting-state connectivity", "ringing in the ears in older patients", "tinnitus"),
 "sleep_deprivation": ("staying awake all night", "total sleep deprivation and working memory", "tired students after a night without sleep", "sleep deprivation"),
 "adhd_methylphenidate": ("stimulant pills for hyperactive kids", "methylphenidate effects in ADHD", "Ritalin for children with attention problems", "methylphenidate ADHD"),
 "stuttering": ("people who stammer when they speak", "developmental stuttering speech production", "children who stutter", "stuttering"),
 "gambling": ("betting money and losing", "pathological gambling reward processing", "gambling addiction in young men", "gambling"),
 "deception": ("telling lies", "deception detection prefrontal cortex", "lying to strangers for money", "deception"),
 "placebo_pain": ("sugar pills reducing pain", "placebo analgesia", "fake painkillers in healthy volunteers", "placebo pain"),
 "dyslexia_children": ("kids who struggle to read", "developmental dyslexia phonological processing in children", "reading difficulties in schoolchildren", "dyslexia children"),
 "parkinson_dopamine_drugs": ("medication for shaking and slow movement", "levodopa in Parkinson's disease", "dopamine drugs for elderly Parkinson's patients", "levodopa Parkinson"),
 "oxytocin": ("the bonding hormone given as a nose spray", "intranasal oxytocin social cognition", "oxytocin in men viewing faces", "oxytocin"),
 "anorexia": ("people who starve themselves to stay thin", "anorexia nervosa body image", "teenage girls with eating disorders", "anorexia"),
 "migraine": ("severe recurring headaches", "migraine without aura", "women with migraines", "migraine"),
}
STYLES = ("lay", "researcher", "topic+population", "keyword")
TERMS = [("kids with hyperactivity", ["adhd", "children"]), ("memory problems in old age", ["memory", "older adults|aging|elderly"]),
 ("looking at faces in autism", ["autism", "faces|face"]), ("tremor and trouble moving", ["parkinson", "motor|movement"]),
 ("feeling sad and hopeless", ["depression|depressive|mdd"]), ("being scared of threatening pictures", ["fear|threat", "pictures|images|stimuli"]),
 ("how the brain understands spoken sentences", ["speech|spoken", "sentences|sentence|comprehension"]), ("children learning to read", ["reading", "children"]),
 ("sugar pills reducing pain", ["placebo", "pain"]), ("urge to smoke cigarettes", ["smokers|smoking|nicotine", "craving"]),
 ("gambling and risky money choices", ["risk|risky", "monetary|gambling|decision"]), ("emotion in tone of voice", ["prosody|voice|vocal", "emotional|emotion"]),
 ("finding your way around a town", ["navigation"]), ("doing sums in your head", ["arithmetic|calculation"]), ("listening to songs", ["music"]),
 ("staying awake all night", ["sleep"]), ("hearing voices in schizophrenia", ["schizophrenia", "hallucinations|auditory"]),
 ("people who worry a lot", ["anxiety"]), ("resisting tempting food", ["food", "control|regulation|inhibition"]),
 ("speaking two languages", ["bilingual|bilinguals"]), ("feeling someone else's pain", ["empathy", "pain"]), ("judging right and wrong", ["moral"]),
 ("heavy drinking", ["alcohol"]), ("flashbacks after trauma", ["ptsd|trauma"]), ("repeated checking and handwashing", ["ocd|obsessive compulsive"]),
 ("regaining hand movement after a stroke", ["stroke", "motor|hand"]), ("teenagers expecting a reward", ["reward|rewards", "adolescents|adolescence"]),
 ("stopping yourself from pressing a button", ["inhibition|stop signal"]), ("remembering a list of words", ["verbal|words|word", "encoding|retrieval|recall"]),
 ("Alzheimer's early warning signs", ["mci|mild cognitive"]), ("dopamine and Parkinson's medication", ["parkinson", "dopamine|dopaminergic|levodopa"]),
 ("attention problems in autism", ["autism", "attention|attentional"])]


def main():
    ns, model_dir, emb_path = sys.argv[1:4]
    coords, meta, vocab, X = P.load(ns)
    z = np.load(emb_path); ids, E = z["ids"], z["E"]; N = len(ids)
    pos = {i: k for k, i in enumerate(meta.id.values)}; rows = np.array([pos[i] for i in ids])
    Xs = X.tocsc()[rows]; T = [P.clean_text(t, "") for t in meta.title.values[rows]]
    idpos = {int(i): k for k, i in enumerate(ids)}
    stem = snowballstemmer.stemmer("english").stemWord
    docs = [[stem(w) for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP] for t in T]
    dl = np.array([len(d) for d in docs]); avg = dl.mean(); df, inv = {}, {}
    for i, d in enumerate(docs):
        for w in set(d):
            df[w] = df.get(w, 0) + 1; inv.setdefault(w, []).append((i, d.count(w)))
    idf = lambda w: math.log(1 + (N - df.get(w, 0) + .5) / (df.get(w, 0) + .5))

    def qstems(q, syn):
        ws = []
        for w in re.findall(r"[a-z0-9]+", q.lower()):
            if w not in STOP: ws += [w] + (SYN.get(w, []) if syn else [])
        return list(dict.fromkeys(stem(x) for x in ws))

    def bm25(q, syn, k1=1.2, b=.75):
        sc = np.zeros(N)
        for w in qstems(q, syn):
            for i, f in inv.get(w, []): sc[i] += idf(w) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[i] / avg))
        return sc

    def rare(q):
        st = qstems(q, True); tot = sum(idf(w) for w in st) or 1; sc = np.zeros(N)
        for w in st:
            for i, f in inv.get(w, []): sc[i] += idf(w) / tot
        return sc

    cases = []  # (set, topic, style, query, relevant mask)
    for topic, qs in MESH.items():
        pm = [int(x) for x in open(os.path.join(ROOT, "tests", "fixtures", "mesh", topic + ".txt")).read().split()]
        r = np.zeros(N, bool); r[[idpos[p] for p in pm if p in idpos]] = True
        for style, q in zip(STYLES, qs): cases.append(("MeSH", topic, style, q, r))
    for q, groups in TERMS:
        r = np.ones(N, bool)
        for g in groups:
            gm = np.zeros(N, bool)
            for t in [t for t in g.split("|") if t in vocab]: gm |= Xs[:, vocab.index(t)].toarray().ravel() > 0.001
            r &= gm
        cases.append(("Terms", q, "lay", q, r))
    Q = Embedder(model_dir).embed([c[3] for c in cases])
    methods = {"keyword": lambda i, q: bm25(q, False), "keyword+synonyms": lambda i, q: bm25(q, True),
               "semantic": lambda i, q: E @ Q[i], "hybrid": lambda i, q: E @ Q[i] + .2 * rare(q)}
    rows_out = []
    for i, (st, topic, style, q, r) in enumerate(cases):
        row = {"set": st, "topic": topic, "style": style, "query": q, "n_relevant": int(r.sum())}
        for name, f in methods.items():
            sc = f(i, q); order = np.argsort(-sc, kind="stable"); top = order[:10]
            hit = np.array([bool(r[j]) and sc[j] > 0 for j in top])
            dcg = float((hit / np.log2(np.arange(2, 12))).sum()); ideal = float((1 / np.log2(np.arange(2, 2 + min(10, int(r.sum()))))).sum())
            first = next((k + 1 for k, j in enumerate(order[:1000]) if r[j] and sc[j] > 0), None)
            row[name] = {"p10": float(hit.mean()), "ndcg10": dcg / ideal if ideal else 0.0, "mrr": 1 / first if first else 0.0}
        rows_out.append(row)

    def summarise(sel, metric):
        return {m: float(np.mean([r[m][metric] for r in sel])) for m in methods}

    def boot(sel, a, b, metric="ndcg10", n=5000, seed=0):
        """Paired bootstrap of mean(a-b), resampling TOPICS so the 4 phrasings of a topic stay together."""
        rng = np.random.default_rng(seed); groups = {}
        for r in sel: groups.setdefault(r["topic"], []).append(r[a][metric] - r[b][metric])
        g = [np.array(v) for v in groups.values()]
        means = [np.concatenate([g[k] for k in rng.integers(0, len(g), len(g))]).mean() for _ in range(n)]
        d = np.concatenate(g)
        return {"diff": float(d.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
                "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}

    summary = {}
    groups = {"MeSH: all 56": [r for r in rows_out if r["set"] == "MeSH"], "Terms: all 32": [r for r in rows_out if r["set"] == "Terms"]}
    for s in STYLES: groups[f"MeSH: {s}"] = [r for r in rows_out if r["set"] == "MeSH" and r["style"] == s]
    for g, sel in groups.items():
        summary[g] = {"n": len(sel), **{k: summarise(sel, k) for k in ("p10", "ndcg10", "mrr")}}
    comparisons = {}
    for g in ("MeSH: all 56", "Terms: all 32"):
        for a, b in (("semantic", "keyword+synonyms"), ("hybrid", "keyword+synonyms"), ("hybrid", "semantic")):
            comparisons[f"{g} | {a} minus {b}"] = boot(groups[g], a, b)
    out = {"corpus": N, "queries": rows_out, "summary": summary, "comparisons": comparisons}
    if len(sys.argv) > 4: json.dump(out, open(sys.argv[4], "w"), indent=1)
    names = list(methods)
    print(f"{'group':24s}{'n':>4s}" + "".join(f"{m:>18s}" for m in names) + "    (nDCG@10)")
    for g, s in summary.items(): print(f"{g:24s}{s['n']:4d}" + "".join(f"{s['ndcg10'][m]:18.3f}" for m in names))
    print(f"\n{'group':24s}{'n':>4s}" + "".join(f"{m:>18s}" for m in names) + "    (precision@10)")
    for g, s in summary.items(): print(f"{g:24s}{s['n']:4d}" + "".join(f"{s['p10'][m]:18.3f}" for m in names))
    print("\nPaired differences in nDCG@10 (95% bootstrap interval over topics):")
    for k, c in comparisons.items(): print(f"  {k:52s} {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]  W/T/L {c['wins']}/{c['ties']}/{c['losses']}")
    print("\nMeSH per query (precision@10):")
    for r in rows_out:
        if r["set"] == "MeSH": print(f"  {r['topic'][:22]:22s} {r['style'][:10]:10s} {r['query'][:44]:44s}" + "".join(f"{r[m]['p10']:6.1f}" for m in names))


if __name__ == "__main__":
    main()
