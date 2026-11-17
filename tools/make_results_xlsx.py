"""Benchmark results workbook. Usage: python tools/make_results_xlsx.py cache/benchmark.json out.xlsx"""
import json, sys
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter as L

B = json.load(open(sys.argv[1])); Q = B["queries"]
M = ["keyword", "keyword+synonyms", "semantic", "hybrid"]
MN = ["Keyword", "Keyword + synonyms", "Semantic", "Hybrid"]
wb = Workbook(); F = Font(name="Arial", size=10); HB = Font(name="Arial", size=10, bold=True, color="FFFFFF")
FILL = PatternFill("solid", fgColor="2F3B52"); BOLD = Font(name="Arial", size=10, bold=True)


def style(ws, widths, header_rows=(1,)):
    for row in ws.iter_rows():
        for c in row:
            c.font = F; c.alignment = Alignment(vertical="top", wrap_text=True)
    for r in header_rows:
        for c in ws[r]:
            if c.value is not None: c.font = HB; c.fill = FILL
    for i, w in enumerate(widths, 1): ws.column_dimensions[L(i)].width = w


# ---- Per query ----
pq = wb.active; pq.title = "Per query"
pq.append(["Test set", "Topic", "Query style", "Query", "Relevant studies in corpus"] + [f"nDCG@10: {m}" for m in MN]
          + [f"Precision@10: {m}" for m in MN] + [f"Reciprocal rank: {m}" for m in MN])
for r in Q:
    pq.append([r["set"], r["topic"], r["style"], r["query"], r["n_relevant"]] + [round(r[m]["ndcg10"], 4) for m in M]
              + [round(r[m]["p10"], 2) for m in M] + [round(r[m]["mrr"], 4) for m in M])
style(pq, [9, 24, 16, 46, 12] + [13] * 12); pq.freeze_panes = "E2"
last = len(Q) + 1
rng = lambda col: f"'Per query'!${col}$2:${col}${last}"

# ---- Summary ----
s = wb.create_sheet("Summary", 0)
s.append(["Search ranking benchmark: keyword vs semantic vs hybrid"]); s["A1"].font = Font(name="Arial", size=13, bold=True)
s.append([f"{B['corpus']:,} Neurosynth fMRI studies. Every method ranks study titles only. Higher is better; 1.0 is perfect."])
s.append([])
hdr = ["Group", "Test set", "Query style", "Queries"] + MN
groups = [("MeSH judge, all queries", "MeSH", None), ("MeSH judge: lay phrasing", "MeSH", "lay"), ("MeSH judge: researcher phrasing", "MeSH", "researcher"),
          ("MeSH judge: topic + population", "MeSH", "topic+population"), ("MeSH judge: bare keyword", "MeSH", "keyword"), ("Abstract-term judge, all queries", "Terms", None)]
heads = []
for title, c0 in (("Mean nDCG@10", 6), ("Mean precision@10", 10), ("Mean reciprocal rank of first relevant result", 14)):
    s.append([title]); s.cell(s.max_row, 1).font = BOLD
    s.append(hdr); heads.append(s.max_row)
    for g, st, sty in groups:
        r = s.max_row + 1
        n = f'=COUNTIFS({rng("A")},B{r},{rng("C")},C{r})' if sty else f'=COUNTIFS({rng("A")},B{r})'
        row = [g, st, sty or "all", n]
        for k in range(4):
            col = L(c0 + k)
            row.append(f'=AVERAGEIFS({rng(col)},{rng("A")},$B{r},{rng("C")},$C{r})' if sty else f'=AVERAGEIFS({rng(col)},{rng("A")},$B{r})')
        s.append(row)
        for k in range(4): s.cell(r, 5 + k).number_format = "0.000"
    s.append([])
s.append(["Paired differences in nDCG@10"]); s.cell(s.max_row, 1).font = BOLD
s.append(["Comparison", "Test set", "", "Queries", "Mean difference", "95% interval low", "95% interval high", "Wins / ties / losses"]); heads.append(s.max_row)
pairs = [("semantic", "keyword+synonyms"), ("hybrid", "keyword+synonyms"), ("hybrid", "semantic")]
for st, label in (("MeSH", "MeSH: all 56"), ("Terms", "Terms: all 32")):
    for a, b in pairs:
        c = B["comparisons"][f"{label} | {a} minus {b}"]; r = s.max_row + 1
        ca, cb = L(6 + M.index(a)), L(6 + M.index(b))
        s.append([f"{MN[M.index(a)]} minus {MN[M.index(b)]}", st, "", f'=COUNTIFS({rng("A")},B{r})',
                  f'=AVERAGEIFS({rng(ca)},{rng("A")},$B{r})-AVERAGEIFS({rng(cb)},{rng("A")},$B{r})',
                  round(c["lo"], 3), round(c["hi"], 3), f'{c["wins"]} / {c["ties"]} / {c["losses"]}'])
        for k in (5, 6, 7): s.cell(r, k).number_format = "+0.000;-0.000;0.000"
s.append(["Intervals and win counts come from tools/benchmark.py (paired bootstrap, 5,000 resamples of topics, seed 0). They are typed values; every mean on this tab is a formula over the Per query tab."])
s.append([])
s.append(["What the results say"]); s.cell(s.max_row, 1).font = BOLD
for t in ["Semantic ranking beats keyword ranking with synonyms on both judges. The intervals for that difference exclude zero.",
          "The hybrid is not measurably different from semantic ranking alone. Its interval spans zero on both judges.",
          "The gap is largest for lay phrasing and for topic + population queries, and smallest when the query is the bare technical keyword.",
          "Keyword ranking wins outright on \"mindful practices for ADHD kids\" (precision 0.5 vs 0.0), because the words \"ADHD kids\" dominate the embedding.",
          "Decision: rank by meaning, fall back to keywords while the model loads, and list studies matching any query word the top results miss."]:
    s.append([t])
style(s, [44, 10, 18, 9, 16, 19, 16, 18], header_rows=heads)
s["A1"].font = Font(name="Arial", size=13, bold=True)
for r in range(1, s.max_row + 1):
    if s.cell(r, 2).value is None and s.cell(r, 1).value and r not in heads and len(str(s.cell(r, 1).value)) > 50:
        s.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8); s.row_dimensions[r].height = 30
for r in range(1, s.max_row + 1):
    if s.cell(r, 1).value in ("Mean nDCG@10", "Mean precision@10", "Mean reciprocal rank of first relevant result", "Paired differences in nDCG@10", "What the results say"):
        s.cell(r, 1).font = BOLD

# ---- Earlier small test ----
e = wb.create_sheet("Earlier 11-query test")
e.append(["Superseded pilot: 1,145 studies, relevance defined from title wording (which favours keyword matching). Share of relevant studies found in the top 10."])
qs = ["mindful practices for ADHD kids", "kids with hyperactivity", "looking at faces in autism", "memory loss in old age", "tremor and slowness of movement",
      "anticipating rewards with stimulant medication", "theory of mind in autistic adults", "levodopa effects on the brain", "early warning signs before dementia",
      "resting brain networks", "meditation"]
res = {"Semantic": [0, 1, 1, .8, .8, .3, .4, 1, .1, 1, 1], "Keyword + synonyms": [1, .9, 1, .3, 1, .5, .2, 1, 0, 1, 1],
       "Hybrid, rank fusion": [0, 1, 1, .4, 1, .6, .3, 1, 0, 1, 1], "Hybrid, rank fusion (semantic x2)": [0, 1, 1, .5, 1, .5, .3, 1, 0, 1, 1],
       "Hybrid, score mix 70% semantic": [.33, 1, 1, .5, 1, .6, .3, 1, 0, 1, 1], "Hybrid, score mix 50/50": [.33, 1, 1, .4, 1, .6, .3, 1, 0, 1, 1],
       "Semantic + word coverage": [0, 1, 1, .5, 1, .6, .3, 1, .1, 1, 1]}
e.append(["Query"] + list(res))
for i, q in enumerate(qs): e.append([q] + [v[i] for v in res.values()])
e.append(["Mean"] + [f"=AVERAGE({L(c)}3:{L(c)}13)" for c in range(2, 9)])
for c in range(2, 9): e.cell(14, c).number_format = "0.000"
e.append([]); e.append(["Rank of the meditation and attention paper for the first query (1 is best, of 1,145)"]); e.append(["Method", "Rank"])
for k, v in [("Semantic", 89), ("Keyword + synonyms", 1), ("Hybrid, rank fusion", 18), ("Hybrid, rank fusion (semantic x2)", 35),
             ("Hybrid, score mix 70% semantic", 7), ("Hybrid, score mix 50/50", 1), ("Semantic + word coverage", 35)]: e.append([k, v])
style(e, [46] + [17] * 7, header_rows=(2, 17)); e.merge_cells("A1:H1"); e.row_dimensions[1].height = 30
for c in e[14]: c.font = BOLD

# ---- Methods ----
m = wb.create_sheet("Methods and limits")
m.append(["Item", "Detail"])
for a, b in [
 ("Corpus", f"{B['corpus']:,} studies from Neurosynth v7 that have at least one usable activation peak (papers from 1997 to 2018)."),
 ("What each method sees", "The study title only. No method sees abstracts, MeSH headings or Neurosynth terms."),
 ("Keyword", "BM25 over title words after dropping common words and reducing each word to a stem."),
 ("Keyword + synonyms", "Keyword, plus a fixed list of 14 lay-to-technical synonyms such as kids to children. The list was fixed before the MeSH test ran."),
 ("Semantic", "Cosine similarity between the query and each title using all-MiniLM-L6-v2 sentence embeddings (int8 ONNX export, 384 dimensions)."),
 ("Hybrid", "Semantic score plus 0.2 times the share of the query's rarity-weighted words found in the title. Both constants were chosen on the abstract-term set, so the MeSH set is a held-out test of the hybrid."),
 ("MeSH judge", "A study is relevant if PubMed indexers assigned the topic's MeSH headings. Lists were retrieved from PubMed on 17 Nov 2026 for 1997 to 2018, restricted to papers whose title or abstract mentions fMRI. 14 topics, 4 phrasings each."),
 ("Abstract-term judge", "A study is relevant if its abstract uses the listed Neurosynth terms (tf-idf above 0.001). 32 lay-language queries."),
 ("nDCG@10", "Rewards relevant studies in the top 10, with more credit for higher positions. 1.0 means the top 10 is as good as possible."),
 ("Precision@10", "Share of the top 10 results that are relevant."),
 ("Limit: incomplete judging", "The fMRI filter on the MeSH lists misses some relevant papers, so all scores are underestimates. This affects every method equally."),
 ("Limit: who wrote the queries", "Claude wrote all 88 queries. They are not real researcher queries, and lay phrasing may be over-represented."),
 ("Limit: one synonym was added for a test query", "The mindful to meditation synonym was added after seeing the user's example query. It helps keyword ranking on one MeSH query."),
 ("Limit: sample size", "14 topics and 32 queries. Intervals resample topics, so they reflect topic-to-topic variation but not other choices such as the model."),
 ("Reproduce", "python tools/benchmark.py <neurosynth_dir> <model_dir> cache/emb_all.npz")]:
    m.append([a, b])
style(m, [34, 110])
wb.save(sys.argv[2]); print("saved", sys.argv[2])
