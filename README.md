# Dopamine Disorder Atlas

Interactive 3D brain linking fMRI studies of autism, Alzheimer's, Parkinson's and ADHD (Neurosynth v7)
to the dopaminergic pathways and D1-D5 receptor territories, with plain-language search over 14,360 studies.
Static site: no server code, no database. Search runs in the visitor's browser; no query leaves the page.

## Build and test

    pip install -r requirements.txt
    npm ci
    npx playwright install chromium      # once
    ./run_tests.sh                       # downloads data and model, builds, runs every test

`run_tests.sh` ends with `ALL TESTS PASSED` and leaves the site in `dist/`. Host that folder on any
static host (GitHub Pages, Netlify, S3, nginx). Preview locally: `python3 -m http.server 8080 -d dist`.

## Layout

| Path | Purpose |
| --- | --- |
| `nodes.py` | The 15 regions (MNI centres, D1-D5 ratings) and 6 pathways. Edit to change the map. |
| `pipeline.py` | Data pipeline: study selection, region hit rates, Fisher tests, mesh packing. |
| `build_data.py` | CLI: Neurosynth files -> `build/` (atlas data, search index, embeddings, model). |
| `embed.py`, `search_build.py` | Title embeddings with all-MiniLM-L6-v2 (checksums pinned) and the search files. |
| `src/page.html`, `src/app.js` | Page markup/styles and the three.js viewer. |
| `src/search-core.js`, `src/search.js` | Ranking, tokenizer and the search bar. |
| `tools/benchmark.py` | 88-query search benchmark (MeSH lists in `tests/fixtures/mesh/`). |
| `assemble.py` | Builds `dist/` (hostable site) and `artifact/` (single-file version). |
| `tests/test_pipeline.py` | Unit tests on hand-computed synthetic data. |
| `tests/test_data.py` | Integration tests on the real build, incl. known-neuroscience checks. |
| `tests/test_search.py` | Embedding and search-file tests, incl. checksum enforcement. |
| `tests/search-core.test.js` | Ranking and tokenizer unit tests (Node). |
| `tests/site.test.js` | Browser tests (Playwright, real Chromium, served over HTTP). |

## How search works

1. Build time: every study title is embedded once (384 numbers, stored as one byte each).
2. In the browser: keyword results (BM25) appear at once. If "Match by meaning" is on, the page downloads the
   model and runtime once (about 38 MB plus 5.5 MB of title vectors), embeds the query and re-ranks by cosine similarity.
3. If none of the top results contains an uncommon word from the query, the page says so and lists studies that do.

On an 88-query benchmark judged by PubMed MeSH headings and by Neurosynth abstract terms, ranking by meaning
scored 0.58 and 0.69 nDCG@10 against 0.44 and 0.44 for keywords with synonyms. Reproduce with `tools/benchmark.py`.

## What the tests cover

- **Pipeline (19 tests):** term thresholding, hemisphere mirroring, 10 mm boundary, base rates and ratios
  against hand-worked values, Fisher p against an independent calculation, missing metadata, strict JSON.
- **Built data (12 tests):** schema, region membership recomputed by brute force, table counts recounted
  from per-study flags, mesh geometry, expected findings, peaks traced back to raw Neurosynth rows.
- **Embeddings and search files (9 tests):** tampered model refused, no padding in embeddings, stored
  vectors match a fresh embedding, atlas studies all searchable, peaks trace back to source.
- **Search core (9 tests):** stemming, synonyms, BM25, missed-word logic, and the JavaScript tokenizer
  checked against the Python reference on every study title.
- **Site (39 tests):** everything in the atlas (controls, 3D clicking, themes, contrast, keyboard, phone,
  failure modes, content-security policy) plus search: browser-vs-Python embedding parity, the benchmark
  run through the shipped code, the mindfulness example, hostile and odd input, model or index failing to
  load, stale results, data-saver default.

## Before publishing

- Check the licence terms of the Neurosynth data and credit it on the site.
- The receptor ratings in `nodes.py` are qualitative summaries, not measured values.
- The page requests web fonts from Google Fonts; it falls back to system fonts if they are blocked.
  Remove the `<link>` in `src/page.html` and the two font hosts in `assemble.py`'s CSP to make zero
  third-party requests.
- The model download is on by default (off for data-saver browsers). `dist/` is about 47 MB; the largest file is 23 MB.
- The CSP is set by a `<meta>` tag. If your host lets you set headers, also send it as a real header
  plus `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`.
