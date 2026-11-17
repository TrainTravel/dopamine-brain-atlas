// Search bar: keyword ranking at once, meaning-based ranking once the language model has loaded.
// Everything runs in the visitor's browser; no query leaves the page.
(function () {
  'use strict';
  const A = window.__atlas, C = window.AtlasSearch, form = document.querySelector('#sform');
  if (!A || !C || !form) return;
  const $ = s => document.querySelector(s), CFG = window.ATLAS_SEARCH_CONFIG || {};
  const base = CFG.base || '', TOP = 20;
  const q = $('#q'), sem = $('#sem'), status = $('#sstatus'), box = $('#results');
  const st = { base: null, semantic: null, sem: 'idle', run: 0, hits: [], query: '', mode: '' };
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const say = t => { status.textContent = t; };
  const pref = { get() { try { return localStorage.getItem('atlas.sem'); } catch (e) { return null; } }, set(v) { try { localStorage.setItem('atlas.sem', v); } catch (e) { /* storage unavailable */ } } };

  if (CFG.noModel) { $('#semrow').hidden = true; sem.checked = false; }
  else { const p = pref.get(); sem.checked = p ? p === '1' : !(navigator.connection && navigator.connection.saveData); }
  form.hidden = false;

  async function get(url, type, onProgress) {
    const r = await fetch(base + url); if (!r.ok) throw new Error(url + ' HTTP ' + r.status);
    if (!onProgress || !r.body) return type === 'json' ? r.json() : r.arrayBuffer();
    const total = +r.headers.get('content-length') || 0, reader = r.body.getReader(), parts = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; onProgress(got, total); }
    const out = new Uint8Array(got); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
    return out.buffer;
  }

  function loadBase() {
    if (!st.base) st.base = (async () => {
      say('Loading the study index…');
      const [meta, pk] = await Promise.all([get('search/meta.json', 'json'), get('search/peaks.bin')]);
      const peaks = new Int8Array(pk), off = new Uint32Array(meta.n + 1);
      for (let i = 0; i < meta.n; i++) off[i + 1] = off[i] + meta.peakCounts[i] * 3;
      if (off[meta.n] !== peaks.length || meta.titles.length !== meta.n) throw new Error('search files are inconsistent');
      return { meta, peaks, off, index: C.buildIndex(meta.titles) };
    })().catch(e => { st.base = null; throw e; });
    return st.base;
  }

  function loadScript(src) {
    return new Promise((ok, fail) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = () => fail(new Error('could not load ' + src)); document.head.append(s); });
  }

  function loadSemantic() {
    if (!st.semantic) st.semantic = (async () => {
      st.sem = 'loading';
      const { meta } = await loadBase();
      const mb = n => (n / 1048576).toFixed(0);
      say('Loading the language model (one-time download)…');
      if (!window.ort) await loadScript(base + 'vendor/ort/ort.wasm.min.js');
      const ort = window.ort; ort.env.wasm.numThreads = 1; ort.env.wasm.wasmPaths = new URL(base + 'vendor/ort/', document.baseURI).href;
      const [embBuf, vocabList, modelBuf] = await Promise.all([get('search/emb.bin'), get('model/vocab.json', 'json'),
        get('model/model_quantized.onnx', 'buf', (got, total) => { if (st.sem === 'loading') say(`Loading the language model, ${mb(got)}${total ? ' of ' + mb(total) : ''} MB (one-time download)…`); })]);   // a download still in flight must not overwrite a failure message
      const emb = new Int8Array(embBuf); if (emb.length !== meta.n * meta.dim) throw new Error('embedding file does not match the study index');
      const session = await ort.InferenceSession.create(new Uint8Array(modelBuf), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      const vocab = new Map(vocabList.map((t, i) => [t, i]));
      const embed = async text => {
        const ids = C.bertTokenize(text, vocab, meta.maxTokens), n = ids.length, big = v => BigInt64Array.from(v, x => BigInt(x));
        const feeds = { input_ids: new ort.Tensor('int64', big(ids), [1, n]), attention_mask: new ort.Tensor('int64', big(ids.map(() => 1)), [1, n]),
          token_type_ids: new ort.Tensor('int64', big(ids.map(() => 0)), [1, n]) };
        const out = await session.run(feeds), h = out.last_hidden_state || out[Object.keys(out)[0]];
        return C.pool(h.data, n, meta.dim);
      };
      st.sem = 'ready';
      return { emb, embed, vocab };
    })().catch(e => { st.sem = 'failed'; st.semantic = null; throw e; });
    return st.semantic;
  }

  function hit(B, i) {
    const m = B.meta;
    return { i, pmid: m.ids[i], title: m.titles[i], year: m.years[i], journal: m.journals[m.journalIdx[i]], author: m.authors[i], peaks: B.peaks.subarray(B.off[i], B.off[i + 1]) };
  }

  function row(h) {
    const li = el('li'), b = el('button'); b.type = 'button';
    const t = el('span', 't'); const di = A.D.studies.findIndex(s => s[0] === h.pmid);
    if (di >= 0) A.D.keys.forEach((k, j) => { if (A.D.studies[di][5] >> j & 1) { const d = el('span', 'gd'); d.style.background = `var(--${k.toLowerCase()})`; d.title = A.D.groups[k].name; t.append(d); } });
    t.append(h.title);
    b.append(t, el('span', 'm', `${h.author} ${h.year || ''} · ${h.journal} · ${h.peaks.length / 3} peaks`));
    b.onclick = () => { box.querySelectorAll('li.on').forEach(x => x.classList.remove('on')); li.classList.add('on'); A.selectHit(h); };
    const a = el('a', null, 'PubMed'); a.href = `https://pubmed.ncbi.nlm.nih.gov/${h.pmid}/`; a.target = '_blank'; a.rel = 'noopener';
    li.append(b, a); li.dataset.pmid = h.pmid; return li;
  }

  function render(B, query, scores, mode, lowConfidence, vocab) {
    const cons = C.concepts(query), top = C.topK(scores, TOP, mode === 'keyword');
    // In meaning mode, flag uncommon query words that none of the top results contain (keyword mode matches words directly).
    const cov = C.coverage(B.index, cons, top, scores, mode === 'meaning' ? { uncommon: w => C.isUncommon(w, vocab), minScore: 0.25 } : { limit: 0 });
    st.hits = top.map(i => hit(B, i)); st.query = query; st.mode = mode;
    box.textContent = ''; box.hidden = false;
    box.append(el('h2', null, 'Search results'));
    const how = mode === 'meaning' ? 'ranked by meaning' : 'ranked by matching words';
    if (!top.length) box.append(el('p', 'note', mode === 'keyword' ? 'No study title contains those words. Turn on Match by meaning, or try other words.' : 'No results.'));
    else box.append(el('p', 'note', `Top ${top.length} of ${B.meta.n.toLocaleString()} studies for “${query}”, ${how}.` +
      (lowConfidence ? ' No study is a close match; these are the nearest.' : '') + (A.three ? ' Their peaks are the large dots on the brain.' : '')));
    const ul = el('ul', 'st'); st.hits.forEach(h => ul.append(row(h))); box.append(ul);
    for (const m of cov.missed) {
      const d = el('div', 'miss');
      d.append(el('p', 'note', `None of the top results mention “${m.word}”. ${m.count === 1 ? 'The one study that does:' : `${m.count} studies do. The closest:`}`));
      const u = el('ul', 'st'); m.docs.forEach(i => u.append(row(hit(B, i)))); d.append(u); d.dataset.word = m.word; box.append(d);
    }
    if (cov.absent.length) { const p = el('p', 'note', `No study title contains: ${cov.absent.join(', ')}.`); p.id = 'absent'; box.append(p); }
    const clear = el('button', 'btn', 'Clear search'); clear.type = 'button'; clear.style.alignSelf = 'flex-end'; clear.onclick = reset; box.append(clear);
    if (A.three) { const pos = [], own = []; st.hits.forEach((h, k) => { for (let p = 0; p < h.peaks.length; p += 3) { pos.push(h.peaks[p], h.peaks[p + 2], -h.peaks[p + 1]); own.push(k); } }); A.three.setSearch(pos, own); }
    if (matchMedia('(max-width:1000px)').matches) box.scrollIntoView({ block: 'start' }); else $('#right').scrollTop = 0;
  }

  function reset() {
    st.run++; st.hits = []; st.query = ''; box.hidden = true; box.textContent = ''; say('');
    if (A.three) A.three.setSearch([], []);
    if (A.S.hit) { A.S.hit = null; A.sync(); }
  }

  async function search(raw) {
    const query = String(raw).replace(/\s+/g, ' ').trim().slice(0, 300), run = ++st.run;
    if (!query) { reset(); return; }
    let B;
    try { B = await loadBase(); } catch (e) { say('Search is unavailable: the study index did not load.'); return; }
    if (run !== st.run) return;
    const cons = C.concepts(query), kw = C.bm25(B.index, cons);
    const wantMeaning = sem.checked && !CFG.noModel;
    if (!wantMeaning || st.sem !== 'ready') render(B, query, kw, 'keyword');
    if (!wantMeaning) { say('Ranked by matching words.'); return; }
    try {
      const S = await loadSemantic(); if (run !== st.run) return;
      const vec = await S.embed(query); if (run !== st.run) return;
      const sc = C.semantic(S.emb, B.meta.n, B.meta.dim, vec, B.meta.scale);
      let best = -1; for (let i = 0; i < sc.length; i++) if (sc[i] > best) best = sc[i];
      render(B, query, sc, 'meaning', best < 0.4, S.vocab); say('Ranked by meaning.');
    } catch (e) {
      if (run !== st.run) return;
      say('The language model could not load, so results are ranked by matching words.');
      if (st.mode !== 'keyword' || st.query !== query) render(B, query, kw, 'keyword');
    }
  }

  form.addEventListener('submit', e => { e.preventDefault(); search(q.value); });
  q.addEventListener('focus', () => { loadBase().then(() => { if (!st.query) say(''); }).catch(() => {}); }, { once: true });
  q.addEventListener('search', () => { if (!q.value) reset(); });
  sem.addEventListener('change', () => { pref.set(sem.checked ? '1' : '0'); if (st.query) search(st.query); });
  A.onSearchPick = k => { const h = st.hits[k]; if (!h) return; const li = box.querySelector(`li[data-pmid="${h.pmid}"]`); box.querySelectorAll('li.on').forEach(x => x.classList.remove('on')); if (li) li.classList.add('on'); A.selectHit(h); };
  window.__atlasSearch = { st, search, reset, loadBase, loadSemantic };
})();
