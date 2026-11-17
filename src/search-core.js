// Search core: pure functions, no DOM. Used by the browser (window.AtlasSearch) and by Node tests.
(function (root) {
  'use strict';
  const STOP = new Set(('a an the of in on for with and or to by from at as is are was were be this that how your their others other ' +
    'people when who what do does they them after before').split(' '));
  // Lay-to-technical synonyms. Kept short on purpose: meaning-based ranking covers paraphrase.
  const SYN = {
    kids: ['children', 'child'], kid: ['children', 'child'], teenagers: ['adolescents', 'adolescent'], teens: ['adolescents'],
    old: ['older', 'aging', 'elderly'], sad: ['depression', 'depressed'], smokers: ['smoking', 'nicotine'], worry: ['anxiety'],
    scared: ['fear'], hyperactivity: ['adhd'], autistic: ['autism'], forgetting: ['memory'], drinking: ['alcohol'],
    mindful: ['mindfulness', 'meditation'], mindfulness: ['meditation'],
  };
  const words = t => String(t).toLowerCase().match(/[a-z0-9]+/g) || [];
  // Deliberately simple stemming (drop a plural s, keep 7 letters). Benchmarked equal to Snowball here.
  const stem = w => (w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w).slice(0, 7);

  function buildIndex(titles) {
    const N = titles.length, inv = new Map(), dl = new Float32Array(N); let total = 0;
    for (let i = 0; i < N; i++) {
      const tf = new Map();
      for (const w of words(titles[i])) if (!STOP.has(w)) { const s = stem(w); tf.set(s, (tf.get(s) || 0) + 1); }
      let len = 0;
      for (const [s, f] of tf) { len += f; let p = inv.get(s); if (!p) inv.set(s, p = { docs: [], tfs: [] }); p.docs.push(i); p.tfs.push(f); }
      dl[i] = len; total += len;
    }
    return { N, inv, dl, avg: total / Math.max(1, N) };
  }

  // One concept per meaningful query word, carrying the stems of the word and its synonyms.
  function concepts(query) {
    const out = [], seen = new Set();
    for (const w of words(query)) {
      if (STOP.has(w) || seen.has(w)) continue; seen.add(w);
      out.push({ word: w, stems: [...new Set([w, ...(SYN[w] || [])].map(stem))] });
    }
    return out;
  }

  const idf = (index, s) => { const p = index.inv.get(s), df = p ? p.docs.length : 0; return Math.log(1 + (index.N - df + 0.5) / (df + 0.5)); };

  function bm25(index, cons, k1 = 1.2, b = 0.75) {
    const sc = new Float32Array(index.N);
    for (const s of new Set(cons.flatMap(c => c.stems))) {
      const p = index.inv.get(s); if (!p) continue; const w = idf(index, s);
      for (let j = 0; j < p.docs.length; j++) { const i = p.docs[j], f = p.tfs[j]; sc[i] += w * f * (k1 + 1) / (f + k1 * (1 - b + b * index.dl[i] / index.avg)); }
    }
    return sc;
  }

  // Cosine similarity of the query vector against int8 title embeddings (unit vectors times 1/scale).
  function semantic(emb, n, dim, qvec, scale) {
    const sc = new Float32Array(n);
    for (let i = 0, o = 0; i < n; i++, o += dim) { let s = 0; for (let d = 0; d < dim; d++) s += emb[o + d] * qvec[d]; sc[i] = s * scale; }
    return sc;
  }

  function topK(scores, k, positiveOnly) {
    const idx = [];
    for (let i = 0; i < scores.length; i++) if (!positiveOnly || scores[i] > 0) idx.push(i);
    idx.sort((a, b) => scores[b] - scores[a] || a - b);
    return idx.slice(0, k);
  }

  // Which query words do the top results fail to cover?
  //  absent: no study title contains the word at all.
  //  missed: some titles contain it, it is rare, and none of the top results do. Best-matching first, at most `limit`.
  function coverage(index, cons, top, scores, opts = {}) {
    const maxShare = opts.maxShare ?? 0.01, limit = opts.limit ?? 2, per = opts.per ?? 5, minScore = opts.minScore ?? -Infinity;
    const uncommon = opts.uncommon || (() => true);   // predicate: is this an uncommon (topical) word?
    const inTop = new Set(top), absent = [], missed = [];
    for (const c of cons) {
      const docs = new Set();
      for (const s of c.stems) { const p = index.inv.get(s); if (p) for (const d of p.docs) docs.add(d); }
      if (docs.size === 0) { absent.push(c.word); continue; }
      if (docs.size > maxShare * index.N || !uncommon(c.word)) continue;
      let covered = false; for (const d of docs) if (inTop.has(d)) { covered = true; break; }
      if (covered) continue;
      const best = [...docs].sort((a, b) => scores[b] - scores[a] || a - b).slice(0, per);
      if (scores[best[0]] >= minScore) missed.push({ word: c.word, count: docs.size, docs: best });
    }
    missed.sort((a, b) => scores[b.docs[0]] - scores[a.docs[0]] || a.count - b.count);   // most relevant concept first
    return { absent, missed: missed.slice(0, limit) };
  }

  // ---- BERT WordPiece tokenizer (uncased), matching Hugging Face `tokenizers` for this model ----
  const isWs = c => c === ' ' || c === '\t' || c === '\n' || c === '\r' || /\p{Zs}/u.test(c);
  const isCtl = c => c !== '\t' && c !== '\n' && c !== '\r' && /\p{C}/u.test(c);
  const isPunct = c => { const k = c.codePointAt(0); return (k >= 33 && k <= 47) || (k >= 58 && k <= 64) || (k >= 91 && k <= 96) || (k >= 123 && k <= 126) || /\p{P}/u.test(c); };
  const isCjk = k => (k >= 0x4E00 && k <= 0x9FFF) || (k >= 0x3400 && k <= 0x4DBF) || (k >= 0x20000 && k <= 0x2A6DF) || (k >= 0x2A700 && k <= 0x2B73F) ||
    (k >= 0x2B740 && k <= 0x2B81F) || (k >= 0x2B820 && k <= 0x2CEAF) || (k >= 0xF900 && k <= 0xFAFF) || (k >= 0x2F800 && k <= 0x2FA1F);

  function basicTokens(text) {
    let s = '';
    for (const c of String(text)) {                         // clean + pad CJK
      const k = c.codePointAt(0);
      if (k === 0 || k === 0xFFFD || isCtl(c)) continue;
      if (isWs(c)) s += ' '; else if (isCjk(k)) s += ' ' + c + ' '; else s += c;
    }
    s = s.normalize('NFD').replace(/\p{Mn}/gu, '').toLowerCase();   // strip accents, lowercase
    const out = [];
    for (const chunk of s.split(/\s+/)) {
      if (!chunk) continue; let cur = '';
      for (const c of chunk) { if (isPunct(c)) { if (cur) out.push(cur); out.push(c); cur = ''; } else cur += c; }
      if (cur) out.push(cur);
    }
    return out;
  }

  function wordpiece(word, vocab, unk) {
    const chars = [...word]; if (chars.length > 100) return [unk];
    const ids = []; let start = 0;
    while (start < chars.length) {
      let end = chars.length, id = -1;
      while (start < end) { const sub = (start > 0 ? '##' : '') + chars.slice(start, end).join(''); const v = vocab.get(sub); if (v !== undefined) { id = v; break; } end--; }
      if (id < 0) return [unk];
      ids.push(id); start = end;
    }
    return ids;
  }

  function bertTokenize(text, vocab, maxLen = 128) {
    const cls = vocab.get('[CLS]'), sep = vocab.get('[SEP]'), unk = vocab.get('[UNK]'), ids = [];
    for (const w of basicTokens(text)) for (const id of wordpiece(w, vocab, unk)) ids.push(id);
    return [cls, ...ids.slice(0, maxLen - 2), sep];
  }

  // Mean-pool token vectors and L2-normalise (no padding is ever used, so every token counts).
  function pool(hidden, tokens, dim) {
    const v = new Float32Array(dim);
    for (let t = 0; t < tokens; t++) for (let d = 0; d < dim; d++) v[d] += hidden[t * dim + d];
    let n = 0; for (let d = 0; d < dim; d++) { v[d] /= tokens; n += v[d] * v[d]; }
    n = Math.sqrt(n) || 1; for (let d = 0; d < dim; d++) v[d] /= n;
    return v;
  }

  // A word is uncommon in general English if the model's vocabulary splits it into pieces or ranks it late.
  const isUncommon = (word, vocab, cutoff = 12000) => { const ids = bertTokenize(word, vocab); return ids.length > 3 || ids[1] >= cutoff; };

  const api = { STOP, SYN, isUncommon, words, stem, buildIndex, concepts, bm25, semantic, topK, coverage, basicTokens, bertTokenize, pool };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.AtlasSearch = api;
})(typeof self !== 'undefined' ? self : this);
