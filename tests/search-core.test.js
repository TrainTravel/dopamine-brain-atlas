// Unit tests for src/search-core.js (no browser). Fixtures come from tools/make_search_fixture.py.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const C = require('../src/search-core.js');
const BUILD = path.resolve(process.env.ATLAS_BUILD || 'build'), FX = path.resolve(process.env.ATLAS_FIXTURES || 'cache/fixtures');
const have = f => fs.existsSync(f);

const DOCS = ['Mindfulness meditation training in adults', 'Children with ADHD and reward', 'Reward anticipation in adolescents',
  'Meditative practice and attention', 'Working memory in children', 'Dopamine release in the striatum'];

test('stemmer: plural and length rules', () => {
  assert.equal(C.stem('children'), 'childre'); assert.equal(C.stem('kids'), 'kid'); assert.equal(C.stem('stress'), 'stress');
  assert.equal(C.stem('mindfulness'), C.stem('mindful')); assert.equal(C.stem('meditation'), C.stem('meditative'));
  assert.notEqual(C.stem('mind'), C.stem('mindful')); assert.equal(C.stem('is'), 'is');
});

test('concepts: stopwords dropped, synonyms added, duplicates merged', () => {
  const c = C.concepts('Mindful practices for the ADHD kids, kids!');
  assert.deepEqual(c.map(x => x.word), ['mindful', 'practices', 'adhd', 'kids']);
  assert.ok(c[3].stems.includes(C.stem('children')) && c[0].stems.includes(C.stem('meditation')));
  assert.deepEqual(C.concepts('the of and'), []); assert.deepEqual(C.concepts(''), []); assert.deepEqual(C.concepts('!!! ???'), []);
});

test('bm25: matching documents score, others are zero, rarer words weigh more', () => {
  const ix = C.buildIndex(DOCS), sc = C.bm25(ix, C.concepts('meditation for kids'));
  assert.ok(sc[0] > 0 && sc[3] > 0 && sc[1] > 0 && sc[4] > 0); assert.equal(sc[2], 0); assert.equal(sc[5], 0);
  assert.deepEqual(C.topK(sc, 10, true).sort(), [0, 1, 3, 4]);
  const one = C.bm25(ix, C.concepts('dopamine children'));                 // 'dopamine' in 1 doc, 'children' in 2
  assert.ok(one[5] > one[1] && one[5] > one[4]);
  assert.ok(C.bm25(ix, []).every(v => v === 0));
});

test('topK: sorted, stable on ties, respects k and positiveOnly', () => {
  const s = Float32Array.from([0.5, 0.9, 0, 0.9, -1]);
  assert.deepEqual(C.topK(s, 3), [1, 3, 0]); assert.deepEqual(C.topK(s, 10, true), [1, 3, 0]); assert.deepEqual(C.topK(s, 10), [1, 3, 0, 2, 4]);
});

test('semantic: int8 dot product times scale equals cosine', () => {
  const emb = Int8Array.from([127, 0, 0, 127, -127, 0]), sc = C.semantic(emb, 3, 2, Float32Array.from([1, 0]), 1 / 127);
  assert.ok(Math.abs(sc[0] - 1) < 1e-6 && Math.abs(sc[1]) < 1e-6 && Math.abs(sc[2] + 1) < 1e-6);
});

test('coverage: absent words, missed uncommon words, covered words', () => {
  const ix = C.buildIndex(DOCS), cons = C.concepts('meditation yoga reward');
  const scores = Float32Array.from([0.3, 0.9, 0.8, 0.4, 0.1, 0.0]), top = [1, 2];           // top results are the reward papers
  const cov = C.coverage(ix, cons, top, scores, { maxShare: 1 });
  assert.deepEqual(cov.absent, ['yoga']);
  assert.equal(cov.missed.length, 1); assert.equal(cov.missed[0].word, 'meditation'); assert.deepEqual(cov.missed[0].docs, [3, 0]);
  assert.deepEqual(C.coverage(ix, cons, [0, 1], scores, { maxShare: 1 }).missed, []);                    // now covered
  assert.deepEqual(C.coverage(ix, cons, top, scores, { maxShare: 1, uncommon: () => false }).missed, []); // common words never flagged
  assert.deepEqual(C.coverage(ix, cons, top, scores, { maxShare: 1, minScore: 0.5 }).missed, []);         // too dissimilar to show
  assert.deepEqual(C.coverage(ix, cons, top, scores, { maxShare: 0.1 }).missed, []);                      // too frequent to be 'rare'
  assert.deepEqual(C.coverage(ix, cons, top, scores, { maxShare: 1, limit: 0 }).missed, []);
});

test('pool: mean over tokens then unit length', () => {
  const v = C.pool(Float32Array.from([3, 0, 0, 4]), 2, 2);
  assert.ok(Math.abs(v[0] - 0.6) < 1e-6 && Math.abs(v[1] - 0.8) < 1e-6);
});

const fixtures = have(path.join(BUILD, 'model/vocab.json')) && have(path.join(FX, 'search_eval.json'));
test('tokenizer matches the Python reference on awkward inputs', { skip: !fixtures && 'build and fixtures not present' }, () => {
  const vocab = new Map(JSON.parse(fs.readFileSync(path.join(BUILD, 'model/vocab.json'), 'utf8')).map((t, i) => [t, i]));
  const F = JSON.parse(fs.readFileSync(path.join(FX, 'search_eval.json'), 'utf8'));
  assert.ok(F.tokenizer.length >= 25);
  for (const c of F.tokenizer) assert.deepEqual(C.bertTokenize(c.text, vocab), c.ids, JSON.stringify(c.text.slice(0, 50)));
  assert.deepEqual(C.bertTokenize('', vocab), [101, 102]);
  assert.equal(C.bertTokenize('word '.repeat(500), vocab).length, 128);          // truncation
  assert.ok(C.isUncommon('mindful', vocab) && C.isUncommon('tinnitus', vocab) && C.isUncommon('meditation', vocab));
  assert.ok(!C.isUncommon('staying', vocab) && !C.isUncommon('practices', vocab) && !C.isUncommon('kids', vocab));
});

test('tokenizer matches the Python reference on every study title', { skip: !(fixtures && have(path.join(FX, 'title_tokens.json'))) && 'fixtures not present' }, () => {
  const vocab = new Map(JSON.parse(fs.readFileSync(path.join(BUILD, 'model/vocab.json'), 'utf8')).map((t, i) => [t, i]));
  const titles = JSON.parse(fs.readFileSync(path.join(BUILD, 'search/meta.json'), 'utf8')).titles, ref = JSON.parse(fs.readFileSync(path.join(FX, 'title_tokens.json'), 'utf8'));
  assert.equal(titles.length, ref.length); assert.ok(titles.length > 14000);
  let bad = 0; titles.forEach((t, i) => { if (JSON.stringify(C.bertTokenize(t, vocab)) !== JSON.stringify(ref[i])) bad++; });
  assert.equal(bad, 0);
});
