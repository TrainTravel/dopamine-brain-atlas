// Browser tests for the built site. Run: node --test tests/   (ATLAS_DIST=dist by default)
// Serves dist/ over real HTTP so the Content-Security-Policy and relative script paths are exercised.
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const DIST = path.resolve(process.env.ATLAS_DIST || 'dist');
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.txt': 'text/plain',
  '.json': 'application/json', '.wasm': 'application/wasm', '.bin': 'application/octet-stream', '.onnx': 'application/octet-stream' };
const FX = path.resolve(process.env.ATLAS_FIXTURES || 'cache/fixtures');
const HAS_SEARCH = fs.existsSync(path.join(DIST, 'search', 'meta.json'));
const HAS_FX = fs.existsSync(path.join(FX, 'search_eval.json'));
let server, base, browser;

before(async () => {
  server = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]);
    const file = path.join(DIST, rel === '/' ? 'index.html' : rel);
    if (!file.startsWith(DIST) || !fs.existsSync(file)) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, { 'content-type': TYPES[path.extname(file)] || 'application/octet-stream', 'content-length': fs.statSync(file).size });
    fs.createReadStream(file).pipe(res);
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}/`;
  browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
});
after(async () => { await browser?.close(); server?.close(); });

// Opens the page; returns {page, errors, requests}. opts.setup(page) runs before navigation.
async function open(opts = {}) {
  const ctx = await browser.newContext({ viewport: opts.viewport || { width: 1440, height: 860 }, colorScheme: opts.colorScheme || 'light', hasTouch: !!opts.touch });
  const page = await ctx.newPage();
  const errors = [], requests = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('request', r => requests.push(r.url()));
  // Web fonts are optional; answer them locally so tests do not depend on the network.
  await page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  if (opts.setup) await opts.setup(page);
  await page.goto(base);
  if (!opts.noWait) await page.waitForFunction(() => window.__atlas && window.__atlas.three, null, { timeout: 20000 });
  return { page, errors, requests, close: () => ctx.close() };
}
const A = (page, fn, arg) => page.evaluate(fn, arg);

test('built files are complete and reference only local scripts', () => {
  for (const f of ['index.html', 'app.js', 'data.js', 'vendor/three.min.js', 'vendor/OrbitControls.js', 'vendor/THREE-LICENSE.txt'])
    assert.ok(fs.statSync(path.join(DIST, f)).size > 500, f + ' missing or empty');
  const html = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8');
  assert.match(html, /^<!doctype html>/i);
  assert.match(html, /<html lang="en">/);
  assert.match(html, /Content-Security-Policy/);
  const srcs = [...html.matchAll(/<script[^>]*src="([^"]+)"/g)].map(m => m[1]);
  assert.deepEqual(srcs, ['vendor/three.min.js', 'vendor/OrbitControls.js', 'data.js', 'app.js'].concat(HAS_SEARCH ? ['search-core.js', 'search.js'] : []));
  if (HAS_SEARCH) for (const f of ['search/meta.json', 'search/peaks.bin', 'search/emb.bin', 'model/model_quantized.onnx', 'model/vocab.json', 'model/MODEL-NOTICE.txt',
    'vendor/ort/ort.wasm.min.js', 'vendor/ort/ort-wasm-simd-threaded.wasm', 'vendor/ort/ort-wasm-simd-threaded.mjs']) assert.ok(fs.statSync(path.join(DIST, f)).size > 100, f + ' missing');
  assert.ok(fs.readdirSync(DIST, { recursive: true }).every(f => fs.statSync(path.join(DIST, f)).size < 100e6), 'GitHub Pages rejects files over 100 MB');
  assert.ok(!/<script(?![^>]*src=)[^>]*>/.test(html), 'no inline scripts allowed');
  assert.ok(!/\son[a-z]+="/.test(html), 'no inline event handlers allowed');
  assert.ok(fs.statSync(path.join(DIST, 'data.js')).size < 3e6, 'data.js unexpectedly large');
});

test('loads cleanly: no console errors, no CSP violations, no third-party requests', async () => {
  const { page, errors, requests, close } = await open();
  assert.equal(await page.title(), 'Dopamine Disorder Atlas');
  assert.deepEqual(errors, []);
  const foreign = requests.filter(u => !u.startsWith(base) && !/fonts\.(googleapis|gstatic)\.com/.test(u));
  assert.deepEqual(foreign, []);
  await close();
});

test('WebGL canvas actually draws the brain', async () => {
  const { page, close } = await open();
  const stats = await A(page, () => {
    const t = __atlas.three, gl = t.renderer.getContext(); t.render();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let drawn = 0; const colours = new Set();
    for (let i = 0; i < px.length; i += 4) if (px[i + 3] > 0) { drawn++; if (colours.size < 500) colours.add(px[i] >> 4 << 8 | px[i + 1] >> 4 << 4 | px[i + 2] >> 4); }
    return { drawn, total: w * h, colours: colours.size };
  });
  assert.ok(stats.drawn / stats.total > 0.05, `only ${stats.drawn} of ${stats.total} pixels drawn`);
  assert.ok(stats.colours > 20, 'image has too few colours to be the brain scene');
  await close();
});

test('scene contents match the data', async () => {
  const { page, close } = await open();
  const r = await A(page, () => {
    const { D, three: t } = __atlas, out = { layers: {}, expected: {} };
    D.keys.forEach((k, j) => {
      out.layers[k] = t.layers[k].geometry.attributes.position.count;
      out.expected[k] = D.studies.reduce((a, s) => a + (s[5] >> j & 1 ? s[7].length / 3 : 0), 0);
    });
    out.nodeMeshes = t.nodeMeshes.length;
    out.expectedNodes = D.nodes.reduce((a, n) => a + (n.mid ? 1 : 2), 0);
    out.labels = document.querySelectorAll('.lbl').length;
    out.paths = Object.fromEntries(D.paths.map(p => [p.id, t.pathGroups[p.id].children.length]));
    out.expectedPaths = Object.fromEntries(D.paths.map(p => [p.id, p.edges.reduce((a, [x, y]) => {
      const nx = D.nodes.find(n => n.id === x), ny = D.nodes.find(n => n.id === y); return a + ((nx.mid && ny.mid) ? 2 : 4); }, 0)]));
    out.hemiVerts = [t.hemi.L.geometry.attributes.position.count, t.hemi.R.geometry.attributes.position.count];
    return out;
  });
  assert.deepEqual(r.layers, r.expected);
  assert.equal(r.nodeMeshes, r.expectedNodes);
  assert.equal(r.labels, 15);
  assert.deepEqual(r.paths, r.expectedPaths);   // a tube and an arrowhead per edge per side
  assert.deepEqual(r.hemiVerts, [10242, 10242]);
  await close();
});

test('study layers: labels, default state and toggling', async () => {
  const { page, close } = await open();
  const rows = await page.locator('#layers label.row .ct').allTextContents();
  const D = await A(page, () => __atlas.D.keys.map(k => __atlas.D.groups[k].n));
  assert.deepEqual(rows.slice(0, 5), D.map(n => `${n} studies`));
  assert.deepEqual(await A(page, () => __atlas.D.keys.map(k => __atlas.three.layers[k].visible)), [true, true, true, true, false]);
  await page.uncheck('#ly_PD'); await page.check('#ly_DA');
  assert.deepEqual(await A(page, () => __atlas.D.keys.map(k => __atlas.three.layers[k].visible)), [true, true, false, true, true]);
  await close();
});

test('near-region filter limits the drawn peaks to the right ones', async () => {
  const { page, close } = await open();
  const r = await A(page, () => {
    const { D, three: t } = __atlas, R2 = D.radius ** 2, out = {};
    D.keys.forEach((k, j) => {
      let near = 0;
      for (const s of D.studies) if (s[5] >> j & 1) for (let q = 0; q < s[7].length; q += 3) {
        const [x, y, z] = s[7].slice(q, q + 3);
        if (D.nodes.some(n => (Math.abs(x) - n.xyz[0]) ** 2 + (y - n.xyz[1]) ** 2 + (z - n.xyz[2]) ** 2 <= R2)) near++;
      }
      // every drawn vertex within range, the first undrawn one out of range
      const pos = t.layers[k].geometry.attributes.position, nN = t.layers[k].userData.nearN;
      const isNear = i => D.nodes.some(n => (Math.abs(pos.getX(i)) - n.xyz[0]) ** 2 + (-pos.getZ(i) - n.xyz[1]) ** 2 + (pos.getY(i) - n.xyz[2]) ** 2 <= R2);
      let ordered = true; for (let i = 0; i < pos.count; i++) if (isNear(i) !== (i < nN)) { ordered = false; break; }
      out[k] = { near, nearN: nN, drawn: t.layers[k].geometry.drawRange.count, total: pos.count, ordered };
    });
    return out;
  });
  for (const k in r) { assert.equal(r[k].nearN, r[k].near, k); assert.equal(r[k].drawn, r[k].near, k); assert.ok(r[k].ordered, k); assert.ok(r[k].near > 0 && r[k].near < r[k].total); }
  await page.uncheck('#near');
  assert.ok(await A(page, () => __atlas.D.keys.every(k => __atlas.three.layers[k].geometry.drawRange.count === Infinity)));
  await close();
});

test('pathway toggles show and hide the tubes', async () => {
  const { page, close } = await open();
  const vis = () => A(page, () => Object.fromEntries(__atlas.D.paths.map(p => [p.id, __atlas.three.pathGroups[p.id].visible])));
  assert.deepEqual(await vis(), { mesolimbic: true, mesocortical: true, nigrostriatal: true, tubero: true, direct: false, indirect: false });
  await page.uncheck('#pa_mesolimbic'); await page.check('#pa_indirect');
  assert.deepEqual(await vis(), { mesolimbic: false, mesocortical: true, nigrostriatal: true, tubero: true, direct: false, indirect: true });
  await close();
});

test('receptor buttons resize regions by density', async () => {
  const { page, close } = await open();
  const btn = i => page.locator('#rseg button').nth(i);
  assert.deepEqual(await page.locator('#rseg button').allTextContents(), ['Off', 'D1', 'D2', 'D3', 'D4', 'D5']);
  for (let rx = 0; rx < 5; rx++) {
    await btn(rx + 1).click();
    assert.deepEqual(await page.locator('#rseg button').evaluateAll(b => b.map(x => x.getAttribute('aria-pressed'))), [0, 1, 2, 3, 4, 5].map(i => String(i === rx + 1)));
    assert.match(await page.locator('#rnote').textContent(), new RegExp('D' + (rx + 1)));
    const ok = await A(page, rx => {
      __atlas.S.node = null; __atlas.sync();
      const sc = __atlas.D.nodes.map(n => [n.expr[rx], n._m[0].scale.x, n._m[0].material.opacity]);
      return sc.every(([e, s, o]) => Math.abs(s - (1.4 + e * 1.5)) < 1e-6 && (e === 0 ? o < 1 : o === 1));
    }, rx);
    assert.ok(ok, 'D' + (rx + 1) + ' sizes do not follow expression');
  }
  await btn(0).click();
  assert.match(await page.locator('#rnote').textContent(), /Pick a receptor/);
  await close();
});

test('opens on the putamen with correct evidence', async () => {
  const { page, close } = await open();
  assert.equal(await page.locator('#detail h3').textContent(), 'Putamen');
  assert.equal(await page.locator('#detail .bar').count(), 5);
  assert.match(await page.locator('#detail .kv').textContent(), /±26, 4, 2/);
  const exp = await A(page, () => {
    const { D, S } = __atlas, ni = D.nodes.findIndex(n => n.id === 'PUT');
    const l = D.studies.filter(s => s[6] >> ni & 1 && D.keys.some((k, j) => S.layers[k] && s[5] >> j & 1));
    return { n: l.length, pd: D.nodes[ni].enr.PD };
  });
  const heads = await page.locator('#detail h2').allTextContents();
  assert.ok(heads.includes(`${exp.n} studies in the active layers`), heads.join(' | '));
  assert.equal(await page.locator('#detail ul.st li').count(), Math.min(60, exp.n));
  const pdBar = await page.locator('#detail .bar', { hasText: "Parkinson's" }).locator('.num').textContent();
  assert.equal(pdBar, `${exp.pd.pct}% · ${exp.pd.ratio.toFixed(1)}×`);
  const years = (await page.locator('#detail ul.st .m').allTextContents()).map(t => +t.match(/ (\d{4}) · /)[1]);
  assert.deepEqual(years, [...years].sort((a, b) => b - a), 'study list must be newest first');
  assert.equal(await A(page, () => __atlas.three.halo.filter(h => h.visible).length), 2);
  await close();
});

test('study list follows the active layers', async () => {
  const { page, close } = await open();
  for (const k of ['ASD', 'AD', 'ADHD']) await page.uncheck('#ly_' + k);
  const n = await A(page, () => { const { D } = __atlas; const ni = D.nodes.findIndex(n => n.id === 'PUT'); return D.nodes[ni].enr.PD.hit; });
  assert.ok((await page.locator('#detail h2').allTextContents()).includes(`${n} studies in the active layers`));
  await page.uncheck('#ly_PD');
  assert.ok((await page.locator('#detail h2').allTextContents()).includes('0 studies in the active layers'));
  assert.equal(await page.locator('#detail ul.st li').count(), 0);
  await close();
});

test('concentration table shows every ratio and selects regions', async () => {
  const { page, close } = await open();
  const exp = await A(page, () => __atlas.D.nodes.filter(n => n.base > 0).map(n => [n.id, ...__atlas.D.keys.map(k => n.enr[k].ratio.toFixed(1))]));
  const got = await page.locator('#mx tr[data-n]').evaluateAll(trs => trs.map(tr => [...tr.querySelectorAll('button')].map(b => b.textContent)));
  assert.deepEqual(got, exp);
  assert.equal(got.length, 14);
  const sigOK = await A(page, () => [...document.querySelectorAll('#mx tr[data-n]')].every(tr => { const n = __atlas.D.nodes.find(x => x.id === tr.dataset.n);
    return [...tr.querySelectorAll('td')].slice(1).every((td, j) => td.firstChild.classList.contains('sig') === (n.enr[__atlas.D.keys[j]].p < 0.05)); }));
  assert.ok(sigOK, 'outlined cells must be exactly those with p < 0.05');
  await page.locator('#mx tr[data-n="HIP"] td').nth(2).locator('button').click();
  assert.equal(await page.locator('#detail h3').textContent(), 'Hippocampus');
  assert.ok(await page.locator('#mx tr[data-n="HIP"]').evaluate(e => e.classList.contains('on')));
  await page.locator('#mx tr[data-n="HIP"] td').first().locator('button').click();   // same region again = deselect
  assert.ok(await page.locator('#detail').isHidden());
  await close();
});

test('selecting a study rings its peaks and links to PubMed safely', async () => {
  const { page, close } = await open();
  await page.locator('#detail ul.st li button').first().click();
  const s = await A(page, () => { const { D, S, three: t } = __atlas; return { id: D.studies[S.study][0], peaks: D.studies[S.study][7].length / 3, rings: t.hi.geometry.attributes.position.count, vis: t.hi.visible, node: S.node }; });
  assert.equal(s.rings, s.peaks); assert.ok(s.vis); assert.equal(s.node, 'PUT', 'region stays selected');
  assert.equal(await page.locator('#detail h2').first().textContent(), 'Selected study');
  const link = page.locator('#detail p.note a', { hasText: 'Open in PubMed' });
  assert.equal(await link.getAttribute('href'), `https://pubmed.ncbi.nlm.nih.gov/${s.id}/`);
  const bad = await page.locator('a[href^="http"]').evaluateAll(as => as.filter(a => a.target !== '_blank' || !/noopener/.test(a.rel) || !a.href.startsWith('https://')).map(a => a.href));
  assert.deepEqual(bad, [], 'external links must be https, new tab, rel=noopener');
  const hrefs = await page.locator('#detail ul.st a').evaluateAll(as => as.map(a => a.getAttribute('href')));
  assert.ok(hrefs.length > 0 && hrefs.every(h => /^https:\/\/pubmed\.ncbi\.nlm\.nih\.gov\/\d+\/$/.test(h)));
  await page.getByRole('button', { name: 'Clear selection' }).click();
  assert.ok(await page.locator('#detail').isHidden());
  assert.deepEqual(await A(page, () => [__atlas.three.hi.visible, __atlas.three.halo.some(h => h.visible)]), [false, false]);
  await close();
});

// screen position of a scene point, in page coordinates (camera matrices refreshed first: they
// otherwise update only on the next rendered frame, which would make these tests timing-dependent)
const project = (page, p) => A(page, p => { const t = __atlas.three; t.cam.updateMatrixWorld(); const v = new THREE.Vector3(p[0], p[1], p[2]).project(t.cam), r = t.renderer.domElement.getBoundingClientRect();
  return { x: r.left + (v.x + 1) / 2 * r.width, y: r.top + (1 - v.y) / 2 * r.height, z: v.z }; }, p);

test('clicking a region sphere in 3D selects it', async () => {
  const { page, close } = await open();
  await page.getByRole('button', { name: 'Clear selection' }).click();
  await page.getByRole('button', { name: 'Right side', exact: true }).click();
  for (const id of ['PUT', 'DLPFC', 'HIP']) {
    const pos = await A(page, id => { const m = __atlas.D.nodes.find(n => n.id === id)._m[0].position; return [m.x, m.y, m.z]; }, id);
    const s = await project(page, pos);
    await page.mouse.move(s.x, s.y);
    await page.waitForFunction(() => !document.querySelector('#hover').hidden);
    const name = await A(page, id => __atlas.D.nodes.find(n => n.id === id).name, id);
    assert.equal(await page.locator('#hover').textContent(), name);
    await page.mouse.click(s.x, s.y);
    assert.equal(await A(page, () => __atlas.S.node), id);
    assert.equal(await page.locator('#detail h3').textContent(), name);
  }
  await close();
});

test('clicking a dot in 3D opens a study from a visible layer', async () => {
  const { page, close } = await open();
  await page.getByRole('button', { name: 'Clear selection' }).click();
  for (const k of ['ASD', 'AD', 'ADHD']) await page.uncheck('#ly_' + k);
  await page.getByRole('button', { name: 'Right side', exact: true }).click();
  // choose drawn Parkinson's peaks that are well away from any region sphere on screen
  const targets = await A(page, () => {
    const t = __atlas.three; t.cam.updateMatrixWorld(); const L = t.layers.PD, pos = L.geometry.attributes.position, r = t.renderer.domElement.getBoundingClientRect();
    const scr = (x, y, z) => { const v = new THREE.Vector3(x, y, z).project(t.cam); return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height]; };
    const nodes = t.nodeMeshes.map(m => scr(m.position.x, m.position.y, m.position.z)), out = [];
    for (let i = 0; i < L.geometry.drawRange.count && out.length < 5; i += 7) {
      const p = scr(pos.getX(i), pos.getY(i), pos.getZ(i));
      if (nodes.every(n => Math.hypot(n[0] - p[0], n[1] - p[1]) > 26) && p[0] > r.left + 150 && p[0] < r.right - 20 && p[1] > r.top + 60 && p[1] < r.bottom - 40) out.push({ x: p[0], y: p[1], study: L.userData.idx[i] });
    }
    return out;
  });
  assert.ok(targets.length >= 3, 'not enough clickable test points');
  for (const tg of targets) {
    await A(page, () => { __atlas.S.study = null; __atlas.S.node = null; __atlas.sync(); });
    await page.mouse.click(tg.x, tg.y);
    const r = await A(page, () => { const { S, D } = __atlas; return S.study == null ? null : { bits: D.studies[S.study][5], node: S.node }; });
    assert.ok(r, 'click on a dot selected nothing');
    assert.ok(r.bits >> 2 & 1, "selected study must be in the only visible layer (Parkinson's)");
    assert.equal(r.node, null);
  }
  // a hidden layer can never be picked: with every layer off, clicking the same spots selects nothing
  await page.uncheck('#ly_PD');
  await A(page, () => { __atlas.S.study = null; __atlas.sync(); });
  await page.mouse.click(targets[0].x, targets[0].y);
  assert.equal(await A(page, () => __atlas.S.study), null);
  await close();
});

test('dragging rotates the brain without selecting anything', async () => {
  const { page, close } = await open();
  const before = await A(page, () => ({ cam: __atlas.three.cam.position.toArray(), node: __atlas.S.node, study: __atlas.S.study }));
  const box = await page.locator('#stage canvas').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + 140, box.y + box.height / 2 + 30, { steps: 8 }); await page.mouse.up();
  await page.waitForTimeout(300);
  const after = await A(page, () => ({ cam: __atlas.three.cam.position.toArray(), node: __atlas.S.node, study: __atlas.S.study }));
  assert.notDeepEqual(after.cam, before.cam);
  assert.equal(after.node, before.node); assert.equal(after.study, before.study);
  await close();
});

test('zoom is clamped and preset views report the right orientation', async () => {
  const { page, close } = await open();
  const box = await page.locator('#stage canvas').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  const dist = () => A(page, () => __atlas.three.cam.position.distanceTo(__atlas.three.ctl.target));
  for (let i = 0; i < 60; i++) await page.mouse.wheel(0, -400);
  await page.waitForTimeout(400); assert.ok(await dist() >= 79.9, 'zoomed inside the minimum distance');
  for (let i = 0; i < 120; i++) await page.mouse.wheel(0, 400);
  await page.waitForTimeout(400); assert.ok(await dist() <= 700.1, 'zoomed past the maximum distance');
  for (const [name, text] of [['Top', 'above'], ['Front', 'the front'], ['Right side', 'the right']]) {
    await page.getByRole('button', { name, exact: true }).click();
    await page.waitForFunction(t => document.querySelector('#orient').textContent === 'Viewing from ' + t, text);
  }
  await close();
});

test('anatomical orientation is not mirrored', async () => {
  const { page, close } = await open();
  await page.getByRole('button', { name: 'Front', exact: true }).click();
  await page.waitForTimeout(200);
  // seen from the front, the subject's right hemisphere is on the viewer's left (radiological convention falls out of true 3D)
  const r = await project(page, [40, 0, 0]), l = await project(page, [-40, 0, 0]), up = await project(page, [0, 60, 0]), dn = await project(page, [0, -40, 0]);
  assert.ok(r.x < l.x, 'right hemisphere should appear on screen-left from the front');
  assert.ok(up.y < dn.y, 'superior should be up');
  const hem = await A(page, () => { const m = h => { const p = __atlas.three.hemi[h].geometry.attributes.position; let s = 0; for (let i = 0; i < p.count; i++) s += p.getX(i); return s / p.count; }; return [m('L'), m('R')]; });
  assert.ok(hem[0] < -20 && hem[1] > 20);
  await close();
});

test('view controls: opacity, hemisphere, labels', async () => {
  const { page, close } = await open();
  await page.locator('#op').fill('0');
  assert.deepEqual(await A(page, () => [__atlas.three.hemi.L.visible, __atlas.three.hemi.R.visible]), [false, false]);
  await page.locator('#op').fill('40'); await page.check('#hideL');
  assert.deepEqual(await A(page, () => [__atlas.three.hemi.L.visible, __atlas.three.hemi.R.visible, __atlas.three.hemi.R.material.opacity]), [false, true, 0.4]);
  assert.equal(await page.locator('.lbl:visible').count(), 15);
  await page.uncheck('#lab');
  assert.equal(await page.locator('.lbl:visible').count(), 0);
  await close();
});

test('canvas follows its container when the window is resized', async () => {
  const { page, close } = await open();
  for (const vp of [{ width: 1100, height: 700 }, { width: 1600, height: 900 }, { width: 700, height: 900 }]) {
    await page.setViewportSize(vp); await page.waitForTimeout(250);
    const r = await A(page, () => { const s = document.querySelector('#stage'), c = s.querySelector('canvas'), t = __atlas.three;
      return { sw: s.clientWidth, sh: s.clientHeight, cw: c.clientWidth, ch: c.clientHeight, aspect: t.cam.aspect, over: document.documentElement.scrollWidth - innerWidth }; });
    assert.ok(Math.abs(r.cw - r.sw) <= 1 && Math.abs(r.ch - r.sh) <= 1, JSON.stringify(r));
    assert.ok(Math.abs(r.aspect - r.sw / r.sh) < 0.01);
    assert.ok(r.over <= 0, 'page scrolls sideways at ' + vp.width);
  }
  await close();
});

test('phone layout: stacked, no sideways scroll, touch works', async () => {
  const { page, errors, close } = await open({ viewport: { width: 390, height: 844 }, touch: true });
  const r = await A(page, () => { const b = s => document.querySelector(s).getBoundingClientRect();
    return { over: document.documentElement.scrollWidth - innerWidth, stage: b('#stage'), left: b('#left'), right: b('#right'), w: innerWidth }; });
  assert.ok(r.over <= 0, 'horizontal overflow of ' + r.over + 'px');
  assert.ok(r.stage.height >= 300 && r.stage.width >= 340);
  assert.ok(r.stage.top < r.left.top && r.left.bottom <= r.right.top + 1, 'expected brain, then controls, then evidence');
  for (const sel of ['#left', '#right']) assert.ok(r[sel.slice(1)].right <= r.w && r[sel.slice(1)].left >= 0);
  await page.locator('#ly_ASD').tap();
  assert.equal(await A(page, () => __atlas.three.layers.ASD.visible), false);
  await page.locator('#mx tr[data-n="AMY"] td').first().locator('button').tap();
  assert.equal(await page.locator('#detail h3').textContent(), 'Amygdala');
  assert.deepEqual(errors, []);
  await close();
});

test('light and dark themes both apply, and an explicit choice beats the system setting', async () => {
  const read = page => A(page, () => ({ bg: getComputedStyle(document.body).backgroundColor, fg: getComputedStyle(document.body).color, pd: '#' + __atlas.three.layers.PD.material.color.getHexString() }));
  const lum = c => { const [r, g, b] = c.match(/\d+/g).map(Number); return 0.299 * r + 0.587 * g + 0.114 * b; };
  const L = await open({ colorScheme: 'light' }), Dk = await open({ colorScheme: 'dark' });
  const l = await read(L.page), d = await read(Dk.page);
  assert.ok(lum(l.bg) > 200 && lum(l.fg) < 80, 'light theme colours'); assert.ok(lum(d.bg) < 40 && lum(d.fg) > 190, 'dark theme colours');
  assert.notEqual(l.pd, d.pd, '3D colours must follow the theme');
  await L.page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark')); await L.page.waitForTimeout(100);
  assert.deepEqual(await read(L.page), d, 'data-theme=dark must win over a light system setting');
  await Dk.page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light')); await Dk.page.waitForTimeout(100);
  assert.deepEqual(await read(Dk.page), l);
  await L.page.emulateMedia({ colorScheme: 'dark' }); await L.page.evaluate(() => document.documentElement.removeAttribute('data-theme')); await L.page.waitForTimeout(100);
  assert.deepEqual(await read(L.page), d, 'live system theme change must be picked up');
  await L.close(); await Dk.close();
});

test('text contrast meets WCAG AA in both themes', async () => {
  for (const scheme of ['light', 'dark']) {
    const { page, close } = await open({ colorScheme: scheme });
    const worst = await A(page, () => {
      const lin = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
      const L = c => { const [r, g, b] = c.match(/[\d.]+/g).map(Number); return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b); };
      const css = n => { const d = document.createElement('div'); d.style.color = `var(${n})`; document.body.append(d); const c = getComputedStyle(d).color; d.remove(); return c; };
      const ratio = (a, b) => { const x = L(css(a)), y = L(css(b)); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
      return { fgPanel: ratio('--fg', '--panel'), mutedPanel: ratio('--muted', '--panel'), mutedBg: ratio('--muted', '--bg'), pressed: ratio('--panel', '--da') };
    });
    for (const [k, v] of Object.entries(worst)) assert.ok(v >= 4.5, `${scheme} ${k} contrast ${v.toFixed(2)} < 4.5`);
    await close();
  }
});

test('controls are labelled and keyboard operable', async () => {
  const { page, close } = await open();
  const unlabelled = await A(page, () => [...document.querySelectorAll('input')].filter(i => !(i.closest('label') || document.querySelector(`label[for="${i.id}"]`))).map(i => i.id));
  assert.deepEqual(unlabelled, []);
  assert.deepEqual(await A(page, () => [...document.querySelectorAll('button')].filter(b => !b.textContent.trim()).length), 0);
  const ids = await A(page, () => [...document.querySelectorAll('[id]')].map(e => e.id));
  assert.equal(ids.length, new Set(ids).size, 'duplicate ids');
  await page.locator('#ly_ASD').focus(); await page.keyboard.press('Space');
  assert.equal(await A(page, () => __atlas.S.layers.ASD), false);
  await page.locator('#rseg button').nth(2).focus(); await page.keyboard.press('Enter');
  assert.equal(await A(page, () => __atlas.S.rx), 1);
  await page.locator('#mx tr[data-n="VTA"] button').first().focus(); await page.keyboard.press('Enter');
  assert.equal(await A(page, () => __atlas.S.node), 'VTA');
  const outline = await page.locator('#mx tr[data-n="VTA"] button').first().evaluate(e => getComputedStyle(e).outlineStyle);
  assert.notEqual(outline, 'none', 'keyboard focus must be visible');
  await close();
});

test('rapid interaction never throws', async () => {
  const { page, errors, close } = await open();
  await A(page, () => {
    const { D, S } = __atlas; let seed = 7; const rnd = n => (seed = seed * 16807 % 2147483647) % n;
    for (let i = 0; i < 400; i++) {
      const a = rnd(7);
      if (a === 0) __atlas.selectNode(D.nodes[rnd(D.nodes.length)].id);
      else if (a === 1) __atlas.selectStudy(rnd(D.studies.length));
      else if (a === 2) { const k = D.keys[rnd(5)]; S.layers[k] = !S.layers[k]; __atlas.sync(); }
      else if (a === 3) { S.rx = rnd(6) - 1; __atlas.sync(); }
      else if (a === 4) { const p = D.paths[rnd(D.paths.length)].id; S.paths[p] = !S.paths[p]; __atlas.sync(); }
      else if (a === 5) { S.node = null; S.study = null; __atlas.sync(); }
      else document.querySelector('#near').click();
    }
    __atlas.three.render();
  });
  assert.deepEqual(errors, []);
  for (const n of ['PIT', 'VTA']) { await A(page, n => { __atlas.S.node = n; __atlas.S.study = null; __atlas.sync(); }, n); assert.ok(await page.locator('#detail h3').isVisible()); }
  assert.match(await A(page, () => { __atlas.S.node = 'PIT'; __atlas.sync(); return document.querySelector('#detail').textContent; }), /outside the standard fMRI brain mask/);
  await close();
});

// ---------- failure modes ----------
test('without WebGL the page degrades to tables instead of breaking', async () => {
  const { page, errors, close } = await open({ noWait: true, setup: p => p.addInitScript(() => {
    const orig = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...a) { return /webgl/i.test(type) ? null : orig.call(this, type, ...a); };
  }) });
  await page.waitForSelector('#stage .err');
  assert.match(await page.locator('#stage .err').textContent(), /3D view could not start/);
  assert.equal(await page.locator('#mx tr[data-n]').count(), 14);
  assert.equal(await page.locator('#detail h3').textContent(), 'Putamen');
  await page.locator('#mx tr[data-n="HIP"] button').first().click();
  assert.equal(await page.locator('#detail h3').textContent(), 'Hippocampus');
  await page.uncheck('#ly_ASD'); await page.locator('#rseg button').nth(1).click();
  assert.deepEqual(errors.filter(e => e.startsWith('pageerror')), []);
  await close();
});

test('if the 3D library fails to download the tables still work', async () => {
  for (const blocked of ['**/vendor/three.min.js', '**/vendor/OrbitControls.js']) {
    const { page, errors, close } = await open({ noWait: true, setup: p => p.route(blocked, r => r.abort()) });
    await page.waitForSelector('#stage .err');
    assert.equal(await page.locator('#mx tr[data-n]').count(), 14);
    await page.locator('#mx tr[data-n="AMY"] button').first().click();
    assert.equal(await page.locator('#detail h3').textContent(), 'Amygdala');
    // the add-on script itself reports the missing library; nothing else may throw
    assert.deepEqual(errors.filter(e => e.startsWith('pageerror') && !/THREE is not defined/.test(e)), []);
    await close();
  }
});

test('if the data file fails to download the visitor is told', async () => {
  const { page, close } = await open({ noWait: true, setup: p => p.route('**/data.js', r => r.abort()) });
  await page.waitForSelector('#dataerr');
  assert.match(await page.locator('#dataerr').textContent(), /data did not load/);
  await close();
});

test('hostile text in study metadata is displayed, never executed', async () => {
  const raw = fs.readFileSync(path.join(DIST, 'data.js'), 'utf8');
  const D = JSON.parse(raw.slice(raw.indexOf('=') + 1, raw.lastIndexOf(';')));
  const evil = '<img src=x onerror="window.__xss=1"><script>window.__xss=2</scr' + 'ipt>"\'&';
  const ni = D.nodes.findIndex(n => n.id === 'PUT');
  let changed = 0;
  for (const s of D.studies) if (s[6] >> ni & 1) { s[1] = evil + s[1]; s[3] = evil; s[4] = evil; changed++; }
  assert.ok(changed > 50);
  const body = 'window.ATLAS_DATA=' + JSON.stringify(D).replace(/<\//g, '<\\/') + ';';
  const { page, errors, close } = await open({ setup: p => p.route('**/data.js', r => r.fulfill({ contentType: 'text/javascript', body })) });
  await page.locator('#detail ul.st li button').first().click();
  await page.waitForTimeout(300);
  assert.equal(await A(page, () => window.__xss), undefined);
  assert.equal(await page.locator('#right img, #right script').count(), 0);
  assert.ok((await page.locator('#detail h3').first().textContent()).startsWith('<img src=x onerror='), 'payload should appear as literal text');
  assert.deepEqual(errors, []);
  await close();
});

test('content-security policy blocks injected inline scripts and outside requests', async () => {
  const { page, close } = await open();
  const r = await A(page, async () => {
    const violations = []; document.addEventListener('securitypolicyviolation', e => violations.push(e.violatedDirective));
    const s = document.createElement('script'); s.textContent = 'window.__inline=1'; document.body.append(s);
    const d = document.createElement('div'); d.innerHTML = '<img src="https://example.com/x.png" onerror="window.__handler=1">'; document.body.append(d);
    let fetched = 'blocked'; try { await fetch('https://example.com/'); fetched = 'allowed'; } catch (e) {}
    await new Promise(r => setTimeout(r, 300));
    return { inline: window.__inline, handler: window.__handler, fetched, violations };
  });
  assert.equal(r.inline, undefined); assert.equal(r.handler, undefined); assert.equal(r.fetched, 'blocked');
  assert.ok(r.violations.length >= 2, 'expected CSP violations to be reported: ' + r.violations);
  await close();
});

// ======================= search =======================
const S = { skip: !HAS_SEARCH && 'site built without search' };
const SF = { skip: !(HAS_SEARCH && HAS_FX) && 'search build or fixtures not present' };
const doSearch = async (page, text, mode = 'meaning') => {
  await page.fill('#q', text); await page.click('#sgo');
  await page.waitForFunction(([m, t]) => { const X = window.__atlasSearch.st, want = t.replace(/\s+/g, ' ').trim().slice(0, 300); return X.query === want && X.mode === m; }, [mode, text], { timeout: 90000 });
};
const resultTitles = page => page.locator('#results > ul.st .t').allTextContents();
const noModel = p => p.route(/model_quantized\.onnx/, r => r.abort());

test('search bar: present, labelled, keyword results appear without the model', S, async () => {
  const { page, errors, requests, close } = await open({ setup: noModel });
  assert.ok(await page.locator('#sform').isVisible());
  assert.equal(await page.locator('label[for="q"]').textContent(), 'Search fMRI studies');
  assert.ok(!requests.some(u => /search\/|model\/|ort/.test(u)), 'search files must not load before the visitor uses search');
  await doSearch(page, 'sleep deprivation', 'keyword');
  await page.waitForFunction(() => /could not load/.test(document.querySelector('#sstatus').textContent));
  const titles = await resultTitles(page);
  assert.equal(titles.length, 20); assert.ok(titles.every(t => /sleep|depriv/i.test(t)), 'keyword results must contain the query words');
  assert.match(await page.locator('#results p.note').first().textContent(), /Top 20 of 14,360 studies for “sleep deprivation”, ranked by matching words/);
  assert.deepEqual(errors.filter(e => e.startsWith('pageerror')), []);
  await close();
});

test('search: browser embeddings match the Python build (parity)', SF, async () => {
  const F = JSON.parse(fs.readFileSync(path.join(FX, 'search_eval.json'), 'utf8'));
  const { page, errors, close } = await open();
  const cos = await A(page, async texts => { const X = await window.__atlasSearch.loadSemantic(), out = []; for (const t of texts) out.push(Array.from(await X.embed(t))); return out; }, F.parity.map(p => p.text));
  F.parity.forEach((p, i) => {
    let d = 0, n = 0; for (let k = 0; k < 384; k++) { d += cos[i][k] * p.vec[k]; n += cos[i][k] ** 2; }
    assert.ok(Math.abs(n - 1) < 1e-3, 'browser vector not unit length');
    assert.ok(d > 0.995, `parity ${d.toFixed(4)} for ${JSON.stringify(p.text.slice(0, 40))}`);
  });
  assert.deepEqual(errors, []);
  await close();
});

test('search: the 88-query benchmark, run through the shipped browser code', SF, async () => {
  const F = JSON.parse(fs.readFileSync(path.join(FX, 'search_eval.json'), 'utf8'));
  const { page, close } = await open();
  const r = await A(page, async ev => {
    const X = window.__atlasSearch, C = window.AtlasSearch, B = await X.loadBase(), Sm = await X.loadSemantic(), out = [];
    const ndcg = (sc, rel, positive) => { const top = C.topK(sc, 10, positive), R = new Set(rel); let d = 0, ideal = 0; top.forEach((j, k) => { if (R.has(j)) d += 1 / Math.log2(k + 2); });
      for (let k = 0; k < Math.min(10, rel.length); k++) ideal += 1 / Math.log2(k + 2); return ideal ? d / ideal : 0; };
    for (const e of ev) out.push({ set: e.set, kw: ndcg(C.bm25(B.index, C.concepts(e.query)), e.relevant, true), sem: ndcg(C.semantic(Sm.emb, B.meta.n, B.meta.dim, await Sm.embed(e.query), B.meta.scale), e.relevant, false) });
    return out;
  }, F.eval);
  const mean = (set, k) => { const v = r.filter(x => x.set === set).map(x => x[k]); return v.reduce((a, b) => a + b, 0) / v.length; };
  for (const [set, key] of [['MeSH', 'MeSH: all 56'], ['Terms', 'Terms: all 32']]) {
    const sem = mean(set, 'sem'), kw = mean(set, 'kw'), py = F.pySummary[key];
    assert.ok(Math.abs(sem - py.semantic) < 0.02, `${set}: browser semantic nDCG ${sem.toFixed(3)} vs Python ${py.semantic.toFixed(3)}`);
    assert.ok(Math.abs(kw - py['keyword+synonyms']) < 0.04, `${set}: browser keyword nDCG ${kw.toFixed(3)} vs Python ${py['keyword+synonyms'].toFixed(3)}`);
    assert.ok(sem - kw > 0.08, `${set}: meaning should beat keywords (${sem.toFixed(3)} vs ${kw.toFixed(3)})`);
  }
  await close();
});

test('search: results re-rank by meaning; paraphrases find the right studies', S, async () => {
  const { page, errors, close } = await open();
  await doSearch(page, 'staying awake all night');
  assert.equal(await page.locator('#sstatus').textContent(), 'Ranked by meaning.');
  const t = await resultTitles(page);
  assert.equal(t.length, 20); assert.ok(t.filter(x => /sleep|awake|wake|circadian|night/i.test(x)).length >= 12, 'expected mostly sleep studies: ' + t.slice(0, 5).join(' | '));
  await doSearch(page, 'levodopa in Parkinson\'s disease');
  assert.ok((await resultTitles(page)).slice(0, 10).filter(x => /parkinson|levodopa|\bPD\b/i.test(x)).length >= 9);
  assert.deepEqual(errors, []);
  await close();
});

test('search: the mindfulness example shows child-ADHD studies and flags the missed word', S, async () => {
  const { page, close } = await open();
  await doSearch(page, 'mindful practices for ADHD kids');
  assert.ok((await resultTitles(page)).slice(0, 10).filter(x => /adhd|attention.deficit/i.test(x)).length >= 8);
  const miss = page.locator('#results .miss');
  assert.equal(await miss.count(), 1, 'exactly one missed-word panel (not one for "practices")');
  assert.equal(await miss.getAttribute('data-word'), 'mindful');
  assert.match(await miss.locator('p').textContent(), /None of the top results mention “mindful”\. \d+ studies do/);
  const mt = await miss.locator('.t').allTextContents();
  assert.ok(mt.length === 5 && mt.every(x => /mindful|meditat/i.test(x)));
  assert.ok(mt.some(x => /possible implications for disorders of attention/.test(x)), 'the meditation-and-attention paper should be listed');
  await doSearch(page, 'staying awake all night');
  assert.equal(await page.locator('#results .miss').count(), 0, 'everyday words must not be flagged');
  await doSearch(page, 'zzzyqx brain connectivity');
  assert.equal(await page.locator('#absent').textContent(), 'No study title contains: zzzyqx.');
  await close();
});

test('search: selecting results, brain dots, and clearing', S, async () => {
  const { page, errors, close } = await open();
  await doSearch(page, 'staying awake all night');
  const st = await A(page, () => { const X = window.__atlasSearch.st, t = __atlas.three; return { peaks: X.hits.reduce((a, h) => a + h.peaks.length / 3, 0), dots: t.searchPts.geometry.attributes.position.count, vis: t.searchPts.visible,
    inAtlas: X.hits.map(h => __atlas.D.studies.some(s => s[0] === h.pmid)) }; });
  assert.equal(st.dots, st.peaks); assert.ok(st.vis && st.dots > 50);
  const outside = st.inAtlas.indexOf(false); assert.ok(outside >= 0, 'expected at least one result outside the disorder layers');
  await page.locator('#results > ul.st li button').nth(outside).click();
  assert.match(await page.locator('#detail').textContent(), /not in the five study layers \(found by search\)/);
  const h = await A(page, () => ({ pmid: __atlas.S.hit.pmid, n: __atlas.S.hit.peaks.length / 3, rings: __atlas.three.hi.geometry.attributes.position.count, vis: __atlas.three.hi.visible }));
  assert.equal(h.rings, h.n); assert.ok(h.vis);
  assert.equal(await page.locator('#detail a', { hasText: 'Open in PubMed' }).getAttribute('href'), `https://pubmed.ncbi.nlm.nih.gov/${h.pmid}/`);
  await doSearch(page, 'methylphenidate in children with ADHD');                 // these are in the ADHD layer
  await page.locator('#results > ul.st li button').first().click();
  assert.equal(await A(page, () => __atlas.S.hit), null); assert.notEqual(await A(page, () => __atlas.S.study), null);
  assert.match(await page.locator('#detail').textContent(), /Layers: .*ADHD/);
  // click one of the large search dots in 3D
  await A(page, () => { __atlas.S.study = null; __atlas.S.node = null; __atlas.sync(); for (const k of __atlas.D.keys) { __atlas.S.layers[k] = false; } __atlas.sync(); });
  await page.getByRole('button', { name: 'Right side', exact: true }).click();
  const tg = await A(page, () => { const t = __atlas.three; t.cam.updateMatrixWorld(); const pos = t.searchPts.geometry.attributes.position, r = t.renderer.domElement.getBoundingClientRect();
    const scr = (x, y, z) => { const v = new THREE.Vector3(x, y, z).project(t.cam); return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height]; }, nodes = t.nodeMeshes.map(m => scr(m.position.x, m.position.y, m.position.z));
    for (let i = 0; i < pos.count; i++) { const p = scr(pos.getX(i), pos.getY(i), pos.getZ(i)); if (nodes.every(n => Math.hypot(n[0] - p[0], n[1] - p[1]) > 26) && p[0] > r.left + 160 && p[0] < r.right - 20 && p[1] > r.top + 60 && p[1] < r.bottom - 40) return { x: p[0], y: p[1] }; } });
  assert.ok(tg, 'no clickable search dot found'); await page.mouse.click(tg.x, tg.y);
  assert.ok(await A(page, () => __atlas.S.study != null || __atlas.S.hit != null), 'clicking a search dot should open its study');
  assert.equal(await page.locator('#results li.on').count(), 1);
  await page.getByRole('button', { name: 'Clear search' }).click();
  assert.ok(await page.locator('#results').isHidden());
  assert.deepEqual(await A(page, () => [__atlas.three.searchPts.visible, window.__atlasSearch.st.hits.length, document.querySelector('#sstatus').textContent]), [false, 0, '']);
  assert.deepEqual(errors, []);
  await close();
});

test('search: awkward input is handled and never executed', S, async () => {
  const { page, errors, close } = await open();
  await doSearch(page, '<img src=x onerror="window.__xss=1"> <script>window.__xss=2</script> attention');
  assert.equal(await A(page, () => window.__xss), undefined);
  assert.equal(await page.locator('#results img, #results script').count(), 0);
  assert.ok((await page.locator('#results p.note').first().textContent()).includes('<img src=x onerror='), 'query should be shown as literal text');
  await page.fill('#q', '   '); await page.click('#sgo'); await page.waitForFunction(() => document.querySelector('#results').hidden);
  for (const qy of ['日本語 naïve café 🧠', '?!…—', 'the of and', 'a']) { await doSearch(page, qy); assert.ok(await page.locator('#results h2').isVisible(), qy); }
  await A(page, () => { document.querySelector('#q').removeAttribute('maxlength'); });
  await doSearch(page, 'working memory '.repeat(400));                                   // 6,000 characters: trimmed to 300, tokens truncated
  assert.equal(await A(page, () => window.__atlasSearch.st.query.length), 300);
  assert.equal((await resultTitles(page)).length, 20);
  assert.deepEqual(errors, []);
  await close();
});

test('search: "Match by meaning" toggle, saved preference, and data-saver default', S, async () => {
  const { page, requests, close } = await open();
  await page.uncheck('#sem'); await doSearch(page, 'sleep deprivation', 'keyword');
  assert.equal(await page.locator('#sstatus').textContent(), 'Ranked by matching words.');
  assert.ok(!requests.some(u => /onnx|ort-wasm/.test(u)), 'the model must not download when meaning search is off');
  await page.reload(); await page.waitForFunction(() => window.__atlasSearch);
  assert.equal(await page.isChecked('#sem'), false, 'preference should persist across reloads');
  await page.check('#sem'); await doSearch(page, 'sleep deprivation', 'meaning');
  assert.ok(requests.some(u => /model_quantized\.onnx/.test(u)));
  await close();
  const ds = await open({ setup: p => p.addInitScript(() => Object.defineProperty(navigator, 'connection', { value: { saveData: true }, configurable: true })) });
  assert.equal(await ds.page.isChecked('#sem'), false, 'data-saver visitors should not auto-download 38 MB');
  await ds.close();
});

test('search: failures are reported, and a newer search always wins', S, async () => {
  const bad = await open({ setup: p => p.route('**/search/meta.json', r => r.abort()) });
  await bad.page.fill('#q', 'sleep'); await bad.page.click('#sgo');
  await bad.page.waitForFunction(() => /study index did not load/.test(document.querySelector('#sstatus').textContent));
  assert.ok(await bad.page.locator('#results').isHidden());
  assert.equal(await bad.page.locator('#detail h3').textContent(), 'Putamen', 'the rest of the atlas keeps working');
  assert.deepEqual(bad.errors.filter(e => e.startsWith('pageerror')), []);
  await bad.close();
  for (const blocked of [/ort-wasm-simd-threaded\.wasm/, /search\/emb\.bin/, /model\/vocab\.json/]) {
    const f = await open({ setup: p => p.route(blocked, r => r.abort()) });
    await doSearch(f.page, 'sleep deprivation', 'keyword');
    await f.page.waitForFunction(() => /could not load/.test(document.querySelector('#sstatus').textContent), null, { timeout: 60000 });
    assert.equal((await resultTitles(f.page)).length, 20);
    assert.deepEqual(f.errors.filter(e => e.startsWith('pageerror')), [], String(blocked));
    await f.close();
  }
  // Force the OLDER search to finish LAST: its embedding is delayed, so its results arrive after the newer search's.
  const { page, close } = await open();
  await A(page, async () => { const Sm = await window.__atlasSearch.loadSemantic(), orig = Sm.embed;
    Sm.embed = async t => { if (t === 'gambling') await new Promise(r => setTimeout(r, 900)); return orig(t); }; });
  await A(page, async () => { window.__atlasSearch.search('gambling'); await new Promise(r => setTimeout(r, 150));   // 'gambling' is now inside its slow embedding
    window.__atlasSearch.search('oxytocin'); });
  await page.waitForTimeout(2500);
  assert.equal(await A(page, () => window.__atlasSearch.st.query), 'oxytocin');
  assert.ok((await resultTitles(page)).slice(0, 5).some(x => /oxytocin/i.test(x)));
  assert.match(await page.locator('#results p.note').first().textContent(), /“oxytocin”/);
  await close();
});

test('search: works on a phone and without WebGL', S, async () => {
  const ph = await open({ viewport: { width: 390, height: 844 }, touch: true });
  assert.ok(await ph.page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'search bar overflows on a phone');
  await doSearch(ph.page, 'staying awake all night');
  assert.ok(await ph.page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await ph.page.locator('#results > ul.st li button').first().tap();
  assert.ok(await ph.page.locator('#detail h3').isVisible());
  assert.deepEqual(ph.errors, []);
  await ph.close();
  const ng = await open({ noWait: true, setup: p => p.addInitScript(() => { const o = HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext = function (t, ...a) { return /webgl/i.test(t) ? null : o.call(this, t, ...a); }; }) });
  await ng.page.waitForSelector('#stage .err'); await ng.page.waitForFunction(() => window.__atlasSearch);
  await doSearch(ng.page, 'sleep deprivation', 'keyword'); assert.equal((await resultTitles(ng.page)).length, 20);
  await ng.page.locator('#results > ul.st li button').first().click(); assert.ok(await ng.page.locator('#detail h3').isVisible());
  assert.deepEqual(ng.errors.filter(e => e.startsWith('pageerror')), []);
  await ng.close();
});
