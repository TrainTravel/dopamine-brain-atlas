"""Builds the hostable site (dist/) and the single-file artifact from one source.

Usage: python assemble.py <build_dir> [--out dist]
<build_dir> is what build_data.py wrote: data.json, plus search/ and model/ when search is enabled.
"""
import argparse, json, os, shutil

# 'wasm-unsafe-eval' lets the page compile the bundled language-model runtime (WebAssembly); it does not allow eval().
# connect-src 'self' lets the page fetch its own search files; no other host can be contacted.
CSP = ("default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'")
ORT_FILES = ["ort.wasm.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]
RESET = "html{-webkit-text-size-adjust:100%}body{margin:0}img{max-width:100%}[hidden]{display:none!important}"
CDN = ('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>\n'
       '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>\n')


def js_safe(json_text):
    """JSON embedded in a <script>: neutralise </script>, <!-- and U+2028/2029."""
    return (json_text.replace("</", "<\\/").replace("<!--", "<\\!--")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--three", default="node_modules/three")
    ap.add_argument("--ort", default="node_modules/onnxruntime-web")
    ap.add_argument("--out", default="dist")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    page = open(os.path.join(here, "src", "page.html"), encoding="utf-8").read()
    app = open(os.path.join(here, "src", "app.js"), encoding="utf-8").read()
    data = json.dumps(json.load(open(os.path.join(a.data, "data.json"), encoding="utf-8")), separators=(",", ":"), allow_nan=False)
    has_search = os.path.exists(os.path.join(a.data, "search", "meta.json"))
    data_js = "window.ATLAS_DATA=" + js_safe(data) + ";\n"

    split = page.index('<div class="app">')
    head, body = page[:split], page[split:]
    head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "").replace("<style>", "<style>\n" + RESET, 1)
    os.makedirs(os.path.join(a.out, "vendor"), exist_ok=True)
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<meta http-equiv="Content-Security-Policy" content="{CSP}">\n'
            '<meta name="description" content="Interactive 3D brain linking fMRI studies of autism, Alzheimer\'s, '
            'Parkinson\'s and ADHD to dopamine pathways and D1-D5 receptor territories.">\n'
            + head + "</head>\n<body>\n" + body +
            '<noscript><p style="padding:16px">This atlas needs JavaScript to draw the 3D brain.</p></noscript>\n'
            '<script src="vendor/three.min.js"></script>\n<script src="vendor/OrbitControls.js"></script>\n'
            '<script src="data.js"></script>\n<script src="app.js"></script>\n'
            + ('<script src="search-core.js"></script>\n<script src="search.js"></script>\n' if has_search else '')
            + '</body>\n</html>\n')
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(html)
    open(os.path.join(a.out, "app.js"), "w", encoding="utf-8").write(app)
    open(os.path.join(a.out, "data.js"), "w", encoding="utf-8").write(data_js)
    shutil.copy(os.path.join(a.three, "build", "three.min.js"), os.path.join(a.out, "vendor", "three.min.js"))
    shutil.copy(os.path.join(a.three, "examples", "js", "controls", "OrbitControls.js"), os.path.join(a.out, "vendor", "OrbitControls.js"))
    shutil.copy(os.path.join(a.three, "LICENSE"), os.path.join(a.out, "vendor", "THREE-LICENSE.txt"))
    if has_search:
        for sub in ("search", "model"):
            shutil.copytree(os.path.join(a.data, sub), os.path.join(a.out, sub), dirs_exist_ok=True)
        os.makedirs(os.path.join(a.out, "vendor", "ort"), exist_ok=True)
        for f in ORT_FILES:
            shutil.copy(os.path.join(a.ort, "dist", f), os.path.join(a.out, "vendor", "ort", f))
        with open(os.path.join(a.out, "vendor", "ort", "ONNXRUNTIME-NOTICE.txt"), "w") as f:
            f.write("ONNX Runtime Web (onnxruntime-web), Copyright (c) Microsoft Corporation. MIT License.\n"
                    "https://github.com/microsoft/onnxruntime\n")
        with open(os.path.join(a.out, "model", "MODEL-NOTICE.txt"), "w") as f:
            f.write("model_quantized.onnx and vocab.json are the int8 ONNX export and vocabulary of sentence-transformers/all-MiniLM-L6-v2\n"
                    "(Xenova/all-MiniLM-L6-v2 on Hugging Face), licensed under the Apache License 2.0. Redistributed unmodified.\n")
        for f in ("search-core.js", "search.js"):
            shutil.copy(os.path.join(here, "src", f), os.path.join(a.out, f))

    # single-file version (for a Claude artifact): CDN libraries, data and app inlined
    os.makedirs("artifact", exist_ok=True)
    single = page + CDN + "<script>\n" + data_js + app.replace("</script", "<\\/script") + "</script>\n"
    if has_search:  # the single-file page cannot host the 23 MB model, so it searches by keywords only
        core = open(os.path.join(here, "src", "search-core.js"), encoding="utf-8").read()
        ui = open(os.path.join(here, "src", "search.js"), encoding="utf-8").read()
        # the artifact host only serves text files, so peaks travel as base64 text and are decoded in the page
        pk = "get('search/peaks.bin')"
        assert ui.count(pk) == 1
        ui = ui.replace(pk, "get('search/peaks.b64.txt','b64')")
        ui = ui.replace("if (!onProgress || !r.body)", "if (type === 'b64') { const s = atob((await r.text()).trim()), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; }\n    if (!onProgress || !r.body)", 1)
        single += "<script>\nwindow.ATLAS_SEARCH_CONFIG={noModel:true};\n" + core + "\n" + ui + "</script>\n"
        os.makedirs(os.path.join("artifact", "search"), exist_ok=True)
        shutil.copy(os.path.join(a.data, "search", "meta.json"), os.path.join("artifact", "search", "meta.json"))
        import base64
        raw = open(os.path.join(a.data, "search", "peaks.bin"), "rb").read()
        open(os.path.join("artifact", "search", "peaks.b64.txt"), "w").write(base64.b64encode(raw).decode())
    open(os.path.join("artifact", "dopamine-disorder-atlas.html"), "w", encoding="utf-8").write(single)
    print("built", a.out, "and artifact/dopamine-disorder-atlas.html")


if __name__ == "__main__":
    main()
