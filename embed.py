"""Sentence embeddings with all-MiniLM-L6-v2 (int8 ONNX), matching what the browser computes.

Mean pooling over the attention mask, then L2 normalisation, as sentence-transformers does.
"""
import hashlib, os
import numpy as np

MODEL_FILES = {  # sha256 of the Xenova/all-MiniLM-L6-v2 exports (commit 751bff37) we ship
    "model_quantized.onnx": "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1",
    "tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
}
MAX_TOKENS = 128   # titles are short; the browser truncates at the same length
DIM = 384


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_dir(model_dir):
    for name, digest in MODEL_FILES.items():
        p = os.path.join(model_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing model file {p}")
        if sha256(p) != digest:
            raise ValueError(f"{name} does not match the pinned sha256; refusing to use it")


class Embedder:
    def __init__(self, model_dir):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        verify_model_dir(model_dir)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.no_padding()   # the shipped tokenizer pads to a fixed length; padding must not enter the mean
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1      # deterministic results
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model_quantized.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def tokenize(self, text):
        return self.tok.encode(text).ids

    def embed(self, texts):
        out = np.zeros((len(texts), DIM), np.float32)
        for i, t in enumerate(texts):
            ids = np.array([self.tokenize(t)], np.int64)
            mask = np.ones_like(ids)
            feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            h = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][0]
            v = h.mean(0)
            out[i] = v / np.linalg.norm(v)
        return out


def quantize(E):
    """Unit vectors -> int8 with one shared scale (tiny, and dot products stay comparable)."""
    scale = float(np.abs(E).max()) / 127.0
    return np.round(E / scale).astype(np.int8), scale
