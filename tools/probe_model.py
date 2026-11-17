"""Sanity-check the embedding model: int8 vs fp32 agreement and basic semantic behaviour.
Usage: python -I tools/probe_model.py <model_dir>   (contains model_quantized.onnx, model_fp32.onnx, tokenizer.json)
"""
import sys
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

d = sys.argv[1].rstrip("/") + "/"
tok = Tokenizer.from_file(d + "tokenizer.json")
tok.enable_truncation(256)


def session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def embed(sess, texts):
    names = {i.name for i in sess.get_inputs()}
    out = []
    for t in texts:
        e = tok.encode(t)
        ids = np.array([e.ids], np.int64)
        mask = np.array([e.attention_mask], np.int64)
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        h = sess.run(None, {k: v for k, v in feed.items() if k in names})[0][0]
        v = (h * mask[0][:, None]).sum(0) / mask.sum()
        out.append(v / np.linalg.norm(v))
    return np.array(out)


q8, f32 = session(d + "model_quantized.onnx"), session(d + "model_fp32.onnx")
print("inputs", [(i.name, i.shape) for i in q8.get_inputs()], "outputs", [o.name for o in q8.get_outputs()])
T = ["mindful practices for ADHD kids",
     "meditation training in children with attention deficit hyperactivity disorder",
     "dopamine release in the striatum of Parkinson's disease patients",
     "hippocampal atrophy in Alzheimer's disease",
     "kids", "children", "hippocampus",
     "A man is eating food.", "A man is eating a piece of bread.", "The girl is carrying a baby."]
a, b = embed(q8, T), embed(f32, T)
agree = (a * b).sum(1)
print("int8 vs fp32 cosine: min %.4f mean %.4f" % (agree.min(), agree.mean()))
S = a @ a.T
print("ADHD-kids query vs meditation/ADHD/children  %.3f" % S[0, 1])
print("ADHD-kids query vs Parkinson/dopamine         %.3f" % S[0, 2])
print("ADHD-kids query vs Alzheimer/hippocampus      %.3f" % S[0, 3])
print("kids~children %.3f   kids~hippocampus %.3f" % (S[4, 5], S[4, 6]))
print("eating food~bread %.3f   eating food~girl/baby %.3f" % (S[7, 8], S[7, 9]))
