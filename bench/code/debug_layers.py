#!/usr/bin/env python3
"""Layer-by-layer numpy reference vs Keras model to isolate the mismatch."""
import sys
sys.path.insert(0, "code")
import numpy as np
from rebuild_model import load_tflite_tensors, dequant, find
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, AveragePooling2D, Flatten, Dense, ReLU
from tensorflow.keras import Model

ts = load_tflite_tensors("models/kws_ref_model_float32.tflite")

def conv2d_np(x, w, b, stride, pad_same=True):
    """x: [1,H,W,C], w: [kh,kw,Cin,Cout]"""
    n, h, wd, c = x.shape
    kh, kw, cin, cout = w.shape
    sh, sw = stride
    if pad_same:
        ph = max((((h - 1) // sh) * sh + kh - h), 0)
        pw = max((((wd - 1) // sw) * sw + kw - wd), 0)
        ph0, ph1 = ph // 2, ph - ph // 2
        pw0, pw1 = pw // 2, pw - pw // 2
        x = np.pad(x, ((0, 0), (ph0, ph1), (pw0, pw1), (0, 0)))
    oh = (h + ph0 + ph1 - kh) // sh + 1
    ow = (wd + pw0 + pw1 - kw) // sw + 1
    out = np.zeros((n, oh, ow, cout))
    for i in range(oh):
        for j in range(ow):
            patch = x[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]  # [1,kh,kw,C]
            out[:, i, j, :] = np.einsum("nhwc,hwcf->f", patch, w) + b
    return out

def depthwise_np(x, w, b, stride=1):
    """w: [kh,kw,Cin,mult] -> TFLite [1,kh,kw,Cin*mult]"""
    n, h, wd, c = x.shape
    kh, kw, cin, mult = w.shape
    ph = (kh - 1) // 2
    pw = (kw - 1) // 2
    x = np.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    out = np.zeros((n, h, wd, c * mult))
    for i in range(h):
        for j in range(wd):
            patch = x[:, i:i+kh, j:j+kw, :]  # [1,kh,kw,c]
            for cc in range(c):
                patch_c = patch[:, :, :, cc]  # [1, kh, kw]
                out[:, i, j, cc] = np.einsum("nhk,hk->n", patch_c, w[:, :, cc, 0]) + b[cc]
    return out

np.random.seed(42)
data = np.random.randn(1, 49, 10, 1).astype(np.float32)

# --- numpy forward ---
k1 = dequant(find(ts, "functional_1/conv2d/Conv2D"))           # [64,10,4,1]
b1 = dequant(find(ts, "functional_1/activation/Relu;functional_1/batch_normalization/FusedBatchNormV3"))
a = conv2d_np(data, np.transpose(k1, (1, 2, 3, 0)), b1, (2, 2))
a = np.maximum(a, 0)
print("conv1 out shape:", a.shape)

for blk in range(4):
    dk = dequant(find(ts, f"functional_1/batch_normalization_{2*blk+1}/FusedBatchNormV3;functional_1/d"))
    db = dequant(find(ts, f"functional_1/activation_{2*blk+1}/Relu;functional_1/batch_normalization_{2*blk+1}/Fused"))
    a = depthwise_np(a, np.transpose(dk, (1, 2, 3, 0)), db)
    a = np.maximum(a, 0)
    ck = dequant(find(ts, f"functional_1/conv2d_{blk+1}/Conv2D"))
    cb = dequant(find(ts, f"functional_1/activation_{2*blk+2}/Relu;functional_1/batch_normalization_{2*blk+2}/Fused"))
    a = conv2d_np(a, np.transpose(ck, (1, 2, 3, 0)), cb, (1, 1))
    a = np.maximum(a, 0)
print("after blocks:", a.shape)
# avg pool (24,5)
a = a[:, :1, :1, :]  # pool (24,5) on (25,5) with stride=pool -> (1,1,64)
w_dense = dequant(find(ts, "functional_1/dense/MatMul"))
b_dense = dequant(find(ts, "functional_1/dense/BiasAdd/ReadVariableOp/resource"))
logits_np = a.reshape(1, -1) @ np.transpose(w_dense, (1, 0)) + b_dense
print("numpy logits:", np.round(logits_np[0], 4))

# --- Keras model ---
inputs = Input(shape=(49, 10, 1))
conv1 = Conv2D(64, (10, 4), strides=(2, 2), padding="same")
x = conv1(inputs); conv1.set_weights([np.transpose(k1, (1, 2, 3, 0)), b1]); x = ReLU()(x)
for blk in range(4):
    dw = DepthwiseConv2D((3, 3), padding="same")
    dk = dequant(find(ts, f"functional_1/batch_normalization_{2*blk+1}/FusedBatchNormV3;functional_1/d"))
    db = dequant(find(ts, f"functional_1/activation_{2*blk+1}/Relu;functional_1/batch_normalization_{2*blk+1}/Fused"))
    x = dw(x); dw.set_weights([np.transpose(dk, (1, 2, 3, 0)), db]); x = ReLU()(x)
    c1 = Conv2D(64, (1, 1), padding="same")
    ck = dequant(find(ts, f"functional_1/conv2d_{blk+1}/Conv2D"))
    cb = dequant(find(ts, f"functional_1/activation_{2*blk+2}/Relu;functional_1/batch_normalization_{2*blk+2}/Fused"))
    x = c1(x); c1.set_weights([np.transpose(ck, (1, 2, 3, 0)), cb]); x = ReLU()(x)
x = AveragePooling2D(pool_size=(24, 5))(x)
x = Flatten()(x)
dense = Dense(12)
x = dense(x); dense.set_weights([np.transpose(w_dense, (1, 0)), b_dense])
model = Model(inputs, x)
logits_keras = model.predict(data, verbose=0)
print("keras logits: ", np.round(logits_keras[0], 4))
print("numpy vs keras max diff:", float(np.max(np.abs(logits_np - logits_keras))))

# --- TFLite ---
import ai_edge_litert.interpreter as tfl
interp = tfl.Interpreter(model_path="models/kws_ref_model_float32.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]
interp.set_tensor(inp["index"], data)
interp.invoke()
tflite_out = interp.get_tensor(interp.get_output_details()[0]["index"])
print("tflite logits: ", np.round(tflite_out[0], 4))
print("numpy vs tflite max diff:", float(np.max(np.abs(logits_np - tflite_out))))
