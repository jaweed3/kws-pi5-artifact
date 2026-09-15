#!/usr/bin/env python3
"""Rebuild the MLPerf Tiny KWS model as a TRUE float32 Keras model.

The published kws_ref_model_float32.tflite is actually MIXED precision:
conv kernels are int8 (quantized), depthwise/dense are float32.
We extract all weights from the TFLite flatbuffer (dequantizing the int8
kernels), rebuild the Keras functional model (BN already fused into
kernel/bias by the converter), verify outputs match the TFLite runtime,
then export:
  - models/kws_ref_model_fp32.tflite   (true float32)
  - models/kws_ref_model_fp32.onnx     (tf2onnx)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D,
                                     AveragePooling2D, Flatten, Dense, ReLU)
from tensorflow.keras import Model
import ai_edge_litert.interpreter as tfl
import tf2onnx
import onnx

SRC = "models/kws_ref_model_float32.tflite"
OUT_TFLITE = "models/kws_ref_model_fp32.tflite"
OUT_ONNX = "models/kws_ref_model_fp32.onnx"


def load_tflite_tensors(path):
    interp = tfl.Interpreter(model_path=path)
    interp.allocate_tensors()
    out = {}
    for d in interp.get_tensor_details():
        try:
            t = interp.get_tensor(d["index"])
        except Exception:
            continue
        if t is None:
            continue
        q = d.get("quantization")
        scale, zp = 1.0, 0
        if q is not None:
            # ai-edge-litert returns QuantizationParameters; handle both shapes
            try:
                scale = float(q.scale) if q.scale is not None else 1.0
                zp = float(q.zero_point) if q.zero_point is not None else 0.0
            except Exception:
                try:
                    scale, zp = float(q[0]), float(q[1])
                except Exception:
                    scale, zp = 1.0, 0.0
        out[d["name"]] = {"tensor": t, "scale": scale, "zp": zp}
    return out


def dequant(tinfo):
    t = tinfo["tensor"]
    if t.dtype == np.int8:
        return (t.astype(np.float32) - tinfo["zp"]) * tinfo["scale"]
    if t.dtype == np.uint8:
        return (t.astype(np.float32) - tinfo["zp"]) * tinfo["scale"]
    return t.astype(np.float32)


def find(ts, prefix):
    # TFLite fused names contain MULTIPLE op names joined by ';'
    # -> match by START of the key, never substring
    for k, v in ts.items():
        if k.startswith(prefix):
            return v
    raise KeyError(prefix)


def build_dscnn_keras(src=SRC):
    """Rebuild the verified true-float32 DS-CNN Keras model from a flatbuffer.

    Dequantizes the int8 conv kernels of the published (mixed-precision)
    artifact and assembles the 9 fused conv-bias-ReLU blocks + GAP + dense.
    Returns the compiled Keras model (linear dense head, no softmax --
    apply softmax when comparing against TFLite outputs).
    """
    ts = load_tflite_tensors(src)

    # ---- build Keras model: conv+BN(fused)+ReLU x9, avgpool, flatten, dense
    inputs = Input(shape=(49, 10, 1), name="input_1")

    # conv1: kernel conv2d/Conv2D [64,10,4,1] int8 -> (10,4,1,64)
    k1 = dequant(find(ts, "functional_1/conv2d/Conv2D"))
    b1 = dequant(find(ts, "functional_1/activation/Relu;functional_1/batch_normalization/FusedBatchNormV3"))
    conv1 = Conv2D(64, (10, 4), strides=(2, 2), padding="same")
    x = conv1(inputs)
    conv1.set_weights([np.transpose(k1, (1, 2, 3, 0)), b1])
    x = ReLU()(x)

    for blk in range(4):
        # depthwise: kernel batch_normalization_<2k+1> [1,3,3,64] -> (3,3,64,1)
        dk = dequant(find(ts, f"functional_1/batch_normalization_{2*blk+1}/FusedBatchNormV3;functional_1/d"))
        db = dequant(find(ts, f"functional_1/activation_{2*blk+1}/Relu;functional_1/batch_normalization_{2*blk+1}/Fused"))
        dw = DepthwiseConv2D((3, 3), padding="same")
        x = dw(x)
        dw.set_weights([np.transpose(dk, (1, 2, 3, 0)), db])
        x = ReLU()(x)
        # 1x1 conv: conv2d_<blk+1> [64,1,1,64]
        ck = dequant(find(ts, f"functional_1/conv2d_{blk+1}/Conv2D"))
        cb = dequant(find(ts, f"functional_1/activation_{2*blk+2}/Relu;functional_1/batch_normalization_{2*blk+2}/Fused"))
        c1 = Conv2D(64, (1, 1), padding="same")
        x = c1(x)
        c1.set_weights([np.transpose(ck, (1, 2, 3, 0)), cb])
        x = ReLU()(x)

    x = AveragePooling2D(pool_size=(24, 5))(x)
    x = Flatten()(x)
    dk = dequant(find(ts, "functional_1/dense/MatMul"))
    db = dequant(find(ts, "functional_1/dense/BiasAdd/ReadVariableOp/resource"))
    dense = Dense(12)
    x = dense(x)
    dense.set_weights([np.transpose(dk, (1, 0)), db])
    model = Model(inputs, x, name="kws_ref_fp32")
    model.summary()
    return model


def main():
    model = build_dscnn_keras(SRC)

    # ---- verify against TFLite interpreter
    interp = tfl.Interpreter(model_path=SRC)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    np.random.seed(42)
    test_x = np.random.randn(1, 49, 10, 1).astype(np.float32)
    interp.set_tensor(inp["index"], test_x)
    interp.invoke()
    tflite_out = interp.get_tensor(interp.get_output_details()[0]["index"])
    keras_out = model.predict(test_x, verbose=0)
    # TFLite model ends with Softmax; Keras Dense is linear -> apply softmax
    kk = keras_out - keras_out.max(axis=-1, keepdims=True)
    keras_out = np.exp(kk) / np.exp(kk).sum(axis=-1, keepdims=True)
    diff = float(np.max(np.abs(tflite_out - keras_out)))
    agree = float(np.mean(np.argmax(tflite_out, -1) == np.argmax(keras_out, -1)))
    print(f"TFLite vs Keras max_abs_diff = {diff:.6e} (argmax agree {agree:.0%})")
    # NOTE: the official 'float32' model is MIXED precision (int8 conv kernels).
    # TFLite computes those convs on an int8 path -> small diff vs our dequantized
    # fp32 rebuild is expected. Assert ranking agreement, not bit-exactness.
    assert agree == 1.0 and diff < 0.02, "rebuild mismatch!"

    # ---- export true float32 TFLite (via SavedModel to dodge converter bug)
    import tempfile, os
    sm_dir = tempfile.mkdtemp(prefix="kws_fp32_sm_")
    model.export(sm_dir)
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = []
    tflite_fp32 = conv.convert()
    open(OUT_TFLITE, "wb").write(tflite_fp32)
    print(f"wrote {OUT_TFLITE} ({len(tflite_fp32)} bytes)")

    # ---- export ONNX
    spec = (tf.TensorSpec([1, 49, 10, 1], tf.float32, name="input_1"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(onnx_model, OUT_ONNX)
    print(f"wrote {OUT_ONNX}")


if __name__ == "__main__":
    main()
