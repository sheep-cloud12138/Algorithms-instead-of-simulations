"""Evaluate recovered wall-flux U-Nets at 4/6/8/10-point stencils.

The WSL service is unresponsive in this session, so this script reads the
recovered torch archives without importing torch and executes the small U-Net
in NumPy.
"""
import argparse
import collections
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


ROOT = Path(__file__).resolve().parents[1]
RECOVERED = ROOT / "tmp" / "wsl_recovered" / "home" / "wjh" / "nc_run"
CKPT = RECOVERED / "p7"
PUB = RECOVERED / "pub"
DTP = 1.2


class StorageRef:
    def __init__(self, key, dtype):
        self.key = key
        self.dtype = dtype


class TensorRef:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = int(offset)
        self.size = tuple(int(v) for v in size)
        self.stride = tuple(int(v) for v in stride)


def _rebuild_tensor_v2(storage, offset, size, stride, *unused):
    return TensorRef(storage, offset, size, stride)


def _rebuild_parameter(data, *unused):
    return data


class TorchUnpickler(pickle.Unpickler):
    def __init__(self, file, archive):
        super().__init__(file, fix_imports=False, encoding="utf-8")
        self.archive = archive

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name == "FloatStorage":
            return ("float32",)
        if module == "torch" and name == "LongStorage":
            return ("int64",)
        raise pickle.UnpicklingError(f"unsupported torch global: {module}.{name}")

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")
        dtype = pid[1][0]
        return StorageRef(str(pid[2]), dtype)


def load_torch(path):
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        prefix = zf.namelist()[0].split("/", 1)[0]
        with zf.open(f"{prefix}/data.pkl") as stream:
            return TorchUnpickler(stream, zf).load()


def materialize(obj, zf, prefix, cache):
    if isinstance(obj, StorageRef):
        if obj.key not in cache:
            cache[obj.key] = np.frombuffer(
                bytearray(zf.read(f"{prefix}/data/{obj.key}")), dtype=obj.dtype
            )
        return cache[obj.key]
    if isinstance(obj, TensorRef):
        base = materialize(obj.storage, zf, prefix, cache)
        offset = obj.offset
        return np.lib.stride_tricks.as_strided(
            base[offset:],
            shape=obj.size,
            strides=tuple(s * base.itemsize for s in obj.stride),
        )
    if isinstance(obj, dict):
        return {k: materialize(v, zf, prefix, cache) for k, v in obj.items()}
    if isinstance(obj, list):
        return [materialize(v, zf, prefix, cache) for v in obj]
    if isinstance(obj, tuple):
        return tuple(materialize(v, zf, prefix, cache) for v in obj)
    return obj


def load_pt(path):
    with zipfile.ZipFile(path) as zf:
        prefix = zf.namelist()[0].split("/", 1)[0]
        with zf.open(f"{prefix}/data.pkl") as stream:
            obj = TorchUnpickler(stream, zf).load()
        return materialize(obj, zf, prefix, {})


def silu(x):
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def conv2d(x, weight, bias, padding=1):
    n, c, h, w = x.shape
    out_c = weight.shape[0]
    kh, kw = weight.shape[2:]
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    cols = sliding_window_view(x, (kh, kw), axis=(2, 3))
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * w, -1)
    out = cols @ weight.reshape(out_c, -1).T
    out += bias
    return out.reshape(n, h, w, out_c).transpose(0, 3, 1, 2)


def conv1x1(x, weight, bias):
    out = np.einsum("nchw,oc->nohw", x, weight[:, :, 0, 0], optimize=True)
    return out + bias.reshape(1, -1, 1, 1)


def group_norm(x, weight, bias, groups=4, eps=1e-5):
    n, c, h, w = x.shape
    g = x.reshape(n, groups, c // groups, h, w)
    mean = g.mean(axis=(2, 3, 4), keepdims=True)
    var = g.var(axis=(2, 3, 4), keepdims=True)
    x = ((g - mean) / np.sqrt(var + eps)).reshape(n, c, h, w)
    return x * weight.reshape(1, c, 1, 1) + bias.reshape(1, c, 1, 1)


def block(x, sd, name):
    x = conv2d(x, sd[f"{name}.c1.weight"], sd[f"{name}.c1.bias"])
    x = silu(group_norm(x, sd[f"{name}.n1.weight"], sd[f"{name}.n1.bias"]))
    x = conv2d(x, sd[f"{name}.c2.weight"], sd[f"{name}.c2.bias"])
    return silu(group_norm(x, sd[f"{name}.n2.weight"], sd[f"{name}.n2.bias"]))


def max_pool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def upsample2(x):
    def axis_weights(size):
        src = np.maximum((np.arange(2 * size) + 0.5) / 2.0 - 0.5, 0.0)
        lo = np.floor(src).astype(np.int64)
        hi = np.minimum(lo + 1, size - 1)
        return lo, hi, (src - lo).astype(np.float32)

    yl, yh, wy = axis_weights(x.shape[2])
    xl, xh, wx = axis_weights(x.shape[3])
    wy = wy.reshape(1, 1, -1, 1)
    wx = wx.reshape(1, 1, 1, -1)
    y = x[:, :, yl, :] * (1.0 - wy) + x[:, :, yh, :] * wy
    return y[:, :, :, xl] * (1.0 - wx) + y[:, :, :, xh] * wx


def unet(x, sd):
    x1 = block(x, sd, "e1")
    x2 = block(max_pool2(x1), sd, "e2")
    x3 = block(max_pool2(x2), sd, "e3")
    b = block(max_pool2(x3), sd, "b")
    y = block(np.concatenate([upsample2(b), x3], axis=1), sd, "d3")
    y = block(np.concatenate([upsample2(y), x2], axis=1), sd, "d2")
    y = block(np.concatenate([upsample2(y), x1], axis=1), sd, "d1")
    return conv1x1(y, sd["o.weight"], sd["o.bias"])


def theta(t):
    return (t - (1.0 - DTP / 2.0)) / DTP


def input_scale(x):
    n = len(x)
    values = []
    for c in range(4):
        means = []
        for k in range(0, n, 32):
            a = x[k : k + 32, c].reshape(min(32, n - k), -1)
            means.append(a.std(axis=1, ddof=1).mean())
        values.append(float(np.mean(means)))
    return np.array(values, np.float32).reshape(1, 4, 1, 1)


def stencil_weights(n, points):
    x = np.arange(points) * (1.0 / (n - 1))
    return np.linalg.solve(np.vander(x, points, increasing=True).T, np.eye(points)[1])


def wall_flux(t, points):
    weights = stencil_weights(t.shape[-1], points)
    left = -(t[:, :, :points] * weights).sum(-1)
    right = (t[:, :, -points:][:, :, ::-1] * weights).sum(-1)
    return left, right


def mean_error(pred, target):
    return float(
        np.abs(pred.mean(1) - target.mean(1)).mean()
        / np.abs(target.mean(1)).mean()
    )


def evaluate_model(name, sd, target_scale, input_scale, xte, yte):
    pred = np.empty_like(xte)
    for k in range(0, len(xte), 8):
        pred[k : k + 8] = unet(xte[k : k + 8] / input_scale, sd) * target_scale
    yh = yte.copy()
    yh[:, 2] = xte[:, 2] + DTP * pred[:, 2]
    row = {
        "relL2_T": float(
            np.linalg.norm(yh[:, 2] - yte[:, 2]) / np.linalg.norm(yte[:, 2])
        )
    }
    for points in (4, 6, 8, 10):
        target_flux = wall_flux(yte[:, 2], points)
        pred_flux = wall_flux(yh[:, 2], points)
        floor_flux = wall_flux(xte[:, 2], points)
        row[f"M{points}"] = {
            "hot": mean_error(pred_flux[0], target_flux[0]),
            "cold": mean_error(pred_flux[1], target_flux[1]),
            "floor": mean_error(floor_flux[0], target_flux[0]),
        }
    print(name, json.dumps(row, indent=1), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--out", default=str(ROOT / "论文" / "public_flux_stencil_10pt.json"))
    args = parser.parse_args()

    x_train = load_pt(PUB / "train" / "X.pt").astype(np.float32, copy=False)
    x_test = load_pt(PUB / "test_legacy" / "X.pt").astype(np.float32, copy=False)
    y_test = load_pt(PUB / "test_legacy" / "Y.pt").astype(np.float32, copy=False)
    x_train[:, 2] = theta(x_train[:, 2])
    x_test[:, 2] = theta(x_test[:, 2])
    y_test[:, 2] = theta(y_test[:, 2])
    scale = input_scale(x_train)
    print("input scale", scale.reshape(-1).tolist(), flush=True)

    x_test = x_test[: args.samples]
    y_test = y_test[: args.samples]
    public = load_pt(CKPT / "pub_nets.pt")
    checkpoints = {
        "base": public["pub_prior"],
        "lam1_s0": load_pt(CKPT / "pub_flux_lam1.0.pt"),
        "lam1_s1": load_pt(CKPT / "pub_flux_lam1.0_s1.pt"),
    }
    result = {}
    for name, checkpoint in checkpoints.items():
        result[name] = evaluate_model(
            name,
            checkpoint["sd"],
            np.asarray(checkpoint["sD"], dtype=np.float32).reshape(1, 4, 1, 1)
            if np.asarray(checkpoint["sD"]).size == 4
            else float(checkpoint["sD"]),
            scale,
            x_test,
            y_test,
        )
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
