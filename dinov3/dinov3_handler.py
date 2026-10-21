#!/usr/bin/env python3
"""RunPod Serverless worker: DINOv3 image features (Meta, default ViT-7B/16 pretrained on LVD-1689M).

Weights are read from the RunPod network volume (/runpod-volume/dinov3/...). If the file is missing there,
the worker downloads it from the owner's Dropbox through the Supabase lab function (worker role) as a fallback.
The model is built on the "meta" device, the checkpoint is memory-mapped, and the tensors go straight to the
GPU in bfloat16 (about 13.4 GB for ViT-7B, so it fits a 24-32 GB card).

Job input:
  {"image_url": "https://...jpg"} or {"image_b64": "..."}     the image
  "resolution": 512        longer image side in pixels, rounded to a multiple of 16 (patch size); max 2048
  "return": ["cls"]        any of "cls", "patch_mean", "storage", "patch_tokens"
  "patch_pca": 0           >0: reduce patch tokens to this many dims (PCA over this image's patches)
  "format": "list"         "list" (JSON floats) or "b64" (base64 little-endian float16, with shape)
Output: requested features, feature grid size (h, w), image size, timings, worker cold-start facts.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.request

T_START = time.time()
LAB_URL = os.environ.get("LAB_URL", "")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
ARCH = os.environ.get("DINOV3_ARCH", "dinov3_vit7b16")
CKPT = os.environ.get("DINOV3_CKPT", "/runpod-volume/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")
CKPT_DROPBOX = os.environ.get("DINOV3_DROPBOX_PATH", "/dual_arm_relay/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")
CKPT_FALLBACK = "/tmp/dinov3.pth"
PATCH = 16
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
COLD = {"worker_boot_unix": T_START, "arch": ARCH}


def lab(action: str, **fields) -> dict:
    req = urllib.request.Request(LAB_URL, data=json.dumps({"token": WORKER_TOKEN, "action": action, **fields}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def report(kind: str, message: str, data=None) -> None:
    if not LAB_URL:
        return
    try:
        lab("report", run="dinov3", kind=kind, message=f"[{os.environ.get('RUNPOD_POD_ID', '?')}] {message}", data=data)
    except Exception as exc:  # noqa: BLE001
        print("report failed:", exc, flush=True)


def checkpoint_path() -> str:
    if os.path.isfile(CKPT):
        COLD["checkpoint"] = "network volume"
        return CKPT
    t0 = time.time()
    link = lab("dropbox_link", path=CKPT_DROPBOX)["link"]
    import shutil
    with urllib.request.urlopen(link, timeout=600) as r, open(CKPT_FALLBACK + ".part", "wb") as f:
        shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
    os.replace(CKPT_FALLBACK + ".part", CKPT_FALLBACK)
    COLD.update(checkpoint="downloaded from Dropbox (volume file missing)", download_s=round(time.time() - t0, 1))
    return CKPT_FALLBACK


def load_model(path: str):
    import torch
    from dinov3.hub import backbones
    t0 = time.time()
    model = getattr(backbones, ARCH)(pretrained=False, device="meta")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    rope_periods = model.rope_embed.periods.detach().float().clone()   # RoPE runs in fp32; keep its buffer fp32
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = model.to(dev, dtype=dtype).eval()
    model.rope_embed.periods = rope_periods.to(dev)
    del state
    COLD.update(model_load_s=round(time.time() - t0, 1), device=dev, dtype=str(dtype).replace("torch.", ""),
                params_b=round(sum(p.numel() for p in model.parameters()) / 1e9, 2), torch=torch.__version__)
    if dev == "cuda":
        COLD.update(gpu=torch.cuda.get_device_name(0),
                    gpu_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                    weights_gb_on_gpu=round(torch.cuda.memory_allocated() / 1e9, 1))
    return model


try:
    report("stage", "dinov3 handler: loading model")
    MODEL = load_model(checkpoint_path())
except Exception:  # noqa: BLE001
    import traceback
    report("error", "dinov3 handler init failed", {"traceback": traceback.format_exc()[-8000:], "cold": COLD})
    raise
COLD["cold_start_total_s"] = round(time.time() - T_START, 1)
report("stage", "dinov3 handler: ready", COLD)
print("DINOv3 worker ready:", json.dumps(COLD), flush=True)
N_JOBS = 0


def _encode(t, fmt: str):
    import numpy as np
    a = t.detach().float().cpu().numpy()
    if fmt == "b64":
        return {"b64": base64.b64encode(a.astype("<f2").tobytes()).decode(), "dtype": "float16", "shape": list(a.shape)}
    return np.round(a, 5).tolist()


def features(img, resolution: int, want: list[str], pca: int, fmt: str) -> dict:
    import torch
    import torch.nn.functional as F
    import numpy as np
    img = img.convert("RGB")
    w0, h0 = img.size
    s = min(max(int(resolution), PATCH), 2048) / max(w0, h0)
    w, h = max(PATCH, round(w0 * s / PATCH) * PATCH), max(PATCH, round(h0 * s / PATCH) * PATCH)
    x = torch.from_numpy(np.asarray(img.resize((w, h), resample=3), dtype=np.float32) / 255.0).permute(2, 0, 1)
    x = ((x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None])[None]
    p = next(MODEL.parameters())
    t0 = time.time()
    with torch.inference_mode():
        out = MODEL.forward_features(x.to(p.device, dtype=p.dtype))
        if p.device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.time()
    res = {"image_size": [w0, h0], "input_size": [w, h], "grid_hw": [h // PATCH, w // PATCH],
           "embed_dim": int(out["x_norm_clstoken"].shape[-1])}
    if "cls" in want:
        res["cls"] = _encode(out["x_norm_clstoken"][0], fmt)
    if "patch_mean" in want:
        res["patch_mean"] = _encode(out["x_norm_patchtokens"][0].float().mean(0), fmt)
    if "storage" in want:
        res["storage"] = _encode(out["x_storage_tokens"][0], fmt)
    if "patch_tokens" in want:
        pt = out["x_norm_patchtokens"][0].float()
        if pca and pca > 0:
            pt = pt - pt.mean(0, keepdim=True)
            _, _, v = torch.pca_lowrank(pt, q=min(int(pca), min(pt.shape)), center=False)
            pt = pt @ v
            res["patch_pca_dims"] = int(pt.shape[-1])
        res["patch_tokens"] = _encode(pt, "b64" if fmt == "b64" or pt.numel() > 300_000 else fmt)
    res["timing_ms"] = {"forward": round((t1 - t0) * 1000), "postprocess": round((time.time() - t1) * 1000)}
    return res


def handler(job):
    global N_JOBS
    from PIL import Image
    t0 = time.time()
    inp = job.get("input") or {}
    if inp.get("image_b64"):
        img = Image.open(io.BytesIO(base64.b64decode(inp["image_b64"])))
    elif inp.get("image_url"):
        req = urllib.request.Request(inp["image_url"], headers={"User-Agent": "dinov3-runpod-worker"})
        with urllib.request.urlopen(req, timeout=60) as r:
            img = Image.open(io.BytesIO(r.read()))
    else:
        return {"error": "give image_url or image_b64"}
    t_img = time.time()
    want = inp.get("return") or ["cls"]
    want = [want] if isinstance(want, str) else list(want)
    try:
        res = features(img, int(inp.get("resolution", 512)), want, int(inp.get("patch_pca", 0) or 0),
                       str(inp.get("format", "list")))
    except Exception:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()[-6000:]
        report("error", "job failed", {"traceback": tb})
        return {"error": tb}
    N_JOBS += 1
    res.update(job_index_on_worker=N_JOBS, image_fetch_ms=round((t_img - t0) * 1000),
               handler_ms=round((time.time() - t0) * 1000), worker=COLD)
    return res


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
