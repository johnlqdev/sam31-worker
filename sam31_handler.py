#!/usr/bin/env python3
"""RunPod Serverless worker: SAM 3.1 (Meta, facebook/sam3.1) text-prompted segmentation of one image.

Runs inside the community image `muhammadshahmirzaman/sam3.1:slim` (Meta's sam3 code with SAM 3.1 support
at /app/sam3, Python 3.12 venv), which ships no weights and no RunPod handler. At cold start this script:
  1. downloads sam3.1_multiplex.pt from the owner's Dropbox through the Supabase lab function
     (worker token: may only fetch files under /dual_arm_relay/sam3.1/), unless it is already on disk;
  2. builds Meta's SAM 3.1 predictor (build_sam3_predictor, version="sam3.1", no FlashAttention 3 so it
     runs on non-Hopper GPUs);
  3. serves jobs. A job treats the image as a one-frame video: start_session -> add_prompt(text) -> close.

Job input:
  {"image_url": "https://...jpg"} or {"image_b64": "..."}   the image
  "prompt": "cat"                                            text prompt (noun phrase)
  "threshold": 0.5                                           optional output probability threshold
  "return_masks": false                                      true -> masks as run-length encoding (COCO style)
Output: objects (id, score, box in pixels xyxy, mask area), image size, timings, GPU, cold-start facts.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import tempfile
import time
import urllib.request

T_START = time.time()
LAB_URL = os.environ.get("LAB_URL", "")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
CKPT_DROPBOX = os.environ.get("SAM31_DROPBOX_PATH", "/dual_arm_relay/sam3.1/sam3.1_multiplex.pt")
CKPT_LOCAL = os.environ.get("SAM31_CKPT", "/checkpoints/sam3.1_multiplex.pt")
CKPT_BYTES = 3502755717

COLD = {"worker_boot_unix": T_START}


def lab(action: str, **fields) -> dict:
    req = urllib.request.Request(LAB_URL, data=json.dumps({"token": WORKER_TOKEN, "action": action, **fields}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def ensure_checkpoint() -> None:
    if os.path.isfile(CKPT_LOCAL) and os.path.getsize(CKPT_LOCAL) == CKPT_BYTES:
        COLD["checkpoint"] = "already on disk"
        return
    os.makedirs(os.path.dirname(CKPT_LOCAL), exist_ok=True)
    t0 = time.time()
    link = lab("dropbox_link", path=CKPT_DROPBOX)["link"]
    tmp = CKPT_LOCAL + ".part"
    with urllib.request.urlopen(link, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
    size = os.path.getsize(tmp)
    if size != CKPT_BYTES:
        raise RuntimeError(f"checkpoint size {size} != {CKPT_BYTES}")
    os.replace(tmp, CKPT_LOCAL)
    dt = time.time() - t0
    COLD.update(checkpoint="downloaded from Dropbox", download_s=round(dt, 1), download_MBps=round(size / dt / 1e6, 1))


def load_predictor():
    import torch
    from sam3.model_builder import build_sam3_predictor
    t0 = time.time()
    pred = build_sam3_predictor(checkpoint_path=CKPT_LOCAL, version="sam3.1", use_fa3=False, compile=False,
                                warm_up=False, async_loading_frames=False)
    COLD.update(model_load_s=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0),
                gpu_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                torch=torch.__version__)
    return pred


def report(kind: str, message: str, data=None) -> None:
    try:
        lab("report", run="sam31", kind=kind, message=f"[{os.environ.get('RUNPOD_POD_ID', '?')}] {message}", data=data)
    except Exception as exc:  # noqa: BLE001
        print("report failed:", exc, flush=True)


try:
    report("stage", "handler: downloading checkpoint")
    ensure_checkpoint()
    report("stage", "handler: checkpoint ready, loading SAM 3.1", COLD)
    PREDICTOR = load_predictor()
    # Meta's snapshot passes offload_state_to_cpu to init_state, which the SAM 3.1 multiplex model does not
    # accept: drop arguments the model's init_state does not know.
    import inspect
    _orig_init_state = PREDICTOR.model.init_state
    _init_params = set(inspect.signature(_orig_init_state).parameters)

    def _init_state(**kw):
        return _orig_init_state(**{k: v for k, v in kw.items() if k in _init_params})
    PREDICTOR.model.init_state = _init_state
except Exception:  # noqa: BLE001
    import traceback
    report("error", "handler init failed", {"traceback": traceback.format_exc()[-8000:], "cold": COLD})
    raise
COLD["cold_start_total_s"] = round(time.time() - T_START, 1)
report("stage", "handler: SAM 3.1 ready", COLD)
print("SAM 3.1 worker ready:", json.dumps(COLD), flush=True)
N_JOBS = 0


def _to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except Exception:  # noqa: BLE001
        pass
    import numpy as np
    return np.asarray(x)


def _rle(mask) -> dict:
    """COCO-style uncompressed RLE (column-major counts starting with zeros)."""
    import numpy as np
    m = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    diffs = np.flatnonzero(np.diff(np.concatenate([[0], m, [0]])))
    runs = np.diff(np.concatenate([[0], diffs, [m.size]]))
    return {"size": list(np.asarray(mask).shape), "counts": runs.tolist()}


def segment(image, prompt: str, threshold: float | None, return_masks: bool) -> dict:
    import numpy as np
    w, h = image.size
    d = tempfile.mkdtemp(prefix="sam31_")
    try:
        image.convert("RGB").save(os.path.join(d, "00000.jpg"), quality=95)
        t0 = time.time()
        sid = PREDICTOR.handle_request(dict(type="start_session", resource_path=d))["session_id"]
        t1 = time.time()
        req = dict(type="add_prompt", session_id=sid, frame_index=0, text=prompt)
        if threshold is not None:
            req["output_prob_thresh"] = threshold
        try:
            out = PREDICTOR.handle_request(req)["outputs"]
        except TypeError:
            req.pop("output_prob_thresh", None)
            out = PREDICTOR.handle_request(req)["outputs"]
        t2 = time.time()
        PREDICTOR.handle_request(dict(type="close_session", session_id=sid))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    ids = _to_np(out.get("out_obj_ids", [])).astype(int).tolist()
    probs = _to_np(out.get("out_probs", [])).tolist()
    boxes = _to_np(out.get("out_boxes_xywh", np.zeros((0, 4))))
    masks = out.get("out_binary_masks")
    masks = _to_np(masks) if masks is not None else None
    objects = []
    for i, oid in enumerate(ids):
        b = boxes[i].tolist() if len(boxes) > i else None
        if b is not None and max(b) <= 1.0001:          # normalized xywh -> pixel xyxy
            b = [b[0] * w, b[1] * h, (b[0] + b[2]) * w, (b[1] + b[3]) * h]
        elif b is not None:
            b = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
        obj = {"id": oid, "score": round(float(probs[i]), 4) if len(probs) > i else None,
               "box_xyxy": [round(v, 1) for v in b] if b else None}
        if masks is not None and len(masks) > i:
            m = masks[i] > 0.5
            obj["mask_area_px"] = int(m.sum())
            if return_masks:
                obj["mask_rle"] = _rle(m)
        objects.append(obj)
    return {"image_size": [w, h], "prompt": prompt, "n_objects": len(objects), "objects": objects,
            "timing_ms": {"start_session": round((t1 - t0) * 1000), "prompt": round((t2 - t1) * 1000)},
            "output_keys": sorted(out.keys())}


def handler(job):
    global N_JOBS
    from PIL import Image
    t0 = time.time()
    inp = job.get("input") or {}
    if inp.get("admin") == "reload" and os.environ.get("SAM31_ALLOW_RELOAD") == "1":   # supervisor setups only
        import threading
        threading.Timer(2.0, lambda: os._exit(0)).start()
        return {"reloading": True, "worker": COLD}
    if inp.get("image_b64"):
        img = Image.open(io.BytesIO(base64.b64decode(inp["image_b64"])))
    elif inp.get("image_url"):
        req = urllib.request.Request(inp["image_url"], headers={"User-Agent": "sam31-runpod-worker"})
        with urllib.request.urlopen(req, timeout=60) as r:
            img = Image.open(io.BytesIO(r.read()))
    else:
        return {"error": "give image_url or image_b64"}
    t_img = time.time()
    try:
        res = segment(img, str(inp.get("prompt", "object")), inp.get("threshold"), bool(inp.get("return_masks")))
    except Exception:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()[-6000:]
        report("error", "job failed", {"traceback": tb})
        return {"error": tb}
    N_JOBS += 1
    res.update(job_index_on_worker=N_JOBS, first_job_on_worker=(N_JOBS == 1),
               image_fetch_ms=round((t_img - t0) * 1000), handler_ms=round((time.time() - t0) * 1000),
               worker=COLD)
    return res


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
