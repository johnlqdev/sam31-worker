# sam31-worker

RunPod Serverless worker for Meta's **SAM 3.1** (text-prompted segmentation of an image).

* Image: `ghcr.io/johnlqdev/sam31-worker:latest`, built by GitHub Actions from `Dockerfile`
  (base `muhammadshahmirzaman/sam3.1:slim` + RunPod SDK + fixes + `sam31_handler.py`).
* Weights are **not** in the image (Meta's licence gates them). At cold start the worker downloads
  `sam3.1_multiplex.pt` through the owner's Supabase lab function (`LAB_URL`, `WORKER_TOKEN` env vars).
* Job input: `{"image_url": "...", "prompt": "cat", "threshold": 0.5, "return_masks": false}`
  Output: objects with id, score, box (pixels, xyxy), mask area (and RLE masks on request), timings.
