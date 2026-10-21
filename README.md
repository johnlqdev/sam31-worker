# Vision workers for RunPod Serverless

Two independent workers, each built by GitHub Actions and pushed to the public package `ghcr.io/johnlqdev/sam31-worker`:

| Tag | Folder | Model | Workflow |
|---|---|---|---|
| `:latest` | repo root | SAM 3.1 text-prompted segmentation (`sam31_handler.py`) | `build.yml` |
| `:dinov3` | `dinov3/` | DINOv3 image features, default ViT-7B/16 (`dinov3_handler.py`) | `build-dinov3.yml` |

No weights are baked in. Both read their checkpoints from the RunPod network volume (`/runpod-volume/sam3.1/...`,
`/runpod-volume/dinov3/...`) and fall back to a Dropbox download through the owner's Supabase lab function.
Job formats are documented at the top of each handler.
