"""Install whatever Meta's sam3 code still misses (one package at a time, max 15)."""
import subprocess, sys
MAP = {"cv2": "opencv-python-headless", "PIL": "pillow", "sklearn": "scikit-learn", "skimage": "scikit-image",
       "yaml": "pyyaml", "pkg_resources": "setuptools<80", "pycocotools": "pycocotools", "omegaconf": "omegaconf",
       "hydra": "hydra-core", "einops": "einops", "iopath": "iopath", "ftfy": "ftfy==6.1.1", "regex": "regex",
       "timm": "timm>=1.0.17", "psutil": "psutil", "huggingface_hub": "huggingface_hub", "decord": "decord", "av": "av"}
test = "from sam3.model_builder import build_sam3_predictor; import sam3.model.sam3_multiplex_video_predictor"
added = []
for _ in range(15):
    r = subprocess.run([sys.executable, "-c", test], capture_output=True, text=True)
    if r.returncode == 0:
        print("sam3 import ok; added:", added)
        sys.exit(0)
    err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?"
    if "No module named" not in err:
        print(r.stderr[-3000:]); sys.exit(1)
    mod = err.split("No module named")[-1].strip().strip("'\"").split(".")[0]
    pkg = MAP.get(mod, mod)
    print("installing", pkg, "for", mod, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg], check=True)
    added.append(pkg)
sys.exit("too many missing modules")
