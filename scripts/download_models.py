"""Download the two models MAGIC ROTO needs into a ComfyUI install.

Usage:
    python scripts/download_models.py "C:/path/to/ComfyUI"

Run it with ComfyUI's own Python (the one in its .venv) so huggingface_hub is available.
Nothing is redistributed by this repo: both models come straight from their official
Hugging Face pages, under their own licences.
"""
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

SAM3_REPO = "Comfy-Org/sam3.1"
SAM3_FILE = "checkpoints/sam3.1_multiplex_fp16.safetensors"  # ~1.7 GB
VITMATTE_REPO = "hustvl/vitmatte-small-composition-1k"         # ~100 MB


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    comfy = Path(sys.argv[1]).expanduser().resolve()
    models = comfy / "models"
    if not models.is_dir():
        sys.exit(f"No 'models' folder in {comfy}: point this at your ComfyUI folder.")

    ckpt_dir = models / "checkpoints"
    target = ckpt_dir / Path(SAM3_FILE).name
    if target.exists():
        print(f"SAM3 already present: {target}")
    else:
        print("Downloading SAM3 (~1.7 GB)...")
        tmp = models / "_sam3_tmp"
        path = hf_hub_download(SAM3_REPO, SAM3_FILE, local_dir=tmp)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        Path(path).replace(target)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"SAM3 saved to {target}")

    vm_dir = models / "vitmatte" / VITMATTE_REPO.split("/")[1]
    print("Downloading VitMatte small (~100 MB)...")
    snapshot_download(VITMATTE_REPO, local_dir=vm_dir,
                      allow_patterns=["*.json", "*.safetensors", "*.txt"])
    print(f"VitMatte saved to {vm_dir}")
    print("Done. Restart ComfyUI.")


if __name__ == "__main__":
    main()
