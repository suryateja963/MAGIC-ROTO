"""MAGIC ROTO: trimap + VitMatte edge refinement + temporal smoothing for ComfyUI.

Pairs with the core SAM3 nodes: SAM3 gives a hard mask, these nodes turn it
into a soft, hair-level alpha matte. Weights live in models/vitmatte/<name>.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
import folder_paths

VITMATTE_DIR = os.path.join(folder_paths.models_dir, "vitmatte")
_CACHE = {}


def _model_names():
    if not os.path.isdir(VITMATTE_DIR):
        return ["vitmatte-small-composition-1k"]
    names = [d for d in os.listdir(VITMATTE_DIR)
             if os.path.isfile(os.path.join(VITMATTE_DIR, d, "config.json"))]
    return sorted(names) or ["vitmatte-small-composition-1k"]


def _load(name):
    if name not in _CACHE:
        from transformers import VitMatteForImageMatting
        path = os.path.join(VITMATTE_DIR, name)
        model = VitMatteForImageMatting.from_pretrained(path).eval()
        _CACHE.clear()
        _CACHE[name] = model
    return _CACHE[name]


def _kernel(px):
    size = max(1, int(px)) * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class VitMatteTrimap:
    """Hard mask -> trimap: 1 = sure object, 0 = sure background, 0.5 = unknown edge band."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "erode_px": ("INT", {"default": 8, "min": 0, "max": 200,
                                 "tooltip": "Shrink the mask inward: what stays is 'sure object'."}),
            "dilate_px": ("INT", {"default": 16, "min": 0, "max": 200,
                                  "tooltip": "Grow the mask outward: beyond it is 'sure background'. Wider = more room for hair."}),
        }}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("trimap",)
    FUNCTION = "run"
    CATEGORY = "MAGIC ROTO"

    def run(self, mask, erode_px, dilate_px):
        out = []
        for m in mask.cpu().numpy():
            hard = (m > 0.5).astype(np.uint8)
            fg = cv2.erode(hard, _kernel(erode_px)) if erode_px > 0 else hard
            band = cv2.dilate(hard, _kernel(dilate_px)) if dilate_px > 0 else hard
            tri = np.where(fg > 0, 1.0, np.where(band > 0, 0.5, 0.0)).astype(np.float32)
            out.append(tri)
        return (torch.from_numpy(np.stack(out)),)


class VitMatteRefine:
    """Runs VitMatte inside the trimap's unknown band, frame by frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "trimap": ("MASK",),
            "model_name": (_model_names(),),
            "max_side": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 32,
                                 "tooltip": "Frames are downscaled to this long side for VitMatte to fit 8 GB VRAM; alpha is upscaled back."}),
            "white_point": ("FLOAT", {"default": 0.99, "min": 0.5, "max": 1.0, "step": 0.01,
                                      "tooltip": "Alpha above this snaps to solid 1.0 (fixes 'almost white' interiors)."}),
        }}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "run"
    CATEGORY = "MAGIC ROTO"

    def run(self, image, trimap, model_name, max_side, white_point):
        device = mm.get_torch_device()
        use_fp16 = device.type == "cuda"
        model = _load(model_name).to(device)
        model = model.half() if use_fp16 else model.float()
        mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

        frames = image.shape[0]
        if trimap.shape[0] not in (1, frames):
            raise ValueError(f"trimap has {trimap.shape[0]} frames but image has {frames}")
        pbar = comfy.utils.ProgressBar(frames)
        out = []
        for i in range(frames):
            img = image[i].to(device).permute(2, 0, 1).unsqueeze(0)          # 1,3,H,W
            tri = trimap[i if trimap.shape[0] > 1 else 0].to(device)[None, None]
            h, w = img.shape[-2:]
            if tri.shape[-2:] != (h, w):
                tri = F.interpolate(tri, size=(h, w), mode="nearest")

            unknown = (tri > 0.25) & (tri < 0.75)
            if not unknown.any():
                out.append(tri[0, 0].float().cpu())
                pbar.update(1)
                continue

            scale = min(1.0, max_side / max(h, w))
            sh, sw = max(32, round(h * scale)), max(32, round(w * scale))
            img_s = F.interpolate(img, size=(sh, sw), mode="bilinear", align_corners=False) if scale < 1 else img
            tri_s = F.interpolate(tri, size=(sh, sw), mode="nearest") if scale < 1 else tri

            ph, pw = (32 - sh % 32) % 32, (32 - sw % 32) % 32
            x = torch.cat([(img_s - mean) / std, tri_s], dim=1)
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            with torch.no_grad():
                alpha = model(pixel_values=x.half() if use_fp16 else x).alphas.float()
            alpha = alpha[..., :sh, :sw]
            if scale < 1:
                alpha = F.interpolate(alpha, size=(h, w), mode="bilinear", align_corners=False)

            # Keep the trimap's certain regions exact; VitMatte only decides the band.
            alpha = torch.where(tri >= 0.75, torch.ones_like(alpha),
                                torch.where(tri <= 0.25, torch.zeros_like(alpha), alpha))
            alpha = (alpha / white_point).clamp(0, 1)
            out.append(alpha[0, 0].cpu())
            pbar.update(1)

        model.to(mm.unet_offload_device())
        mm.soft_empty_cache()
        return (torch.stack(out),)


class MatteTemporalSmooth:
    """Damps edge 'boiling' by blending each frame's edge with the previous smoothed frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "alpha": ("MASK",),
            "strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.95, "step": 0.05,
                                   "tooltip": "0 = off. Higher = calmer edges but more trailing on fast motion."}),
            "motion_threshold": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                                           "tooltip": "Where alpha changes more than this between frames, real motion is assumed and no blending happens."}),
        }, "optional": {"trimap": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "run"
    CATEGORY = "MAGIC ROTO"

    def run(self, alpha, strength, motion_threshold, trimap=None):
        if strength <= 0 or alpha.shape[0] < 2:
            return (alpha,)
        out = [alpha[0]]
        for i in range(1, alpha.shape[0]):
            cur, prev = alpha[i], out[-1]
            still = (cur - prev).abs() < motion_threshold
            if trimap is not None:
                t = trimap[min(i, trimap.shape[0] - 1)]
                still &= (t > 0.25) & (t < 0.75)
            out.append(torch.where(still, (1 - strength) * cur + strength * prev, cur))
        return (torch.stack(out),)


class MattePreviewOnColor:
    """Composites the footage over a solid colour using the matte, sized to the footage automatically."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "alpha": ("MASK",),
            "color": ("STRING", {"default": "#00FF00", "tooltip": "Background colour as hex, e.g. #00FF00 (green) or #000000."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "MAGIC ROTO"

    def run(self, image, alpha, color):
        hexcode = color.strip().lstrip("#")
        if len(hexcode) != 6:
            raise ValueError(f"color must be a 6-digit hex like #00FF00, got {color!r}")
        rgb = torch.tensor([int(hexcode[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=image.dtype)
        a = alpha.to(image.dtype)
        if a.shape[-2:] != image.shape[1:3]:
            a = F.interpolate(a[:, None], size=image.shape[1:3], mode="bilinear", align_corners=False)[:, 0]
        if a.shape[0] == 1 and image.shape[0] > 1:
            a = a.expand(image.shape[0], -1, -1)
        a = a[..., None]
        return (image * a + rgb.view(1, 1, 1, 3) * (1 - a),)


NODE_CLASS_MAPPINGS = {
    "VitMatteTrimap": VitMatteTrimap,
    "VitMatteRefine": VitMatteRefine,
    "MatteTemporalSmooth": MatteTemporalSmooth,
    "MattePreviewOnColor": MattePreviewOnColor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VitMatteTrimap": "MAGIC ROTO: Mask to Trimap",
    "VitMatteRefine": "MAGIC ROTO: Refine Edges",
    "MatteTemporalSmooth": "MAGIC ROTO: Temporal Smooth",
    "MattePreviewOnColor": "MAGIC ROTO: Preview on Colour",
}
