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

        # One image with several object trimaps, several frames with one trimap, or frame-for-frame.
        frames = max(image.shape[0], trimap.shape[0])
        if image.shape[0] not in (1, frames) or trimap.shape[0] not in (1, frames):
            raise ValueError(f"trimap has {trimap.shape[0]} entries but image has {image.shape[0]}")
        pbar = comfy.utils.ProgressBar(frames)
        out = []
        for i in range(frames):
            img = image[min(i, image.shape[0] - 1)].to(device).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
            tri = trimap[min(i, trimap.shape[0] - 1)].to(device)[None, None]
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
        # One frame at a time, so long clips don't need extra full-size temporary copies in RAM.
        n, h, w = _batch_size(image, alpha), image.shape[1], image.shape[2]
        out = torch.empty((n, h, w, 3), dtype=image.dtype)
        for i, img, a in _frames(image, alpha):
            out[i] = img * a[..., None] + rgb * (1 - a[..., None])
        return (out,)


def _batch_size(image, alpha):
    n = max(image.shape[0], alpha.shape[0] if alpha.dim() == 3 else 1)
    a_n = alpha.shape[0] if alpha.dim() == 3 else 1
    if image.shape[0] not in (1, n) or a_n not in (1, n):
        raise ValueError(f"alpha has {a_n} entries but image has {image.shape[0]}")
    return n


def _frames(image, alpha, device=None):
    """Yields (index, image[H,W,3], alpha[H,W]) one frame at a time, alpha resized to the image."""
    if alpha.dim() == 2:
        alpha = alpha[None]
    h, w = image.shape[1:3]
    for i in range(_batch_size(image, alpha)):
        img = image[min(i, image.shape[0] - 1)][..., :3]
        a = alpha[min(i, alpha.shape[0] - 1)].to(image.dtype)
        if device is not None:
            img, a = img.to(device), a.to(device)
        if a.shape != (h, w):
            a = F.interpolate(a[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        yield i, img, a


def _match_batches(image, alpha):
    """Resizes alpha to the image and broadcasts whichever batch has one entry. Returns (image, alpha[..., None])."""
    a = alpha.to(image.dtype)
    if a.dim() == 2:
        a = a[None]
    if a.shape[-2:] != image.shape[1:3]:
        a = F.interpolate(a[:, None], size=image.shape[1:3], mode="bilinear", align_corners=False)[:, 0]
    n = max(image.shape[0], a.shape[0])
    if image.shape[0] not in (1, n) or a.shape[0] not in (1, n):
        raise ValueError(f"alpha has {a.shape[0]} entries but image has {image.shape[0]}")
    return image.expand(n, -1, -1, -1), a.expand(n, -1, -1)[..., None]


class MatteCutout:
    """Footage + matte -> RGBA cut-out with a transparent background (saves as transparent PNG)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "alpha": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba",)
    FUNCTION = "run"
    CATEGORY = "MAGIC ROTO"

    def run(self, image, alpha):
        image, a = _match_batches(image[..., :3], alpha)
        return (torch.cat([image, a], dim=-1),)


def _box_blur(x, radius, passes=3):
    """Approximate Gaussian blur of x[C,H,W] by repeated box filters (edge-aware averaging)."""
    k = 2 * radius + 1
    x = x[None]
    for _ in range(passes):
        x = F.avg_pool2d(x, k, stride=1, padding=radius, count_include_pad=False)
    return x[0]


class MatteCleanEdges:
    """Removes green/blue screen spill and fixes edge colours so cut-outs don't carry a coloured fringe.

    1. Despill: caps the screen channel at the brighter of the other two ("balanced", keeps yellows),
       or at their average ("strong").
    2. Edge colour fix: semi-transparent edge pixels still contain background colour mixed in. They are
       replaced by colour pulled inward from the nearby solid part of the subject, more strongly the more
       transparent the pixel is. The matte itself is not changed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "alpha": ("MASK",),
            "screen_color": (["green", "blue", "none"], {"default": "green",
                             "tooltip": "Colour of the screen behind the subject. 'none' skips despill and only fixes edge colours."}),
            "despill_method": (["balanced (keeps yellows)", "strong (average)"], {"default": "balanced (keeps yellows)",
                               "tooltip": "Balanced caps green at the brighter of red/blue, so yellow, orange and skin keep their colour. Strong caps it at their average: removes more spill but shifts yellows towards orange."}),
            "despill_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "1 = remove all excess green/blue. Lower it if real green/blue parts of the subject turn grey."}),
            "despill_area": (["whole subject", "edges only"], {"default": "whole subject",
                             "tooltip": "'whole subject' also removes green light bounced onto the subject. 'edges only' protects genuinely green objects."}),
            "edge_color_fix": ("BOOLEAN", {"default": True,
                               "tooltip": "Replace the background colour mixed into soft edges with the subject's own colour."}),
            "edge_reach_px": ("INT", {"default": 20, "min": 1, "max": 64,
                              "tooltip": "How far (roughly x3) colour is pulled in from the solid subject. Raise it for wide fur/hair edges."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "MAGIC ROTO"

    def run(self, image, alpha, screen_color, despill_method, despill_strength, despill_area, edge_color_fix, edge_reach_px):
        device = mm.get_torch_device()
        n, h, w = _batch_size(image, alpha), image.shape[1], image.shape[2]
        out = torch.empty((n, h, w, 3), dtype=image.dtype)
        pbar = comfy.utils.ProgressBar(n)
        for i, img, a in _frames(image, alpha, device):
            c = img.float().permute(2, 0, 1).clone()       # 3,H,W
            a = a.float()

            if screen_color != "none" and despill_strength > 0:
                s = 1 if screen_color == "green" else 2
                o1, o2 = [ch for ch in (0, 1, 2) if ch != s]
                limit = torch.maximum(c[o1], c[o2]) if despill_method.startswith("balanced") else (c[o1] + c[o2]) / 2
                excess = (c[s] - limit).clamp(min=0)
                weight = despill_strength * (a > 0.004).float()
                if despill_area == "edges only":
                    band = ((a > 0.004) & (a < 0.98)).float()
                    band = F.max_pool2d(band[None, None], 2 * edge_reach_px + 1, stride=1, padding=edge_reach_px)[0, 0]
                    weight = weight * band
                c[s] = c[s] - excess * weight

            if edge_color_fix:
                solid = (a >= 0.95).float()[None]
                den = _box_blur(solid, edge_reach_px)
                num = _box_blur(c * solid, edge_reach_px)
                est = num / den.clamp(min=1e-4)
                t = ((0.95 - a) / 0.95).clamp(0, 1) * (den[0] > 1e-3).float()
                c = c * (1 - t) + est * t

            out[i] = c.clamp(0, 1).permute(1, 2, 0).to(out.dtype).cpu()
            pbar.update(1)
        mm.soft_empty_cache()
        return (out,)


class MagicRotoSelect:
    """Runs SAM3 once per prompt type so points, boxes and text all work together.

    Points (all green/red clicks) = one object. Each box = one object. Text = up to
    N objects per comma-separated term ('person:3'). Returns one mask per object.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "image": ("IMAGE",),
            "text": ("STRING", {"default": "", "multiline": True,
                                "tooltip": "Optional. What to find, e.g. 'person' or 'dog, ball'. Add ':N' for up to N of a term ('person:3'). Leave empty to use only clicks/boxes."}),
            "threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Text detection confidence. Raise it if wrong things are found, lower it if the object is missed."}),
            "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5}),
        }, "optional": {
            "positive_coords": ("STRING", {"forceInput": True}),
            "negative_coords": ("STRING", {"forceInput": True}),
            "bboxes": ("BBOX", {"forceInput": True}),
            "editor_mask": ("MASK", {"tooltip": "Connect the Points Editor's bbox_mask: it tells this node the editor's canvas size so clicks map exactly onto the image."}),
        }}

    RETURN_TYPES = ("MASK", "INT", "STRING")
    RETURN_NAMES = ("masks", "count", "labels")
    FUNCTION = "run"
    CATEGORY = "MAGIC ROTO"

    def run(self, model, clip, image, text, threshold, refine_iterations,
            positive_coords=None, negative_coords=None, bboxes=None, editor_mask=None):
        import json
        from comfy_extras.nodes_sam3 import SAM3_Detect

        image = image[:1]
        H, W = image.shape[1:3]
        sx = sy = 1.0
        if editor_mask is not None and editor_mask.dim() >= 2:
            eh, ew = editor_mask.shape[-2:]
            sx, sy = W / ew, H / eh

        def detect(**kw):
            out = SAM3_Detect.execute(model=model, image=image, threshold=threshold,
                                      refine_iterations=refine_iterations, individual_masks=True, **kw)
            return out.result[0]

        def scaled(points_json):
            pts = json.loads(points_json) if points_json else []
            return [{"x": round(p["x"] * sx), "y": round(p["y"] * sy)} for p in pts]

        masks, labels = [], []
        pos, neg = scaled(positive_coords), scaled(negative_coords)
        if pos:
            m = detect(positive_coords=json.dumps(pos), negative_coords=json.dumps(neg) if neg else None)
            masks.append(m[:1])
            labels.append("points")

        boxes = []
        for b in (bboxes or []):
            # Points Editor sends (x0, y0, x1, y1) tuples, or its raw [{}] placeholder when no box is drawn.
            if isinstance(b, dict):
                if not all(k in b for k in ("startX", "startY", "endX", "endY")):
                    continue
                b = (min(b["startX"], b["endX"]), min(b["startY"], b["endY"]),
                     max(b["startX"], b["endX"]), max(b["startY"], b["endY"]))
            elif not isinstance(b, (list, tuple)) or len(b) < 4:
                continue
            x0, y0, x1, y1 = (float(v) for v in b[:4])
            if x1 > x0 and y1 > y0:
                boxes.append({"x": x0 * sx, "y": y0 * sy, "width": (x1 - x0) * sx, "height": (y1 - y0) * sy})
        if boxes:
            m = detect(bboxes=boxes)
            masks.append(m)
            labels += [f"box{i + 1}" for i in range(m.shape[0])]

        if text.strip():
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(text.strip()))
            m = detect(conditioning=cond)
            masks.append(m)
            labels += [f"text{i + 1}" for i in range(m.shape[0])]

        if not masks or sum(m.shape[0] for m in masks) == 0:
            raise ValueError("MAGIC ROTO found nothing to cut out. Shift+click the object or Ctrl+drag a box "
                             "in the Points Editor, or type what to find in 'text' (try lowering threshold).")
        out = torch.cat([m.float().cpu() for m in masks], dim=0)
        return (out, out.shape[0], ", ".join(labels))


NODE_CLASS_MAPPINGS = {
    "VitMatteTrimap": VitMatteTrimap,
    "VitMatteRefine": VitMatteRefine,
    "MatteTemporalSmooth": MatteTemporalSmooth,
    "MattePreviewOnColor": MattePreviewOnColor,
    "MatteCutout": MatteCutout,
    "MatteCleanEdges": MatteCleanEdges,
    "MagicRotoSelect": MagicRotoSelect,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VitMatteTrimap": "MAGIC ROTO: Mask to Trimap",
    "VitMatteRefine": "MAGIC ROTO: Refine Edges",
    "MatteTemporalSmooth": "MAGIC ROTO: Temporal Smooth",
    "MattePreviewOnColor": "MAGIC ROTO: Preview on Colour",
    "MatteCutout": "MAGIC ROTO: Transparent Cut-out",
    "MatteCleanEdges": "MAGIC ROTO: Remove Green Spill / Clean Edge Colours",
    "MagicRotoSelect": "MAGIC ROTO: Select Objects (points + boxes + text)",
}
