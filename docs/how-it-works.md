# How MAGIC ROTO works

## The core idea

Cutting a subject out of video is two different problems, so MAGIC ROTO uses two different AIs.

| Problem | Question | Model |
|---|---|---|
| Segmentation | *Which pixels belong to the object?* A yes/no mask. | **SAM3** |
| Matting | *How much of each edge pixel is the object?* A 0–1 alpha. | **VitMatte** |

SAM3 is excellent at finding and following an object, but its edges are hard and blocky.
VitMatte is excellent at fine edges, but it can't find objects. It has to be told where the edge is.
Chaining them gives you both.

## Step by step

### 1. Load the video
`LoadVideo` → `GetVideoComponents` splits the clip into frames (an `IMAGE` batch) and also outputs its frame rate, which the preview videos reuse.

### 2. Find the object (SAM3)
The text prompt is encoded with SAM3's own text encoder (`CLIPTextEncode` on the SAM3 checkpoint's CLIP output).
SAM3 runs at a fixed internal resolution and returns a mask for each detected instance.

### 3. Track it through the clip (SAM3 video tracking)
`Run SAM3 Video Track` follows each object from frame to frame using SAM3's memory of what the object looked like.
With `detect_interval = 1`, detection also runs on every frame, which lets the track recover after a scene cut.
`SAM3 Track to Mask` then merges the tracked objects into one mask per frame (leave `object_indices` empty to keep all of them).

### 4. Build a trimap
VitMatte needs a **trimap**, a map with three zones:

```
1.0 (white) = definitely the object      ← the mask eroded inward by erode_px
0.5 (grey)  = unknown: the edge band     ← between the eroded and dilated mask
0.0 (black) = definitely background      ← outside the mask dilated outward by dilate_px
```

A wider grey band gives VitMatte more room to find hair, glow and motion blur, but it's slower and riskier.
A narrower band is tighter and faster.

### 5. Refine the edge (VitMatte)
`MAGIC ROTO: Refine Edges` sends each frame plus its trimap through VitMatte:

- The frame is normalised the way VitMatte was trained (`(x − 0.5) / 0.5`), and the trimap is passed as a 4th channel.
- Frames larger than `max_side` are downscaled for inference, then the alpha is upscaled back.
- The input is padded to a multiple of 32 (a VitMatte requirement) and cropped back afterwards.
- It runs in **fp16 on the GPU** (about 0.6 GB VRAM at 1280 px).
- Afterwards the **certain** regions of the trimap are forced back to exact 0 and 1, so VitMatte only ever decides the grey band.
- `white_point` remaps near-white values to solid 1.0, removing the "almost opaque" interior that AI mattes often have.

### 6. Calm the flicker (temporal smoothing)
VitMatte works **one frame at a time** with no memory, so edge pixels shift slightly between frames because of noise, compression and grain.
On video this looks like a fine tremor, or "boiling".

`MAGIC ROTO: Temporal Smooth` blends each frame with the previous smoothed frame, **but only where**:

- the pixel is in the trimap's unknown band, if a trimap is connected, and
- the alpha changed less than `motion_threshold`, meaning it looks like flicker rather than real movement.

Real motion passes through untouched; small jitter is damped.

### 7. Save
- `SaveImage` writes the matte as a numbered **PNG sequence**, which is the deliverable.
- Three `SaveVideo` nodes write previews: the matte, the cut-out on green, and the raw SAM3 mask.

## Image workflow: how clicks, boxes and text work together

ComfyUI's core `SAM3 Detect` node accepts text, boxes and points, but on its own they interfere:
when text is present, boxes stop being objects and only limit where the text is searched; and an empty text box still counts as a text prompt.

`MAGIC ROTO: Select Objects` avoids this by calling SAM3 **separately for each prompt type**:

1. **Points** (if any): all include/exclude clicks → one mask.
2. **Boxes** (if any): run without text, so each box → its own mask.
3. **Text** (only if the field isn't empty): each match → its own mask.

The masks are stacked into one batch (one entry per object) and flow through the same trimap → VitMatte steps.
The refine node reuses the single image for every object, so the outputs are one cut-out, one matte and one on-colour image per object.

Clicks from the Points Editor are in the editor's own canvas pixels. The node reads the editor's size from its `bbox_mask` output and rescales the clicks and boxes to the real image size.

## Why not just use SAM3's mask?
Compare the raw SAM3 mask with the refined matte in `assets/pipeline_result.png`.
The raw mask has a hard, stair-stepped edge that looks cut out when composited.
The refined matte has a soft falloff that blends with any new background.
