# Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| Can't find `Run SAM3 Video Track` | ComfyUI is too old | Update ComfyUI. The SAM3 nodes are core nodes in recent versions (tested on 0.38.2) |
| No **MAGIC ROTO** nodes after install | Folder in the wrong place, or ComfyUI not restarted | The folder must be `ComfyUI/custom_nodes/comfyui-magic-roto/` with `__init__.py` directly inside. Restart ComfyUI and check its console for import errors |
| `ckpt_name` error on load | SAM3 model missing | Run `scripts/download_models.py`, or put `sam3.1_multiplex_fp16.safetensors` in `models/checkpoints/` |
| `model_name` list is empty or wrong | VitMatte model missing | It must be in `models/vitmatte/vitmatte-small-composition-1k/` with `config.json` and `model.safetensors` |
| Mask is empty (all black) | Prompt didn't match, or threshold too high | Use a simpler noun (`person`, `cat`), lower `detection_threshold` to 0.3, and check `sam_raw_*.mp4` |
| Wrong object is cut out | Prompt too vague, or several matches | Be more specific, lower `max_objects`, or raise `detection_threshold` |
| Part of the subject is missing (a held item, for example) | SAM3 treats it as a separate object | Add it to the prompt (`person, guitar`) and raise `max_objects` |
| Edges look hard or cut-out | Edge band too narrow | Increase `dilate_px` (20 → 35) |
| Halo or background bleeding into the edge | Band too wide, or the mask boundary is off | Lower `dilate_px`, increase `erode_px` |
| Edges flicker ("boil") | Per-frame matting | Raise Temporal Smooth `strength` (0.4 → 0.6) on slow shots |
| Ghost trails on fast motion | Smoothing too strong | Lower `strength`, or lower `motion_threshold` |
| CUDA out of memory | Frames too big | Lower `max_side` (1280 → 960) and use a shorter clip |
| ComfyUI crashes or the PC freezes on long clips | All frames are held in RAM | Split the clip into ≤ 10 s parts |
| Preview video won't save | Codec option missing in older ComfyUI | Set the `SaveVideo` codec to `h264` in the node |
