# Settings reference

## MAGIC ROTO: Select Objects (image workflow)

| Setting / input | Default | What it does | Tips |
|---|---|---|---|
| `text` | empty | What to find by name. Comma-separate several terms; `term:N` keeps up to N of that term | Leave it empty to use only clicks and boxes. Without `:N`, each term keeps its single best match |
| `threshold` | 0.4 | Confidence needed for a **text** match | Raise it if wrong things are found. Lower it if the object is missed |
| `refine_iterations` | 2 | Extra SAM decoder passes that tighten each mask | 0 = raw detector mask. 2–3 is a good balance |
| `positive_coords` / `negative_coords` | from Points Editor | Include and exclude clicks. **All clicks together make one object** | Add red (exclude) points to cut away parts SAM3 grabbed by mistake |
| `bboxes` | from Points Editor | **Each box becomes its own object** | Draw boxes snug around the object |
| `editor_mask` | Points Editor `bbox_mask` | Tells the node the editor's canvas size, so clicks land exactly on the image | Keep it connected |

Outputs: `masks` (one per object, in the order points → boxes → text), `count`, and `labels` (e.g. `points, box1, text1`).

## MAGIC ROTO: Transparent Cut-out

No settings. Image + matte → RGBA image. Feed it to **Save Image** to get transparent PNGs.

## MAGIC ROTO: Remove Green Spill / Clean Edge Colours

Put it between your footage and the cut-out / preview nodes. It changes colours only, never the matte.

| Setting | Default | What it does | Tips |
|---|---|---|---|
| `screen_color` | none (in the workflows) | `green` / `blue` removes that screen's spill. `none` only fixes edge colours | Use `green` for green-screen footage, `none` for normal footage |
| `despill_method` | balanced (keeps yellows) | **Balanced:** caps green at the brighter of red/blue, so yellow, orange and skin tones stay true. **Strong:** caps it at their average, removing more spill but shifting yellows toward orange | Start with balanced. Try strong only if a green tint remains on neutral/grey areas |
| `despill_strength` | 1.0 | How much of the excess green/blue to remove | Lower it if genuinely green parts of the subject go grey |
| `despill_area` | whole subject | `whole subject` also removes green light bounced onto the subject. `edges only` protects green objects (a green shirt) | |
| `edge_color_fix` | on | Replaces the background colour mixed into soft edges with colour pulled in from the solid part of the subject | Turn off only to compare |
| `edge_reach_px` | 20 | How far colour is pulled in (about 3× this value) | Raise it (30–40) for very wide fur/hair edges |

Tested on a green-screen lion: the measured green excess on the subject dropped from 1.42 to 0.01, and yellow fur on non-screen footage kept the same average colour.

## Run SAM3 Video Track (core ComfyUI node)

| Setting | Default in workflow | What it does | When to change it |
|---|---|---|---|
| `detection_threshold` | 0.4 | How confident SAM3 must be to accept an object | Raise it (0.5–0.6) if it picks wrong things. Lower it (0.3) if it misses the subject |
| `max_objects` | 1 | How many objects to keep | Raise it for several people or a prompt with several items |
| `detect_interval` | 1 | Re-detect every N frames | Keep it at 1 for clips with cuts. 2–5 is faster on single continuous shots |

Text prompt (the **What to cut out** node): a short noun phrase. Separate several subjects with commas; `word:N` caps the count for that word.

## MAGIC ROTO: Mask to Trimap

| Setting | Default | What it does | Tips |
|---|---|---|---|
| `erode_px` | 8 | Shrinks the mask inward. Whatever stays is "definitely object" | Increase it if the edge of the mask is wrong or halo-y |
| `dilate_px` | 20 | Grows the mask outward. Beyond it is "definitely background" | **5–10** for hard edges (products, cars) · **20–40** for hair, fur, glow, motion blur |

## MAGIC ROTO: Refine Edges

| Setting | Default | What it does | Tips |
|---|---|---|---|
| `model_name` | vitmatte-small-composition-1k | Which VitMatte model in `models/vitmatte/` to use | Any VitMatte folder with a `config.json` shows up here |
| `max_side` | 1280 | Long side the frame is scaled to for VitMatte | Lower it (960) if you get out-of-memory errors. Raise it (1920) for 4K detail if you have VRAM |
| `white_point` | 0.99 | Alpha above this becomes solid 1.0 | Use 0.95 if the interior still looks semi-transparent |

## MAGIC ROTO: Temporal Smooth

| Setting | Default | What it does | Tips |
|---|---|---|---|
| `strength` | 0.4 | How much of the previous frame is blended in (0 = off) | 0.2–0.4 for moving subjects · 0.5–0.7 for locked-off, slow shots |
| `motion_threshold` | 0.25 | Alpha changes larger than this count as real motion and are not blended | Lower it if you see trails on fast motion |
| `trimap` (optional input) | connected | Limits smoothing to the edge band | Keep it connected so the solid interior is never touched |

## MAGIC ROTO: Preview on Colour

| Setting | Default | What it does |
|---|---|---|
| `color` | `#00FF00` | Background colour for the preview cut-out. Try `#000000` or `#FFFFFF` to check edges against dark and light backgrounds |
