# Using the matte in DaVinci Resolve (free version)

MAGIC ROTO saves the matte as a PNG sequence: `alpha_00001_.png`, `alpha_00002_.png`, and so on.
White means the subject and black means the background.

## 1. Import the sequence as one clip
1. On the **Media** page, browse to `ComfyUI/output/magic_roto/` in the Media Storage panel.
2. Resolve normally groups numbered frames into one clip. If you see single frames instead, open the Media Storage options menu (`…`) and turn **Show Individual Frames** off.
3. Drag the `alpha_[00001-000NN]_.png` clip into the Media Pool.
4. Right-click it → **Clip Attributes** → set the frame rate to match your source clip.

## Option A: Color page (grade only the subject or only the background)
1. Put your original clip on the timeline and open the **Color** page.
2. Open the **Media Pool** on the Color page and drag the matte clip into the **Node Editor**. It appears as an **External Matte** source.
3. Connect the external matte's output to the **key input** (the small triangle) of a corrector node.
4. That node now affects only the white area. Invert the key in the **Key** panel to affect only the background.

## Option B: Fusion page (real transparency, new background)
1. Select your original clip on the timeline and open the **Fusion** page. Its `MediaIn1` is your footage.
2. Drag the matte clip from the Media Pool into the node area. It becomes `MediaIn2`.
3. Add a **MatteControl** node:
   - connect `MediaIn1` (your footage) to its yellow **Background** input,
   - connect `MediaIn2` (the matte) to its green **Foreground** input,
   - in the MatteControl's **Matte** tab, set **Combine** to **Combine Luminance** (uses the matte's white as alpha),
   - turn on **Post-Multiply Image**.
4. Add a **Merge**: your new background goes into its **Background** input, the MatteControl output into its **Foreground** input. Then connect the Merge to `MediaOut1`.

## Option C: quickest, lower quality
The `on_green_*.mp4` preview can be keyed with Resolve's **3D Keyer** on the Color page.
It's faster to set up but loses edge quality compared with using the matte directly.

## Tips
- Keep the matte and the source clip at **the same length, frame rate and start frame**, or they'll drift apart.
- If the edge looks too hard, re-run MAGIC ROTO with a larger `dilate_px` rather than blurring the matte in Resolve.
- Menu names can differ slightly between Resolve versions.
