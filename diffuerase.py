import torch
import os 
import sys
sys.path.append("DiffuEraser_np_array")
import time
import tools
import argparse
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device
import cv2
import numpy as np
import scipy


device = None
last_ckpt = None
video_inpainting_sd = None
propainter = None
   
def run_infill_on_frames(frames_rgb, mask_frames, mask_dilation_iter=8, ckpt="2-Step", 
                         propainer_frames=None, max_img_size=960, keep_unmasked_original=True, feather_px=3, prog = None):
    global device, last_ckpt, video_inpainting_sd, propainter

    H0, W0 = frames_rgb[0].shape[:2]

    if prog is not None: prog(5, "dilating frames")
    dilated_mask_frames = []
    for m in mask_frames:
        m = np.any(m > 0, axis=2).astype(np.uint8)
        m = scipy.ndimage.binary_dilation(m > 0, iterations=mask_dilation_iter).astype(np.uint8)*255
        dilated_mask_frames.append(m)

    if prog is not None: prog(10, "loading weights")
    # PCM params
    if last_ckpt != ckpt:
        device = get_device()
        ckpt = "2-Step"
        last_ckpt = ckpt
        video_inpainting_sd = DiffuEraser(
            device, 
            "stable-diffusion-v1-5/stable-diffusion-v1-5", 
            "stabilityai/sd-vae-ft-mse", 
            "lixiaowen/diffuEraser", 
            ckpt=ckpt
        )

    if propainer_frames is None:
        if propainter is None:
            propainter = Propainter("ruffy369/propainter", device=device)

        if prog is not None: prog(20, "running propainter prior")
        propainer_frames = propainter.forward(
            frames_rgb, dilated_mask_frames,
            ref_stride=10, neighbor_length=10, subvideo_length=50,
            mask_dilation=0,
            progress = prog
        )

    if prog is not None: prog(50, "running DiffuEraser")
    # diffueraser
    guidance_scale = None  # default = 0
    inpainted_frames = video_inpainting_sd.forward(
        frames_rgb, dilated_mask_frames, propainer_frames,
        max_img_size=max_img_size, mask_dilation_iter=0,
        guidance_scale=guidance_scale,
        progress = prog
    )

    if prog is not None: prog(90, "resizing and merging finished frames")
    for i, f in enumerate(inpainted_frames):
        # resize back if necessary
        if f.shape[0] != H0 or f.shape[1] != W0:
            inpainted_frames[i] = cv2.resize(f, (W0, H0))

        if keep_unmasked_original:
            # --- robust mask prep ---
            m = dilated_mask_frames[i]
            # collapse 3ch mask -> 1ch by "any" across channels; or keep 1ch as-is
            if m.ndim == 3:
                m = np.any(m > 0, axis=2).astype(np.uint8)
            else:
                m = (m > 0).astype(np.uint8)

            # ensure mask size matches output frame size
            if m.shape[:2] != (H0, W0):
                m = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST)

            # binarize: >0 -> 255, 0 -> 0 (what distanceTransform expects)
            _, m_bin = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)
            inv_bin = cv2.bitwise_not(m_bin)

            # --- feathered alpha using distance transforms ---
            if feather_px > 0:
                # distances in pixels to the opposite region
                d_in  = cv2.distanceTransform(m_bin,  cv2.DIST_L2, 5)
                d_out = cv2.distanceTransform(inv_bin, cv2.DIST_L2, 5)

                # alpha: 1 inside, 0 outside, linear ramp of width ~2*feather_px around boundary
                alpha = 0.5 + (d_in - d_out) / (2.0 * float(feather_px))
                alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
            else:
                # hard composite
                alpha = (m_bin > 0).astype(np.float32)

            alpha3 = alpha[..., None]

            out  = inpainted_frames[i]
            orig = frames_rgb[i]
            if orig.dtype != out.dtype:
                orig = orig.astype(out.dtype)

            inpainted_frames[i] = np.clip(np.rint(alpha3 * out + (1.0 - alpha3) * orig), 0, 255).astype(np.uint8)

        return inpainted_frames



# =============================
# CLI entry point
# =============================
def main():
    ap = argparse.ArgumentParser(description="Create colored mask video with SAM2 (one color per object, black background).")
    ap.add_argument("--color_video", required=True, type=str, help="Input color video path.")
    ap.add_argument("--mask_video", required=True, type=str, help="Input mask video path.")
    ap.add_argument("--prior_video", required=False, type=str, help="Input prior video path.")
    ap.add_argument("--start_frame", type=int, default=0, help="Index of first frame to process (default: 0).")
    ap.add_argument("--max_frames", type=int, default=-1, help="Max number of frames to process after start_frame.")
    ap.add_argument("--out", type=str, default=None, help="Output video path (default: <input>_vanished.mkv)")
    args = ap.parse_args()

    assert os.path.isfile(args.color_video), "input video missing"
    out_video = args.out or (args.color_video + "_vanished.mkv")

    # ---- load frames ----
    frames, fps = tools.load_video_frames_from_path(args.color_video, args.start_frame, args.max_frames)
    H0, W0 = frames[0].shape[:2]

    mask_frames, mask_fps = tools.load_video_frames_from_path(args.mask_video, args.start_frame, args.max_frames)
    Hm, Wm = mask_frames[0].shape[:2]

    prior_frames = None
    if args.prior_video is None:
        prior_frames, prior_fps = tools.load_video_frames_from_path(args.prior_video, args.start_frame, args.max_frames)
        Hp, Wp = prior_frames[0].shape[:2]
        assert (H0 == Hp and W0 == Wp), "prior and color video are diffrent sizes"

    assert (H0 == Hm and W0 == Wm), "mask and color video are diffrent sizes"

    # ---- run model ----
    mask_frames = run_infill_on_frames(frames, mask_frames, propainer_frames = prior_frames)
    tools.write_video_frames_to_path(out_video, mask_frames, fps, H0, W0)


if __name__ == '__main__':
    main()


   
