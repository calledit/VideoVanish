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


device = None
last_ckpt = None
video_inpainting_sd = None
propainter = None

def run_infill_on_frames(frames_rgb, mask_frames, mask_dilation_iter = 8, ckpt = "2-Step", propainer_frames = None, max_img_size = 960):
    global device, last_ckpt, video_inpainting_sd, propainter
    # PCM params
    if last_ckpt != ckpt:
        device = get_device()
        ckpt = "2-Step"
        last_ckpt = ckpt
        video_inpainting_sd = DiffuEraser(device, "stable-diffusion-v1-5/stable-diffusion-v1-5", "stabilityai/sd-vae-ft-mse", "lixiaowen/diffuEraser", ckpt=ckpt)

    H0, W0 = frames_rgb[0].shape[:2]

    if propainer_frames is None:
        if propainter is None:
            propainter = Propainter("ruffy369/propainter", device=device)

        propainer_frames = propainter.forward(frames_rgb, mask_frames, 
                        ref_stride=10, neighbor_length=10, subvideo_length=50,
                        mask_dilation = mask_dilation_iter) 



    ## diffueraser
    guidance_scale = None    # The default value is 0.  
    inpainted_frames = video_inpainting_sd.forward(frames_rgb, mask_frames, propainer_frames,
                                max_img_size = max_img_size, mask_dilation_iter=mask_dilation_iter,
                                guidance_scale=None)

    for i, f in enumerate(inpainted_frames):
        if f.shape[0] != H0 or f.shape[1] != W0:
            inpainted_frames[i] = cv2.resize(f, (W0, H0))
    
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


   
