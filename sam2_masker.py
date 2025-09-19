#!/usr/bin/env python3
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import cv2
import torch
import numpy as np

import tools

from sam2.build_sam import build_sam2_video_predictor

# =============================
# Hardcoded SAM2 config/checkpoint
# =============================
SAM2_CHECKPOINT = "sam2_numpy_frames/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = None

# =============================
# Color mapping per object id
# =============================
def color_for_obj(obj_id):
    """
    Deterministic, bright BGR color for a given obj_id using HSV cycling.
    """
    # OpenCV HSV H in [0,179]
    h = int((obj_id * 37) % 180)       # step by 37 to spread hues
    s = 200
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in bgr)  # (B, G, R)


# =============================
# Library API
# =============================
def run_sam2_on_frames(frames_rgb, annotations, device=None):
    """
    Run SAM2 segmentation on a list of frames, then return COLORED mask frames
    (black background; each obj_id rendered in its own solid color).

    Args:
        frames_rgb (list[np.ndarray]): list of (H,W,3) BGR uint8 frames.
        annotations (dict): JSON-like dict with:
            keyframes: [{
              frame_idx: int,
              pos_clicks: [{x,y,obj}], neg_clicks: [{x,y,obj}],
              rects: [{x,y,w,h,obj}]
            }, ...]
        device (torch.device or None)

    Returns:
        list[np.ndarray]: list of (H,W,3) BGR uint8 colored-mask frames.
    """
    global predictor
    assert isinstance(frames_rgb, (list, tuple)) and len(frames_rgb) > 0, "frames must be a non-empty list"
    H0, W0 = frames_rgb[0].shape[:2]

    # Select device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available() and device.type == "mps":
        print("[warn] MPS support is preliminary; outputs may differ from CUDA.")

    if predictor is None:
        predictor = build_sam2_video_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)

    # SAM2 examples load RGB via decord; convert BGR->RGB for the model to be safe
    inference_state = predictor.init_state(video_path=frames_rgb)

    # ----- coordinate helpers (accept normalized [0..1] or absolute pixels) -----
    def _to_px_x(x): return float(x) * W0 if 0.0 <= x <= 1.0 else float(x)
    def _to_px_y(y): return float(y) * H0 if 0.0 <= y <= 1.0 else float(y)
    def denorm_point_xy(x, y): return np.array([_to_px_x(x), _to_px_y(y)], dtype=np.float32)
    def denorm_rect_xywh_to_xyxy(x, y, w, h):
        x1, y1 = _to_px_x(x), _to_px_y(y)
        x2 = _to_px_x(x + w) if 0.0 <= w <= 1.0 else (x1 + float(w))
        y2 = _to_px_y(y + h) if 0.0 <= h <= 1.0 else (y1 + float(h))
        return np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)

    # ----- feed annotations (per-frame, per-obj) -----
    keyframes = sorted(annotations.get("keyframes", []), key=lambda k: int(k["frame_idx"]))
    for kf in keyframes:
        frame_idx = int(kf["frame_idx"])
        clicks_by_obj = {}  # obj_id -> { "pts": [xy], "labels": [0/1] }

        def _add_click(obj_id, x, y, label):
            d = clicks_by_obj.setdefault(int(obj_id), {"pts": [], "labels": []})
            d["pts"].append(denorm_point_xy(x, y))
            d["labels"].append(label)

        for c in kf.get("pos_clicks", []):
            _add_click(c.get("obj", 1), c["x"], c["y"], 1)
        for c in kf.get("neg_clicks", []):
            _add_click(c.get("obj", 1), c["x"], c["y"], 0)

        # batch clicks per object
        for obj_id, d in clicks_by_obj.items():
            pts = np.vstack(d["pts"]).astype(np.float32)
            labels = np.array(d["labels"], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                points=pts,
                labels=labels,
            )

        # rects: x,y,w,h (top-left + size)
        for r in kf.get("rects", []):
            obj_id = int(r.get("obj", 1))
            box = denorm_rect_xywh_to_xyxy(r["x"], r["y"], r["w"], r["h"])
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
            )

    # ----- propagate and collect per-frame masks -----
    video_segments = {}  # {frame_idx: {obj_id: mask_bool[h,w]}}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            int(out_obj_id): (out_mask_logits[i] > 0.0).detach().cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # ----- build COLORED mask frames -----
    mask_frames = []
    for idx in range(len(frames_rgb)):
        masks_dict = video_segments.get(idx, {})
        # start black canvas
        out = np.zeros((H0, W0, 3), dtype=np.uint8)

        # draw in a consistent order so overlaps are deterministic
        # rule: higher obj_id overwrites lower obj_id where they overlap
        for obj_id in sorted(masks_dict.keys()):
            m = masks_dict[obj_id]
            if m is None or m.size == 0:
                continue
            m = np.asarray(m)
            if m.ndim > 2:
                m = m.squeeze()
            if m.shape != (H0, W0):
                m = cv2.resize(m.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                m = m.astype(bool)

            color = color_for_obj(int(obj_id))  # (B,G,R)
            # paint color where mask is True
            out[m] = color

        mask_frames.append(out)

    return mask_frames


# =============================
# CLI entry point
# =============================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Create colored mask video with SAM2 (one color per object, black background).")
    ap.add_argument("--color_video", required=True, type=str, help="Input color video path.")
    ap.add_argument("--annotations", required=True, type=str, help="JSON annotation file.")
    ap.add_argument("--start_frame", type=int, default=0, help="Index of first frame to process (default: 0).")
    ap.add_argument("--max_frames", type=int, default=-1, help="Max number of frames to process after start_frame.")
    ap.add_argument("--out", type=str, default=None, help="Output video path (default: <input>_sam2_mask.mkv)")
    args = ap.parse_args()

    assert os.path.isfile(args.color_video), "input video missing"
    out_video = args.out or (args.color_video + "_sam2_mask.mkv")

    # ---- load frames ----
    frames, fps = tools.load_video_frames_from_path(args.color_video, args.start_frame, args.max_frames)
    H0, W0 = frames[0].shape[:2]

    # ---- read annotations ----
    with open(args.annotations, "r") as f:
        ann = json.load(f)

    # ---- run model ----
    mask_frames = run_sam2_on_frames(frames, ann)
    tools.write_video_frames_to_path(out_video, mask_frames, fps, H0, W0)
    

if __name__ == "__main__":
    main()

