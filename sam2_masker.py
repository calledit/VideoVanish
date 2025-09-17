
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse, json, cv2, torch, numpy as np


ap = argparse.ArgumentParser(description=(
        'Create mask with SAM2'
    ))
ap.add_argument("--color_video", required=True, type=str)
ap.add_argument("--annotations", required=True, type=str, help="JSON annotation file")
ap.add_argument("--max_frames", type=int, default=-1)
args = ap.parse_args()

assert os.path.isfile(args.color_video), "input video missing"
out_video = args.color_video + "_sam2_mask.mkv"

# ---- read video & metadata ----
cap = cv2.VideoCapture(args.color_video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


frames = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame)
    if args.max_frames > 0 and len(frames) >= args.max_frames:
        break
cap.release()
N = len(frames)
assert N > 0, "No frames read"

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


#READ JSON AND convert to LABELS
with open(args.annotations, "r") as f:
    ann = json.load(f)

def _to_px_x(x):  # accepts 0..1 normalized or absolute pixels
    return float(x) * W if 0.0 <= x <= 1.0 else float(x)

def _to_px_y(y):
    return float(y) * H if 0.0 <= y <= 1.0 else float(y)

def denorm_point_xy(x, y):
    return np.array([_to_px_x(x), _to_px_y(y)], dtype=np.float32)

def denorm_rect_xywh_to_xyxy(x, y, w, h):
    x1, y1 = _to_px_x(x), _to_px_y(y)
    # w/h may be normalized or absolute; convert appropriately
    x2 = _to_px_x(x + w) if 0.0 <= w <= 1.0 else (x1 + float(w))
    y2 = _to_px_y(y + h) if 0.0 <= h <= 1.0 else (y1 + float(h))
    # ensure proper ordering
    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

# init inference state from your already-read numpy frames (or a path)
inference_state = predictor.init_state(video_path=frames)

keyframes = sorted(ann.get("keyframes", []), key=lambda k: int(k["frame_idx"]))

for kf in keyframes:
    frame_idx = int(kf["frame_idx"])

    # Group clicks per object id
    clicks_by_obj = {}   # obj_id -> { "pts": [xy], "labels": [0/1] }
    def _add_click(obj_id, x, y, label):
        d = clicks_by_obj.setdefault(int(obj_id), {"pts": [], "labels": []})
        d["pts"].append(denorm_point_xy(x, y))
        d["labels"].append(label)

    for c in kf.get("pos_clicks", []):
        _add_click(c.get("obj", 1), c["x"], c["y"], 1)
    for c in kf.get("neg_clicks", []):
        _add_click(c.get("obj", 1), c["x"], c["y"], 0)

    # Send all clicks per obj in one go
    for obj_id, d in clicks_by_obj.items():
        points = np.vstack(d["pts"]).astype(np.float32)
        labels = np.array(d["labels"], dtype=np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

    # Rects (x,y,w,h) per object; send one box call per rect
    for r in kf.get("rects", []):
        obj_id = int(r.get("obj", 1))
        box = denorm_rect_xywh_to_xyxy(r["x"], r["y"], r["w"], r["h"])
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )


# 1) Propagate and collect masks (as in your example)
video_segments = {}  # {frame_idx: {obj_id: mask_bool[h,w]}}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        int(out_obj_id): (out_mask_logits[i] > 0.0).detach().cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# 2) Utilities for coloring/rendering
def _bgr_from_tab10(idx):
    # matplotlib tab10 (approx) in RGB, then convert to BGR for OpenCV
    tab10 = np.array([
        [31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189],
        [140,86,75],[227,119,194],[127,127,127],[188,189,34],[23,190,207]
    ], dtype=np.float32) / 255.0
    rgb = tab10[idx % len(tab10)]
    bgr = rgb[::-1]
    return bgr

def overlay_masks_on_frame(frame_bgr, masks_dict, alpha=0.6, draw_contours=False):
    """
    frame_bgr: HxWx3 uint8 (OpenCV frame)
    masks_dict: {obj_id: mask_bool[h,w]}
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty frame passed to overlay_masks_on_frame")

    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    overlay = np.zeros_like(out, dtype=np.float32)
    any_mask = np.zeros((h, w), dtype=bool)

    for obj_id, m in masks_dict.items():
        if m is None or m.size == 0:
            continue

        # make sure m is 2D bool
        m = np.asarray(m)
        if m.ndim > 2:
            m = m.squeeze()
        if m.ndim != 2:
            raise ValueError(f"Mask for obj_id {obj_id} has unexpected shape {m.shape}")

        # resize mask to frame if needed
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        color_bgr = _bgr_from_tab10(int(obj_id))  # in 0..1
        color_vec = (color_bgr * 255.0).astype(np.float32)

        # Paint overlay where mask==True (use 2D mask, broadcast over last dim)
        overlay[m] = color_vec
        any_mask |= m

    # Alpha blend only where there was any mask
    if any_mask.any():
        alpha_mask = (any_mask.astype(np.float32) * alpha)[..., None]  # HxWx1
        base = out.astype(np.float32)
        out = (base * (1.0 - alpha_mask) + overlay * alpha_mask).astype(np.uint8)

    # Optional crisp contours
    if draw_contours and any_mask.any():
        for obj_id, m in masks_dict.items():
            if m is None or m.size == 0:
                continue
            m = np.asarray(m)
            if m.ndim > 2:
                m = m.squeeze()
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            m_uint8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color_255 = tuple(((_bgr_from_tab10(int(obj_id)) * 255).astype(np.uint8)).tolist())
            cv2.drawContours(out, contours, -1, color_255, thickness=2)

    return out


# 3) Build a list of rendered frames in original order
# You already have `frames` (BGR from cv2). We'll respect their size.
H0, W0 = frames[0].shape[:2]

print(H0, W0)
rendered_frames = []
for idx in range(len(frames)):
    masks_dict = video_segments.get(idx, {})  # empty dict -> no overlay
    rendered = overlay_masks_on_frame(frames[idx], masks_dict, alpha=0.6, draw_contours=True)
    rendered_frames.append(rendered)

# 4) Write with FFV1 in MKV (lossless)
output_video_path = out_video  # or set explicitly
fps_out = fps  # reuse input fps
rescale_width, rescale_height = W0, H0  # keep original size

out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"FFV1"),
    fps_out,
    (int(rescale_width), int(rescale_height))
)
assert out.isOpened(), "Failed to open VideoWriter (FFV1/MKV). Try another fourcc if needed."

for f in rendered_frames:
    if f.shape[1] != rescale_width or f.shape[0] != rescale_height:
        f = cv2.resize(f, (rescale_width, rescale_height), interpolation=cv2.INTER_LINEAR)
    out.write(f)

out.release()
print(f"[ok] wrote {len(rendered_frames)} frames to {output_video_path}")
