import cv2


def load_video_frames_from_path(video_path, start_frame=0, max_frames=-1):
    """
    Load frames from a video file with OpenCV (BGR uint8).
    Returns (frames, fps).
    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start_frame:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames > 0 and len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    assert len(frames) > 0, "No frames read"
    return frames, fps

def write_video_frames_to_path(out_video, mask_frames, fps, H0, W0):
    # ---- write output video ----
    writer = cv2.VideoWriter(
        out_video,
        cv2.VideoWriter_fourcc(*"FFV1"),  # lossless; switch to MJPG/mp4v if needed
        fps,
        (W0, H0)
    )
    assert writer.isOpened(), "Failed to open VideoWriter (FFV1/MKV). Try MJPG or mp4v if needed."
    for f in mask_frames:
        if f.shape[0] != H0 or f.shape[1] != W0:
            f = cv2.resize(f, (W0, H0), interpolation=cv2.INTER_NEAREST)
        writer.write(f)
    writer.release()
    print(f"[ok] wrote {len(mask_frames)} frames to {out_video}")
