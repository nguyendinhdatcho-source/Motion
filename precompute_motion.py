"""
Bước 1 – Tiền tính chuyển động cho từng video (chạy 1 lần, offline).

Với mỗi video:
  1. Đọc toàn bộ frame gốc (thường 30 frame).
  2. LOẠI frame đồng màu (std mỗi kênh <= 1/255) TRƯỚC khi tính chuyển động.
     Nếu không, cặp (frame đen -> frame có người) sẽ sinh ra optical flow giả rất lớn.
  3. Tính trên các cặp frame LIÊN TIẾP:
       - optical flow Farneback (u, v)  -> dùng cho mô hình flow
       - |frame_t - frame_{t-1}| (gray) -> dùng cho baseline frame-difference
  4. Lượng tử flow về uint8 (cắt ở ±BOUND pixel) và lưu .npz nén.

Lấy mẫu 16 bước thời gian được làm SAU, lúc train (để còn augment tốc độ).

Input: một CSV có cột `path` (đường dẫn video, tuyệt đối hoặc tương đối so với --data_root).
Nên dùng đúng CSV chia tập theo nhóm mà nhóm đã tạo ở mục 4.3 (đã loại 9 video lỗi).

Ví dụ:
  python precompute_motion.py --csv splits.csv --data_root /data/vsl --out_root /data/vsl_motion
"""
import argparse
import os
import shutil
import tempfile
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

BOUND = 20.0  # pixel; flow ngoài ±BOUND bị cắt. Xem clip_ratio trong motion_stats.csv để chỉnh.


def _read(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def read_frames(path):
    frames = _read(path)
    if not frames and not path.isascii():
        # OpenCV trên Windows thường không mở được đường dẫn có ký tự tiếng Việt
        # -> chép tạm sang một file tên ASCII rồi đọc.
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(path)[1])
        os.close(fd)
        try:
            shutil.copyfile(path, tmp)
            frames = _read(tmp)
        finally:
            os.remove(tmp)
    return frames


def _init_worker():
    cv2.setNumThreads(1)


def is_blank(frame_bgr):
    # cùng tiêu chí với mục 4.2: độ lệch chuẩn từng kênh <= 1/255 (tức <= 1 trên thang uint8)
    return bool((frame_bgr.reshape(-1, 3).std(axis=0) <= 1.0).all())


def out_path_for(video_path, data_root, out_root):
    rel = os.path.relpath(video_path, data_root)
    return os.path.join(out_root, os.path.splitext(rel)[0] + ".npz")


def process(job):
    video_path, out_path, size = job
    if os.path.exists(out_path):
        return None
    t0 = time.time()
    frames = read_frames(video_path)
    kept = [f for f in frames if not is_blank(f)]
    gray = [
        cv2.cvtColor(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        for f in kept
    ]

    flows, diffs = [], []
    for a, b in zip(gray[:-1], gray[1:]):
        # FPS chỉ ~12 nên chuyển động giữa 2 frame khá lớn -> dùng nhiều tầng pyramid + cửa sổ rộng
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, pyr_scale=0.5, levels=5, winsize=21,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flows.append(flow)
        diffs.append(cv2.absdiff(b, a))

    if flows:
        flow = np.stack(flows)  # (n, H, W, 2) float32, đơn vị pixel
        clip_ratio = float((np.abs(flow) > BOUND).mean())
        q = np.round((np.clip(flow, -BOUND, BOUND) + BOUND) / (2 * BOUND) * 255).astype(np.uint8)
        diff = np.stack(diffs)  # (n, H, W) uint8
    else:  # video gần như toàn frame trống -> chuyển động bằng 0
        clip_ratio = 0.0
        q = np.full((1, size, size, 2), 128, np.uint8)
        diff = np.zeros((1, size, size), np.uint8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, flow=q, diff=diff)
    return dict(
        video=video_path, npz=out_path, n_frames=len(frames),
        n_blank=len(frames) - len(kept), n_motion=len(q),
        clip_ratio=clip_ratio, sec=time.time() - t0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    paths = [p if os.path.isabs(p) else os.path.join(args.data_root, p) for p in df["path"]]
    jobs = [(p, out_path_for(p, args.data_root, args.out_root), args.size) for p in paths]

    rows = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(process, jobs, chunksize=4)):
            if r:
                rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} video")

    os.makedirs(args.out_root, exist_ok=True)
    if rows:
        st = pd.DataFrame(rows)
        st.to_csv(os.path.join(args.out_root, "motion_stats.csv"), index=False)
        print(f"Xong {len(st)} video | {st.sec.mean() * 1000:.0f} ms/video (1 CPU core)")
        print(f"Video có frame trống đã bị bỏ: {(st.n_blank > 0).sum()}")
        bad = st[st.n_frames == 0]
        if len(bad):
            print(f"CẢNH BÁO: {len(bad)} video không đọc được frame nào (đã lưu chuyển động = 0), "
                  f"xem motion_stats.csv, ví dụ: {bad.video.iloc[0]}")
        print(f"Tỷ lệ flow bị cắt ở ±{BOUND:.0f}px: trung bình {st.clip_ratio.mean():.5f}, "
              f"p99 {st.clip_ratio.quantile(0.99):.5f} (nếu p99 > ~0.01 thì tăng BOUND)")


if __name__ == "__main__":
    main()
