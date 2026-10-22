"""
Demo M0 – frame difference + MobileNetV2 + trung bình thời gian.

Chạy:
  streamlit run demo_m0.py

Chọn một video trong tập test (hoặc tải video lên) -> xem 16 bước frame difference mà mô hình
thực sự nhận vào, top-5 dự đoán và thời gian xử lý. Tiền xử lý dùng lại đúng hàm của
precompute_motion.py và train_motion.py để kết quả khớp với lúc đánh giá.
"""
import io
import json
import os
import tempfile
import time
import types

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "data")
WORK = os.path.join(ROOT, "vsl_motion_work")
RUN = os.path.join(WORK, "runs", "m0_diff_avg")
SIZE = 224  # kích thước lúc tiền tính chuyển động


# ----------------------------------------------------------------------------- mô hình
@st.cache_resource
def load_model():
    import tensorflow as tf
    import train_motion as tm

    with open(os.path.join(RUN, "metrics.json"), encoding="utf-8") as fh:
        args = types.SimpleNamespace(**json.load(fh)["config"])
    args.no_pretrained = True  # trọng số lấy từ best.weights.h5, không cần tải ImageNet
    with open(os.path.join(WORK, "splits_classes.json"), encoding="utf-8") as fh:
        names = json.load(fh)
    tm.NUM_CLASSES = len(names)

    model, _ = tm.build_model(args)
    model.load_weights(os.path.join(RUN, "best.weights.h5"))
    infer = tf.function(lambda x: model(x, training=False))
    infer(tf.zeros((1, args.T, args.size, args.size, 3))).numpy()  # biên dịch trước
    return infer, args, names, tm


@st.cache_data
def load_test():
    df = pd.read_csv(os.path.join(WORK, "splits.csv"))
    test = df[df.split == "test"].reset_index(drop=True)
    pred = pd.read_csv(os.path.join(RUN, "test_pred.csv"))  # cùng thứ tự với test
    test["m0_pred"] = pred.pred.values
    return test


@st.cache_data(show_spinner=False)
def browser_video(video_bytes, ext):
    """mp4 của bộ dữ liệu dùng codec MPEG-4 Part 2 (FMP4) mà trình duyệt không phát được
    -> chuyển sang WebM VP8 bằng chính OpenCV (không cần cài ffmpeg)."""
    with tempfile.TemporaryDirectory() as d:  # đường dẫn ASCII, tránh lỗi OpenCV với tên tiếng Việt
        src, dst = os.path.join(d, "in" + ext), os.path.join(d, "out.webm")
        with open(src, "wb") as fh:
            fh.write(video_bytes)
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
        writer = None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"VP80"), fps, (f.shape[1], f.shape[0]))
            writer.write(f)
        cap.release()
        if writer is None:
            return None
        writer.release()
        with open(dst, "rb") as fh:
            return fh.read()


def motion_clip(video_path, args, tm):
    """Video -> (T, S, S, 3) giống hệt lúc đánh giá (bỏ phần optical flow cho nhanh)."""
    from precompute_motion import is_blank, read_frames

    frames = read_frames(video_path)
    kept = [f for f in frames if not is_blank(f)]
    gray = [cv2.cvtColor(cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            for f in kept]
    diffs = [cv2.absdiff(b, a) for a, b in zip(gray[:-1], gray[1:])]
    diff = np.stack(diffs) if diffs else np.zeros((1, SIZE, SIZE), np.uint8)

    buf = io.BytesIO()
    np.savez(buf, diff=diff)
    buf.seek(0)
    x = tm.load_clip(buf, "diff", args.T, args.size, False, np.random.default_rng(0))
    return x, len(frames), len(frames) - len(kept)


def predict(video_path):
    infer, args, names, tm = load_model()
    t0 = time.perf_counter()
    x, n_frames, n_blank = motion_clip(video_path, args, tm)
    t1 = time.perf_counter()
    probs = tm.softmax(infer(x[None]).numpy())[0]
    t2 = time.perf_counter()
    return dict(x=x, probs=probs, n_frames=n_frames, n_blank=n_blank,
                prep_ms=(t1 - t0) * 1000, model_ms=(t2 - t1) * 1000)


# ----------------------------------------------------------------------------- giao diện
def main():
    st.set_page_config(page_title="Demo M0 – VSL Motion", layout="wide")
    st.title("Demo M0: Frame difference")
    st.caption("Nhận dạng ngôn ngữ ký hiệu Việt Nam (100 lớp) chỉ từ chuyển động giữa các frame liên tiếp.")

    with st.spinner("Đang nạp mô hình..."):
        _, args, names, _ = load_model()

    src = st.sidebar.radio("Nguồn video", ["Tập test", "Tải video lên"])
    label, tmp = None, None
    if src == "Tập test":
        test = load_test()
        if st.sidebar.checkbox("Chỉ hiện video M0 đoán sai"):
            test = test[test.label != test.m0_pred]
        cls = st.sidebar.selectbox("Lớp", sorted(test.class_name.unique()))
        sub = test[test.class_name == cls]
        row = sub.iloc[st.sidebar.selectbox("Video", range(len(sub)),
                                            format_func=lambda i: os.path.basename(sub.path.iloc[i]))]
        path, label = os.path.join(DATA_ROOT, row.path), int(row.label)
        with open(path, "rb") as fh:
            video_bytes = fh.read()
    else:
        up = st.sidebar.file_uploader("Video ký hiệu", type=["mp4", "avi", "mov", "mkv"])
        if up is None:
            st.info("Tải lên một video ký hiệu (một người ký một từ, khoảng 2–3 giây).")
            st.stop()
        video_bytes = up.getvalue()
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(up.name)[1])
        with os.fdopen(fd, "wb") as fh:
            fh.write(video_bytes)
        path = tmp

    try:
        with st.spinner("Đang tính frame difference và dự đoán..."):
            r = predict(path)
    finally:
        if tmp:
            os.remove(tmp)

    top = np.argsort(-r["probs"])[:5]
    left, right = st.columns([1, 1])
    with left:
        webm = browser_video(video_bytes, os.path.splitext(path)[1])
        if webm:
            st.video(webm, format="video/webm", loop=True, autoplay=True, muted=True)
        else:
            st.warning("Không đọc được video để hiển thị.")
        st.caption(f"{r['n_frames']} frame · loại {r['n_blank']} frame trống")
    with right:
        if label is not None:
            if top[0] == label:
                st.success(f"Đúng: **{names[label]}**")
            else:
                st.error(f"Sai: mô hình đoán **{names[top[0]]}**, nhãn đúng là **{names[label]}**")
        st.subheader("Top-5 dự đoán")
        for k in top:
            mark = "  ✓" if k == label else ""
            st.write(f"**{names[k]}**{mark} — {r['probs'][k]:.1%}")
            st.progress(float(r["probs"][k]))
        c1, c2 = st.columns(2)
        c1.metric("Tiền xử lý", f"{r['prep_ms']:.0f} ms")
        c2.metric("Mô hình", f"{r['model_ms']:.0f} ms")

    st.subheader(f"Đầu vào mô hình: {args.T} bước frame difference")
    imgs = [np.clip((f[..., 0] + 1) / 2, 0, 1) for f in r["x"]]
    for start in range(0, len(imgs), 8):
        cols = st.columns(8)
        for i, (col, img) in enumerate(zip(cols, imgs[start:start + 8])):
            col.image(img, caption=f"bước {start + i + 1}", width="stretch")


if __name__ == "__main__":
    main()
