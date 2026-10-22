"""
Bước 2 – Train mô hình motion-based (TensorFlow/Keras).

Kiến trúc (kiểu TSN / two-stream, nhánh motion):
  clip chuyển động (T=16, 224, 224, 3)
    -> TimeDistributed(CNN 2D pretrained ImageNet)   : đặc trưng không gian từng bước
    -> mô-đun thời gian: avg | tconv | gru            : gộp thông tin theo thời gian
    -> Dense(100) logits

Đầu vào chuyển động, chọn bằng --mode:
  diff : |frame_t - frame_{t-1}| lặp thành 3 kênh                (baseline, rẻ nhất)
  flow : optical flow mã hóa 3 kênh (u, v, độ lớn) trong [-1, 1]  (để dùng thẳng trọng số ImageNet)

CSV cần các cột: path, label (0–99; test để -1), split (train | val | public_test | private_test).
Dùng đúng file chia tập theo nhóm của cả nhóm để so sánh công bằng với RGB/landmark.

Ví dụ:
  # M0 – baseline frame difference
  python train_motion.py --csv splits.csv --data_root /data/vsl --motion_root /data/vsl_motion \
      --mode diff --temporal avg --out runs/m0_diff_avg
  # M1 – optical flow + trung bình thời gian
  python train_motion.py ... --mode flow --temporal avg   --out runs/m1_flow_avg
  # M2 – optical flow + temporal conv
  python train_motion.py ... --mode flow --temporal tconv --out runs/m2_flow_tconv
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

NUM_CLASSES = 100
DIFF_GAIN = 4.0  # frame difference thường rất nhỏ -> khuếch đại trước khi đưa vào CNN


# ----------------------------------------------------------------------------- dữ liệu
def npz_path(p, data_root, motion_root):
    p = p if os.path.isabs(p) else os.path.join(data_root, p)
    rel = os.path.relpath(p, data_root)
    return os.path.join(motion_root, os.path.splitext(rel)[0] + ".npz")


def load_clip(path, mode, T, S, train, rng):
    d = np.load(path)
    raw = d["flow"] if mode == "flow" else d["diff"]
    n = raw.shape[0]

    # Lấy đều T bước; train thì co giãn tốc độ 0.9–1.1 (giống mục 4.3c). n < T thì tự lặp bước.
    s = rng.uniform(0.9, 1.1) if train else 1.0
    idx = np.clip(np.round(np.linspace(0, n - 1, T) * s), 0, n - 1).astype(np.int64)
    raw = raw[idx]

    if mode == "flow":
        x = raw.astype(np.float32) / 255.0 * 2.0 - 1.0  # (T,H,W,2) ≈ flow / BOUND
    else:
        x = (np.minimum(raw.astype(np.float32) * DIFF_GAIN, 255.0) / 255.0)[..., None]  # (T,H,W,1)

    # Random crop giữ 85–100% diện tích, cùng vị trí cho mọi bước thời gian.
    H, W = x.shape[1:3]
    if train:
        c = int(round(min(H, W) * np.sqrt(rng.uniform(0.85, 1.0))))
        y0, x0 = int(rng.integers(0, H - c + 1)), int(rng.integers(0, W - c + 1))
        x = x[:, y0:y0 + c, x0:x0 + c]

    ch, cw = x.shape[1:3]
    if (ch, cw) != (S, S):
        x = np.stack([cv2.resize(f, (S, S), interpolation=cv2.INTER_LINEAR) for f in x])
        if x.ndim == 3:
            x = x[..., None]
        if mode == "flow":
            # phóng to ảnh thì vector dịch chuyển cũng phải phóng to theo
            x[..., 0] *= S / cw
            x[..., 1] *= S / ch

    if mode == "flow":
        x = np.clip(x, -1.0, 1.0)
        mag = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2) / np.sqrt(2.0)  # [0, 1]
        x = np.concatenate([x, mag[..., None] * 2.0 - 1.0], axis=-1)
    else:
        x = np.repeat(x * 2.0 - 1.0, 3, axis=-1)
    return x.astype(np.float32)  # (T, S, S, 3) trong [-1, 1]


class BalancedIndices:
    """Tương đương WeightedRandomSampler: trọng số 1/n_c, có hoàn lại, N mẫu mỗi epoch."""

    def __init__(self, labels, seed):
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        w = 1.0 / counts[labels]
        self.p, self.n, self.seed, self.epoch = w / w.sum(), len(labels), seed, 0

    def __call__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.choice(self.n, size=self.n, replace=True, p=self.p)
        seeds = rng.integers(0, 2 ** 31 - 1, size=self.n)
        for i, s in zip(idx, seeds):
            yield int(i), int(s)


def make_ds(paths, labels, args, train, sampler=None):
    def _load(i, s):
        x = load_clip(paths[int(i)], args.mode, args.T, args.size, train, np.random.default_rng(int(s)))
        return x, np.int64(labels[int(i)])

    def _map(i, s):
        x, y = tf.numpy_function(_load, [i, s], [tf.float32, tf.int64])
        x.set_shape((args.T, args.size, args.size, 3))
        y.set_shape(())
        return x, tf.one_hot(y, NUM_CLASSES)

    spec = (tf.TensorSpec((), tf.int64), tf.TensorSpec((), tf.int64))
    if train:
        ds = tf.data.Dataset.from_generator(sampler, output_signature=spec)
    else:
        n = len(paths)
        ds = tf.data.Dataset.from_tensor_slices((np.arange(n, dtype=np.int64), np.zeros(n, np.int64)))
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not train)
    return ds.batch(args.bs, drop_remainder=train).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------- mô hình
def build_model(args):
    weights = None if args.no_pretrained else "imagenet"
    inp = keras.Input((args.T, args.size, args.size, 3))
    if args.backbone == "mobilenetv2":
        bb = keras.applications.MobileNetV2(include_top=False, weights=weights,
                                            input_shape=(args.size, args.size, 3), pooling="avg")
        x = inp  # MobileNetV2 nhận đầu vào [-1, 1]
    else:
        bb = keras.applications.EfficientNetB0(include_top=False, weights=weights,
                                               input_shape=(args.size, args.size, 3), pooling="avg")
        x = layers.Rescaling(127.5, offset=127.5)(inp)  # EfficientNet của Keras nhận [0, 255]

    f = layers.TimeDistributed(bb, name="frame_encoder")(x)  # (B, T, D)
    f = layers.Dropout(0.3)(f)
    if args.temporal == "avg":
        h = layers.GlobalAveragePooling1D()(f)
    elif args.temporal == "gru":
        h = layers.Bidirectional(layers.GRU(256))(f)
    else:  # tconv: 2 lớp Conv1D theo trục thời gian
        h = layers.Conv1D(256, 3, padding="same", activation="relu")(f)
        h = layers.Dropout(0.3)(h)
        h = layers.Conv1D(256, 3, padding="same", activation="relu")(h)
        h = layers.GlobalAveragePooling1D()(h)
    h = layers.Dropout(0.4)(h)
    out = layers.Dense(NUM_CLASSES, dtype="float32", name="logits")(h)
    return keras.Model(inp, out), bb


def set_backbone_trainable(bb, trainable):
    bb.trainable = trainable
    if trainable:  # giữ BatchNorm đóng băng: batch nhỏ, các frame trong clip tương quan mạnh
        for l in bb.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = False


class MacroF1Checkpoint(keras.callbacks.Callback):
    """Chọn mô hình theo Macro-F1 trên validation (chỉ số của cuộc thi) + early stopping."""

    def __init__(self, val_ds, y_val, out_path, patience):
        super().__init__()
        self.val_ds, self.y_val, self.out_path, self.patience = val_ds, y_val, out_path, patience
        self.best, self.wait = -1.0, 0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.val_ds, verbose=0).argmax(1)
        f1 = f1_score(self.y_val, pred, average="macro", labels=np.arange(NUM_CLASSES), zero_division=0)
        acc = accuracy_score(self.y_val, pred)
        if logs is not None:
            logs["val_macro_f1"], logs["val_top1"] = f1, acc
        print(f"  epoch {epoch + 1}: val macro-F1 {f1:.4f} | val top-1 {acc:.4f}")
        if f1 > self.best:
            self.best, self.wait = f1, 0
            self.model.save_weights(self.out_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--motion_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["flow", "diff"], default="flow")
    ap.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
    ap.add_argument("--temporal", choices=["avg", "tconv", "gru"], default="avg")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--warmup_epochs", type=int, default=3, help="giai đoạn 1: đóng băng backbone")
    ap.add_argument("--epochs", type=int, default=30, help="giai đoạn 2: fine-tune toàn bộ")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true", help="chỉ dùng để chạy thử")
    ap.add_argument("--eval_only", action="store_true",
                    help="bỏ qua train, nạp <out>/best.weights.h5 rồi đánh giá lại + đo thời gian")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    keras.utils.set_random_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU: {len(gpus)}" + ("" if gpus else "  (đang chạy trên CPU, sẽ rất chậm)"))
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    if args.mixed_precision and gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")

    global NUM_CLASSES
    classes_json = os.path.splitext(args.csv)[0] + "_classes.json"
    class_names = None
    if os.path.exists(classes_json):  # sinh bởi make_splits.py từ label_mapping.pkl
        with open(classes_json, encoding="utf-8") as fh:
            class_names = json.load(fh)
        NUM_CLASSES = len(class_names)
    print(f"Số lớp: {NUM_CLASSES}" + (f" (theo {classes_json})" if class_names else ""))

    df = pd.read_csv(args.csv)
    df["npz"] = [npz_path(p, args.data_root, args.motion_root) for p in df["path"]]
    missing = df[~df["npz"].map(os.path.exists)]
    if len(missing):
        raise SystemExit(f"{len(missing)} video chưa có file .npz, hãy chạy precompute_motion.py trước")
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    y_tr, y_va = tr.label.values.astype(np.int64), va.label.values.astype(np.int64)
    print(f"train {len(tr)} | val {len(va)}")

    train_ds = make_ds(tr.npz.values, y_tr, args, True, BalancedIndices(y_tr, args.seed))
    val_ds = make_ds(va.npz.values, y_va, args, False)

    model, bb = build_model(args)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    best_path = os.path.join(args.out, "best.weights.h5")
    ckpt = MacroF1Checkpoint(val_ds, y_va, best_path, args.patience)
    logger = keras.callbacks.CSVLogger(os.path.join(args.out, "history.csv"), append=True)

    if args.eval_only:
        if not os.path.exists(best_path):
            raise SystemExit(f"Không thấy {best_path}")
        print("eval_only: bỏ qua train")
    else:
        train(model, bb, train_ds, loss, ckpt, logger, args, len(tr))

    evaluate_all(model, df, va, y_va, best_path, class_names, args)


def train(model, bb, train_ds, loss, ckpt, logger, args, n_train):
    # Giai đoạn 1: chỉ train phần thời gian + classifier
    set_backbone_trainable(bb, False)
    model.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
                  loss=loss, metrics=["accuracy"])
    model.fit(train_ds, epochs=args.warmup_epochs, callbacks=[ckpt, logger], verbose=2)

    # Giai đoạn 2: mở backbone, learning rate nhỏ, cosine decay
    set_backbone_trainable(bb, True)
    steps = max(1, n_train // args.bs)
    sched = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps * args.epochs)
    model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-4),
                  loss=loss, metrics=["accuracy"])
    ckpt.wait = 0
    model.fit(train_ds, initial_epoch=args.warmup_epochs,
              epochs=args.warmup_epochs + args.epochs, callbacks=[ckpt, logger], verbose=2)


def evaluate_all(model, df, va, y_va, best_path, class_names, args):
    # ---------------- đánh giá mô hình tốt nhất
    model.load_weights(best_path)

    def predict(sub, name):
        labels = sub.label.values.astype(np.int64) if "label" in sub else np.full(len(sub), -1)
        probs = softmax(model.predict(make_ds(sub.npz.values, labels, args, False), verbose=0))
        np.save(os.path.join(args.out, f"{name}_probs.npy"), probs)  # dùng cho late fusion
        sub[["path"]].assign(pred=probs.argmax(1)).to_csv(
            os.path.join(args.out, f"{name}_pred.csv"), index=False)
        return probs

    labs = np.arange(NUM_CLASSES)
    names = class_names
    if names is None and "class_name" in df.columns:
        m = df.dropna(subset=["class_name"]).groupby("label")["class_name"].first()
        names = [str(m.get(i, i)) for i in labs]

    def evaluate(sub, name):
        y = sub.label.values.astype(np.int64)
        pred = predict(sub, name).argmax(1)
        res = dict(
            top1=accuracy_score(y, pred),
            macro_f1=f1_score(y, pred, average="macro", labels=labs, zero_division=0),
            macro_precision=precision_score(y, pred, average="macro", labels=labs, zero_division=0),
            macro_recall=recall_score(y, pred, average="macro", labels=labs, zero_division=0),
        )
        with open(os.path.join(args.out, f"{name}_report.txt"), "w", encoding="utf-8") as fh:
            fh.write(classification_report(y, pred, labels=labs, target_names=names, zero_division=0))
        cm = confusion_matrix(y, pred, labels=labs)
        np.save(os.path.join(args.out, f"{name}_confusion.npy"), cm)
        off = cm.copy()
        np.fill_diagonal(off, 0)
        top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:20], off.shape))[0]
        pd.DataFrame([dict(true=names[t] if names else t, pred=names[p] if names else p, count=off[t, p])
                      for t, p in top if off[t, p] > 0]).to_csv(
            os.path.join(args.out, f"{name}_top_confusions.csv"), index=False)
        return res

    metrics = {"val": evaluate(va, "val")}
    for split in ["test", "public_test", "private_test"]:
        sub = df[df.split == split].reset_index(drop=True)
        if not len(sub):
            continue
        if (sub.label >= 0).all():
            metrics[split] = evaluate(sub, split)  # test có nhãn -> chấm điểm luôn
        else:
            predict(sub, split)

    # Thời gian suy luận của riêng mô hình cho 1 clip (chưa tính thời gian tính flow).
    # Dùng tf.function (đồ thị đã biên dịch) + .numpy() để chờ GPU chạy xong; gọi model() ở
    # chế độ eager như bản cũ đo cả overhead Python nên ra số lớn bất thường.
    infer = tf.function(lambda x: model(x, training=False))
    dummy = tf.zeros((1, args.T, args.size, args.size, 3))
    for _ in range(5):
        infer(dummy).numpy()
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        infer(dummy).numpy()
        times.append((time.perf_counter() - t0) * 1000)
    metrics["model_ms_per_clip"] = float(np.median(times))
    metrics["device"] = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    metrics["config"] = vars(args)

    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
