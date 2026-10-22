"""
Tạo splits.csv từ cấu trúc thư mục đã chia sẵn, dùng ĐÚNG label_mapping.pkl của nhóm:
    <data_root>/label_mapping.pkl
    <data_root>/train/<tên lớp>/*.mp4
    <data_root>/valid/<tên lớp>/*.mp4
    <data_root>/test/<tên lớp>/*.mp4     (hoặc test/*.mp4 nếu test không có nhãn)

Cột đầu ra: path (tương đối so với data_root), label, class_name, split (train | val | test)

- KHÔNG tự tạo mapping: bắt buộc có label_mapping.pkl (mặc định <data_root>/label_mapping.pkl).
- So khớp tên thư mục lớp với mapping sau khi chuẩn hóa Unicode NFC + bỏ khoảng trắng hai đầu
  (tránh lệch NFC/NFD giữa Windows/macOS/Linux). Lớp nào không có trong mapping -> dừng, không đoán.
- Kiểm tra rò rỉ: báo các file giống hệt từng byte nằm ở hai tập khác nhau.
"""
import argparse
import hashlib
import json
import os
import pickle
import unicodedata

import numpy as np
import pandas as pd

SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
EXTS = (".mp4", ".avi", ".mov", ".mkv")


def norm(s):
    return unicodedata.normalize("NFC", str(s)).strip()


def is_int_like(x):
    return isinstance(x, (int, np.integer)) or (isinstance(x, str) and x.strip().lstrip("-").isdigit())


def load_mapping(path):
    """Trả về dict {tên lớp đã chuẩn hóa: id}. Chấp nhận các dạng thường gặp:
    {tên: id}, {id: tên}, list/tuple/array tên (vị trí = id), LabelEncoder (classes_),
    hoặc dict bọc ngoài có khóa như label2id / class_to_idx / id2label."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)  # chỉ dùng với file do chính nhóm tạo

    if hasattr(obj, "classes_"):  # sklearn LabelEncoder
        obj = list(obj.classes_)
    if isinstance(obj, dict):
        for k in ("label2id", "class_to_idx", "label_to_idx", "name2id", "id2label", "idx_to_class"):
            if k in obj and isinstance(obj[k], (dict, list, tuple)):
                obj = obj[k]
                break
    if isinstance(obj, (list, tuple, np.ndarray)):
        obj = {i: name for i, name in enumerate(obj)}
    if not isinstance(obj, dict):
        raise SystemExit(f"Không hiểu định dạng label_mapping: {type(obj)}")

    keys_int = all(is_int_like(k) for k in obj)
    vals_int = all(is_int_like(v) for v in obj.values())
    if keys_int and not vals_int:          # {id: tên}
        m = {norm(v): int(k) for k, v in obj.items()}
    elif vals_int and not keys_int:        # {tên: id}
        m = {norm(k): int(v) for k, v in obj.items()}
    else:
        raise SystemExit("label_mapping phải là {tên: id} hoặc {id: tên}; "
                         f"ví dụ phần tử đầu: {list(obj.items())[:3]}")

    if len(m) != len(obj):
        raise SystemExit("label_mapping có tên lớp trùng nhau sau khi chuẩn hóa Unicode")
    ids = sorted(m.values())
    if ids != list(range(len(ids))):
        raise SystemExit(f"ID trong label_mapping không liên tục 0..{len(ids) - 1}: {ids[:5]}...{ids[-5:]}")
    return m


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="splits.csv")
    ap.add_argument("--mapping", default=None, help="mặc định: <data_root>/label_mapping.pkl")
    ap.add_argument("--skip_leak_check", action="store_true")
    args = ap.parse_args()

    mapping_path = args.mapping or os.path.join(args.data_root, "label_mapping.pkl")
    if not os.path.exists(mapping_path):
        raise SystemExit(f"Không thấy {mapping_path}. Script không tự tạo mapping, hãy chỉ đúng --mapping.")
    mapping = load_mapping(mapping_path)
    print(f"Mapping: {mapping_path} | {len(mapping)} lớp | ví dụ: "
          + ", ".join(f"{i}={n}" for n, i in sorted(mapping.items(), key=lambda t: t[1])[:3]))

    rows = []
    for d in sorted(os.listdir(args.data_root)):
        split = SPLITS.get(d.lower())
        if split is None or not os.path.isdir(os.path.join(args.data_root, d)):
            continue
        for dirpath, _, files in os.walk(os.path.join(args.data_root, d)):
            for f in sorted(files):
                if not f.lower().endswith(EXTS):
                    continue
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, args.data_root)
                parts = rel.split(os.sep)
                cls = norm(parts[1]) if len(parts) >= 3 else None
                rows.append(dict(path=rel, class_name=cls, split=split, full=full))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Không tìm thấy video nào, kiểm tra lại --data_root")

    folders = set(df.class_name.dropna())
    unknown = sorted(folders - set(mapping))
    if unknown:
        raise SystemExit(f"{len(unknown)} thư mục lớp không có trong label_mapping (không tự gán): {unknown[:10]}")
    unused = sorted(set(mapping) - set(df[df.split == "train"].class_name.dropna()))
    if unused:
        print(f"CẢNH BÁO: {len(unused)} lớp trong mapping không có video train: {unused[:10]}")

    df["label"] = [mapping[c] if c is not None else -1 for c in df.class_name]
    print(df.groupby("split").agg(videos=("path", "size"), classes=("class_name", "nunique")))

    if not args.skip_leak_check:
        df["md5"] = [md5(p) for p in df.full]
        cross = df.groupby("md5").split.nunique()
        leak = df[df.md5.isin(cross[cross > 1].index)]
        if len(leak):
            print(f"CẢNH BÁO rò rỉ: {leak.md5.nunique()} nội dung giống hệt xuất hiện ở nhiều tập "
                  f"({len(leak)} file). Ví dụ:")
            print(leak.sort_values("md5")[["split", "path"]].head(10).to_string(index=False))
        else:
            print("Không có file trùng byte giữa các tập.")

    df[["path", "label", "class_name", "split"]].to_csv(args.out, index=False)
    # Lưu kèm số lớp + thứ tự tên lớp để train_motion.py dùng đúng theo mapping
    meta = os.path.splitext(args.out)[0] + "_classes.json"
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump([n for n, _ in sorted(mapping.items(), key=lambda t: t[1])], fh, ensure_ascii=False, indent=0)
    print(f"Đã ghi {args.out} và {meta}")


if __name__ == "__main__":
    main()
