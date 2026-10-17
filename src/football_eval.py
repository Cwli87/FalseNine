from pathlib import Path
import cv2
import numpy as np


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def load_gt(lbl, W, H, class_ids):
    boxes = []
    for line in Path(lbl).read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and int(p[0]) in class_ids:
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            boxes.append([(cx - bw / 2) * W, (cy - bh / 2) * H,
                          (cx + bw / 2) * W, (cy + bh / 2) * H])
    return boxes


def evaluate(predict, dataset_dir, splits, pred_ball, pred_person,
             gt_ball=(0,), gt_person=(1, 2),
             ball_conf=0.10, person_conf=0.30, iou_thr=0.3):
    """predict(img) must return (class_ids, confidences, xyxy_boxes) as numpy arrays."""
    r = dict(ball_found=0, ball_total=0, ball_extra=0,
             person_found=0, person_total=0, person_extra=0)
    for split in splits:
        for img_path in sorted((Path(dataset_dir) / split / "images").glob("*")):
            lbl = Path(dataset_dir) / split / "labels" / (img_path.stem + ".txt")
            if not lbl.exists():
                continue
            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]
            cls, conf, xyxy = predict(img)
            bp = xyxy[np.isin(cls, list(pred_ball)) & (conf >= ball_conf)]
            pp = xyxy[np.isin(cls, list(pred_person)) & (conf >= person_conf)]
            gb = load_gt(lbl, W, H, gt_ball)
            gp = load_gt(lbl, W, H, gt_person)
            r["ball_total"] += len(gb)
            r["person_total"] += len(gp)
            r["ball_found"] += sum(any(iou(g, p) >= iou_thr for p in bp) for g in gb)
            r["person_found"] += sum(any(iou(g, p) >= iou_thr for p in pp) for g in gp)
            r["ball_extra"] += sum(not any(iou(g, p) >= iou_thr for g in gb) for p in bp)
            r["person_extra"] += sum(not any(iou(g, p) >= iou_thr for g in gp) for p in pp)
    return r