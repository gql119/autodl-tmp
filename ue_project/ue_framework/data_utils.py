import os
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np



def list_images(img_dir: str) -> List[str]:
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    names = [n for n in os.listdir(img_dir) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    names.sort()
    return [os.path.join(img_dir, n) for n in names]



def stem_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]



def label_path_for_image(image_path: str, label_dir: str) -> str:
    return os.path.join(label_dir, stem_of(image_path) + ".txt")



def read_yolo_annotations(label_path: str) -> List[Dict]:
    anns: List[Dict] = []
    if not os.path.isfile(label_path):
        return anns
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            anns.append({"cls": cls_id, "bbox": bbox})
    return anns



def image_has_target(annotations: List[Dict], target_class_id: int) -> bool:
    return any(int(a["cls"]) == int(target_class_id) for a in annotations)



def load_image_rgb_float(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0



def save_image_rgb_float(path: str, img_float: np.ndarray, jpg_quality: int = 100) -> None:
    img_uint8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    else:
        cv2.imwrite(path, img_bgr)



def copy_label(src_label: str, dst_label: str) -> None:
    os.makedirs(os.path.dirname(dst_label), exist_ok=True)
    if os.path.isfile(src_label):
        shutil.copy2(src_label, dst_label)
    else:
        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("")



def split_val_image_lists(
    val_img_dir: str,
    val_label_dir: str,
    target_class_id: int,
) -> Tuple[List[str], List[str]]:
    person_free: List[str] = []
    person_cooccur: List[str] = []
    for img_path in list_images(val_img_dir):
        label_path = label_path_for_image(img_path, val_label_dir)
        anns = read_yolo_annotations(label_path)
        if image_has_target(anns, target_class_id):
            person_cooccur.append(img_path)
        else:
            person_free.append(img_path)
    return person_free, person_cooccur



def write_image_list_txt(path: str, image_paths: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in image_paths:
            f.write(p + "\n")



def parse_manifest_flags(rows: List[Dict], field: str, true_value: str = "1") -> List[Dict]:
    return [r for r in rows if str(r.get(field, "")) == true_value]

