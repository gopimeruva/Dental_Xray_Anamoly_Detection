import os
import cv2
import numpy as np
from pathlib import Path

# Paths to images and masks
images_dir = Path("./TUFTS-project/Radiographs")
masks_dir = Path("./TUFTS-project/Expert/mask")
labels_dir = Path("./TUFTS-project/Expert/labels")
labels_dir.mkdir(parents=True, exist_ok=True)

# Class ID for segmentation
class_id = 0

# Iterate over images and generate labels
for image_path in images_dir.glob("*.jpg"):
    mask_path = masks_dir / image_path.name  # Match filename
    label_path = labels_dir / image_path.with_suffix(".txt").name
    print(label_path)
    # Read the mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Write YOLO format labels
    with open(label_path, "w") as f:
        for contour in contours:
            contour = contour.squeeze()
            if len(contour) < 3:  # Skip invalid polygons
                continue
            # Normalize points
            h, w = mask.shape
            points = contour / [w, h]
            points = points.flatten()
            points_str = " ".join(map(str, points))
            f.write(f"{class_id} {points_str}\n")
