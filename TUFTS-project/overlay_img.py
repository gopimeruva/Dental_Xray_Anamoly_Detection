import cv2
import numpy as np
from pathlib import Path

# Load paths
image_path = Path("dataset/images/95.JPG")
mask_mm = Path("Segmentation/maxillomandibular/95.jpg")
mask_teeth = Path("Segmentation/teeth_mask/95.jpg")
mask_anomaly = Path("dataset/mask/95.JPG")
mask_gaze = Path("Expert/gaze_map/quantized/95.JPG")

# Load images
image = cv2.imread(str(image_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mm_mask = cv2.imread(str(mask_mm), cv2.IMREAD_GRAYSCALE)
teeth_mask = cv2.imread(str(mask_teeth), cv2.IMREAD_GRAYSCALE)
anomaly_mask = cv2.imread(str(mask_anomaly), cv2.IMREAD_GRAYSCALE)
gaze_heatmap = cv2.imread(str(mask_gaze))  # already in color

# Overlay function with dark tones and weighted transparency
def apply_overlay(base_img, mask, color_bgr, alpha=0.35):
    overlay = base_img.copy()
    color_layer = np.full_like(base_img, color_bgr, dtype=np.uint8)
    mask_bool = mask > 0
    mask_3ch = np.stack([mask_bool] * 3, axis=-1)
    overlay = np.where(mask_3ch, cv2.addWeighted(base_img, 1 - alpha, color_layer, alpha, 0), base_img)
    return overlay

# Darker muted colors (in BGR)
colors = {
    'mm': (128, 64, 0),        # Dark orange-brown
    'teeth': (128, 0, 128),    # Deep purple
    'anomaly': (0, 255, 255)    # Dark teal
}

# Generate overlays
overlay_mm = apply_overlay(image_rgb, mm_mask, colors['mm'])
overlay_teeth = apply_overlay(image_rgb, teeth_mask, colors['teeth'])
overlay_anomaly = apply_overlay(image_rgb, anomaly_mask, colors['anomaly'])
overlay_gaze = cv2.addWeighted(image_rgb, 0.7, gaze_heatmap, 0.3, 0)  # Blend with heatmap

# Convert to BGR for saving
cv2.imwrite("overlay_maxillomandibular.jpg", cv2.cvtColor(overlay_mm, cv2.COLOR_RGB2BGR))
cv2.imwrite("overlay_teeth.jpg", cv2.cvtColor(overlay_teeth, cv2.COLOR_RGB2BGR))
cv2.imwrite("overlay_anomalies.jpg", cv2.cvtColor(overlay_anomaly, cv2.COLOR_RGB2BGR))
# cv2.imwrite("overlay_gaze.jpg", cv2.cvtColor(overlay_gaze, cv2.COLOR_RGB2BGR))

print("✅ Muted overlay images saved.")
