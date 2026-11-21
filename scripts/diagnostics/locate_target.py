#!/usr/bin/env python3
import numpy as np
from PIL import Image


def find_target(img_path):
    im = Image.open(img_path).convert("RGB")
    arr = np.array(im)
    # Target is amber/orange: R high, G medium, B low
    # e.g., R > 150, G in [70, 160], B < 70
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    mask = (r > 160) & (g > 80) & (g < 170) & (b < 70)
    y_idx, x_idx = np.where(mask)
    if len(x_idx) > 0:
        return float(np.mean(x_idx)), float(np.mean(y_idx)), len(x_idx)
    return None


for t in ["000s", "003s", "006s", "009s", "012s"]:
    res = find_target(f"outputs/reports/media/gazebo_frame_{t}.png")
    if res:
        x, y, count = res
        print(f"T={t}: Target pixel at (u={x:.1f}, v={y:.1f}) with {count} orange pixels")
    else:
        print(f"T={t}: Target not found")
