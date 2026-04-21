import cv2
import os
import numpy as np
from pathlib import Path

real_dir = "/mnt/d/arpit/anti-spoof-face-verification/data/real"
spoof_dir = "/mnt/d/arpit/anti-spoof-face-verification/data/spoof"

os.makedirs(spoof_dir, exist_ok=True)

files = os.listdir(real_dir)

count = 0

for file in files:
    path = os.path.join(real_dir, file)
    img = cv2.imread(path)

    if img is None:
        continue

    h, w = img.shape[:2]

    # 1 Blur
    blur = cv2.GaussianBlur(img, (15,15), 0)

    # 2 Bright screen glare
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=40)

    # 3 Pixelated screen effect
    small = cv2.resize(img, (w//4, h//4))
    pixel = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 4 Rotated photo
    M = cv2.getRotationMatrix2D((w//2,h//2), 8, 1)
    rotate = cv2.warpAffine(img, M, (w,h))

    # 5 Low quality jpeg
    temp = "temp.jpg"
    cv2.imwrite(temp, img, [cv2.IMWRITE_JPEG_QUALITY, 20])
    lowq = cv2.imread(temp)

    variants = [blur, bright, pixel, rotate, lowq]

    for v in variants:
        out = os.path.join(spoof_dir, f"spoof_{count}.jpg")
        cv2.imwrite(out, v)
        count += 1

print("Spoof images created:", count)