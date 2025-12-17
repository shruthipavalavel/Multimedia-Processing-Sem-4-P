import cv2
import numpy as np
import os
import sys

print("Current working directory:", os.getcwd())
print("Files here:", os.listdir())
print("-------------------------------------------------------")

# -----------------------------
#   Filenames EXACTLY as uploaded
# -----------------------------
img_files = ["img1.jpg", "img2.jpg", "img3.jpg"]

# Exposure times in seconds
exposure_times = np.array([1/30.0, 1/8.0, 1/2.0], dtype=np.float32)

# -----------------------------
#   Check existence
# -----------------------------
for f in img_files:
    print(f, "exists:", os.path.exists(f))
print("-------------------------------------------------------")

# -----------------------------
#   Load images
# -----------------------------
images = []
for f in img_files:
    img = cv2.imread(f)
    if img is None:
        print("Could not read:", f)
    else:
        images.append(img)

print("Total images read:", len(images))

if len(images) == 0:
    print("ERROR: No images loaded. Fix file names.")
    sys.exit(1)

# -----------------------------
#   Estimate CRF
# -----------------------------
calibrate = cv2.createCalibrateDebevec()
response = calibrate.process(images, exposure_times)
print("CRF estimated.")

# -----------------------------
#   Compute HDR
# -----------------------------
merge_debevec = cv2.createMergeDebevec()
hdr = merge_debevec.process(images, exposure_times, response)
print("HDR map computed.")

cv2.imwrite("output_hdr.hdr", hdr)
print("Saved output_hdr.hdr")

# -----------------------------
#   Tone mapping
# -----------------------------
tonemap = cv2.createTonemap(gamma=2.2)
ldr = tonemap.process(hdr)
ldr_8bit = np.clip(ldr * 255, 0, 255).astype("uint8")

cv2.imwrite("output_tonemapped.jpg", ldr_8bit)
print("Saved output_tonemapped.jpg")