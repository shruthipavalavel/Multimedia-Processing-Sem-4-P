import cv2
import numpy as np
from matplotlib import pyplot as plt

# ---------- Helper functions ----------

def compute_bit_planes(gray_img):
    """
    Compute 8 bit-planes for an 8-bit grayscale image.
    Returns a list of 8 images (bit-plane 0 to 7).
    """
    bit_planes = []
    for bit in range(8):
        # Extract bit 'bit' using bitwise shift and AND
        plane = (gray_img >> bit) & 1
        # Scale to 0–255 for display
        plane = plane * 255
        bit_planes.append(plane.astype(np.uint8))
    return bit_planes


def reconstruct_from_lowest_three_bits(gray_img):
    """
    Reconstruct image using union (combination) of the 3 lowest bit-planes:
    bit 0, bit 1, bit 2.
    
    This is equivalent to keeping only those bits from the original image.
    """
    reconstructed = np.zeros_like(gray_img, dtype=np.uint8)
    for bit in range(3):  # bits 0, 1, 2
        plane = (gray_img >> bit) & 1  # get this bit (0 or 1)
        reconstructed |= (plane.astype(np.uint8) << bit)
    return reconstructed


def process_image(image_path, title="Image"):
    # Read image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image at: {image_path}")
        return

    # 1. Compute bit-planes
    bit_planes = compute_bit_planes(img)

    # 2. Reconstruct using union of the three lowest bit-planes
    low_bits_recon = reconstruct_from_lowest_three_bits(img)

    # 3. Difference between union image and original image
    # (original - union) in absolute value to stay in valid range
    diff_img = cv2.absdiff(img, low_bits_recon)

    # ---- Display results (optional, for understanding) ----
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title(f"{title} - Original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title(f"{title} - 3 Lowest Bits Reconstructed")
    plt.imshow(low_bits_recon, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title(f"{title} - Difference (Original - Union)")
    plt.imshow(diff_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Optionally save outputs
    cv2.imwrite(f"{title.lower().replace(' ', '_')}_lowbits.png", low_bits_recon)
    cv2.imwrite(f"{title.lower().replace(' ', '_')}_difference.png", diff_img)

    # Return in case you want to use them later
    return {
        "original": img,
        "bit_planes": bit_planes,
        "low_bits_reconstructed": low_bits_recon,
        "difference": diff_img
    }


# ---------- MAIN PART: change these paths to your images ----------

if __name__ == "__main__":
    # Give your own image paths here
    low_light_path = "C:\\Users\\shrut_c82091v\\OneDrive\\Desktop\\New folder (3)\\bright.png"      # e.g. "images/myphoto_low.png"
    bright_light_path = "C:\\Users\\shrut_c82091v\\OneDrive\\Desktop\\New folder (3)\\bright.png" # e.g. "images/myphoto_bright.png"

    print("Processing low-light image...")
    low_results = process_image(low_light_path, title="Low Light")

    print("Processing bright-light image...")
    bright_results = process_image(bright_light_path, title="Bright Light")
