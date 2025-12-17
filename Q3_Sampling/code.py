from PIL import Image
import numpy as np

# -----------------------------
# Load Image
# -----------------------------
path = "image.jpg"

img_gray = Image.open(path).convert("L")     # For frequency sampling (grayscale)
img_rgb  = Image.open(path).convert("RGB")   # For spatial sampling (color)

gray = np.array(img_gray)
rgb  = np.array(img_rgb)

print("\nGrayscale Resolution :", gray.shape)   # (H, W)
print("RGB Resolution        :", rgb.shape)     # (H, W, 3)


# -----------------------------
# FREQUENCY SAMPLING (FFT)
# -----------------------------
def freq_sample(im, f):
    """
    im : grayscale numpy array
    f  : sampling factor (2,4,8,16)
    keeps center (w/f, h/f) of the FFT spectrum.
    """
    F = np.fft.fftshift(np.fft.fft2(im))        # 2D FFT + center low frequency
    H, W = F.shape

    # crop dimensions
    h2 = max(1, H // f)
    w2 = max(1, W // f)
    F_low = np.zeros_like(F)

    # center region coordinates
    hs = (H - h2) // 2
    he = hs + h2
    ws = (W - w2) // 2
    we = ws + w2

    # keep low-frequency region
    F_low[hs:he, ws:we] = F[hs:he, ws:we]

    # inverse FFT → blurred image
    out = np.abs(np.fft.ifft2(np.fft.ifftshift(F_low)))

    # normalize to 0-255
    mn, mx = out.min(), out.max()
    if mx - mn < 1e-9:
        out_u8 = out.astype(np.uint8)
    else:
        out_u8 = ((out - mn) / (mx - mn) * 255).astype(np.uint8)

    return out_u8


# -----------------------------
# SPATIAL SAMPLING (RGB)
# -----------------------------
def spatial_sample(im, f):
    """
    im : RGB array
    f  : sampling factor
    Returns every f-th row and column
    """
    return im[::f, ::f].copy()


# -----------------------------
# RUN for each factor
# -----------------------------
factors = [2, 4, 8, 16]

for f in factors:
    # frequency (grayscale)
    out_f = freq_sample(gray, f)
    fname_f = f"freq_1_{f}.png"
    Image.fromarray(out_f).save(fname_f)

    print(f"\nFrequency Sampling 1/{f}")
    print("→ Output Resolution:", out_f.shape)
    print("Saved:", fname_f)

    # spatial (RGB)
    out_s = spatial_sample(rgb, f)
    fname_s = f"spatial_1_{f}.png"
    Image.fromarray(out_s).save(fname_s)

    print(f"Spatial Sampling 1/{f}")
    print("→ Output Resolution:", out_s.shape)
    print("Saved:", fname_s)

print("\nAll images saved in the current directory.")