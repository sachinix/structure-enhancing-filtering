#!/usr/bin/env python3
"""
Structure tensor slope estimation and structure-enhancing filtering
after Morelatto & Biloti (2013), "Structure enhancing filtering with the
structure tensor" (SBGf).

Arrays are indexed [t, x]  (axis 0 = time/rows, axis 1 = traces/columns).
Slopes are in samples-of-t per sample-of-x. Multiply by dt/dx to get
physical units (e.g. s/km, ms/cmp).

Usage
-----
  python structure_tensor.py demo
  python structure_tensor.py seismic gulf.npy --dt 0.004 [--zeta 2]
  python structure_tensor.py palm  palm_roi.bmp
Accepted inputs: .npy, Madagascar .rsf/.hh (float32), or any image file.
"""
import argparse
import os
import re
import numpy as np
from scipy.ndimage import (gaussian_filter, gaussian_filter1d, convolve1d,
                           map_coordinates)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def load_rsf(path):
    """Minimal Madagascar RSF reader (float32, native endian, 2-D)."""
    hdr = open(path, "r", errors="ignore").read()
    n1 = int(re.search(r"\bn1=(\d+)", hdr).group(1))
    n2 = int(re.search(r"\bn2=(\d+)", hdr).group(1))
    m = re.search(r'\bin="?([^"\s]+)"?', hdr)
    binpath = m.group(1)
    if not os.path.isabs(binpath) and not os.path.exists(binpath):
        binpath = os.path.join(os.path.dirname(path), binpath)
    raw = np.fromfile(binpath, dtype=np.float32)
    # RSF: n1 is the fastest axis (time) -> reshape (n2, n1), then to [t, x]
    return raw[: n1 * n2].reshape(n2, n1).T.astype(float)


def load_array(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(float)
    if ext in (".rsf", ".hh", ".h"):
        return load_rsf(path)
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=float)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def agc(data, window_samples=125):
    """AGC with triangular weights (paper: 0.5 s window at 4 ms = 125 samples)."""
    w = np.bartlett(window_samples)
    w /= w.sum()
    env = convolve1d(np.abs(data), w, axis=0, mode="nearest")
    return data / (env + 1e-12 * env.max() + 1e-30)


# --------------------------------------------------------------------------
# Structure tensor and slopes (eqs. 1-10)
# --------------------------------------------------------------------------
def structure_tensor(P, sigma_d=2.5, sigma_w=3.5, half=5):
    """Return <Px^2>, <Pt^2>, <PxPt>.  11x11 support (half=5) as in the paper.
    sigma_d: Gaussian for derivative estimation; sigma_w: tensor window (eq. 10).
    """
    Px = gaussian_filter(P, sigma_d, order=(0, 1), truncate=half / sigma_d)
    Pt = gaussian_filter(P, sigma_d, order=(1, 0), truncate=half / sigma_d)
    g = lambda a: gaussian_filter(a, sigma_w, truncate=half / sigma_w)
    return g(Px * Px), g(Pt * Pt), g(Px * Pt)


def eigen_and_slope(Jxx, Jtt, Jxt, smax=1e3):
    """Eigenvalues (eqs. 7-8) and slope (eq. 9). Returns slope, l1, l2, coherence."""
    disc = np.sqrt((Jxx - Jtt) ** 2 + 4 * Jxt ** 2)  # == sqrt of eq. 7 radicand
    l1 = 0.5 * (Jxx + Jtt + disc)
    l2 = 0.5 * (Jxx + Jtt - disc)                    # == det(M)/l1 (eq. 8)

    def safe(d):
        return np.where(np.abs(d) < 1e-12, 1e-12, d)

    s_a = (l2 - Jxx) / safe(Jxt)                     # used when <PxPt> is large
    s_b = -Jxt / safe(l1 - Jxx)                      # otherwise
    # "<PxPt> >> 0": pick the better-conditioned branch
    slope = np.where(np.abs(Jxt) >= np.abs(l1 - Jxx), s_a, s_b)
    slope = np.clip(slope, -smax, smax)
    coherence = ((l1 - l2) / (l1 + l2 + 1e-30)) ** 2
    return slope, l1, l2, coherence


def estimate_slopes(P, sigma_d=2.5, sigma_w=3.5, half=5):
    Jxx, Jtt, Jxt = structure_tensor(P, sigma_d, sigma_w, half)
    return eigen_and_slope(Jxx, Jtt, Jxt)


# --------------------------------------------------------------------------
# Structure prediction filtering (eqs. 11-12)
# --------------------------------------------------------------------------
def _warp(trace, shift):
    """pred(t) = trace(t + shift(t)), linear interpolation."""
    t = np.arange(trace.size, dtype=float)
    return np.interp(t + shift, t, trace)


def _predictions(data, sigma, K):
    """fwd[k][:, j] = trace j predicted from trace j-k (chained, eq. 11);
       bwd[k][:, j] = trace j predicted from trace j+k."""
    nt, nx = data.shape
    fwd, bwd = [data], [data]
    for k in range(1, K + 1):
        f = np.zeros_like(data)
        b = np.zeros_like(data)
        for j in range(1, nx):
            f[:, j] = _warp(fwd[-1][:, j - 1], -sigma[:, j - 1])
        for j in range(nx - 1):
            b[:, j] = _warp(bwd[-1][:, j + 1], +sigma[:, j + 1])
        fwd.append(f)
        bwd.append(b)
    return fwd, bwd


def local_similarity(a, b, smooth=10.0, eps=1e-12):
    """Approximation of Fomel's (2007a) local similarity: smooth, in [0, 1],
    ~1 where traces agree locally, ~0 where they differ. (Fomel's original uses
    shaping regularisation; a Gaussian-smoothed ratio is used here.)"""
    S = lambda x: gaussian_filter1d(x, smooth, axis=0, mode="nearest")
    ab = S(a * b)
    a1 = np.clip(ab / (S(b * b) + eps), 0, 1)
    a2 = np.clip(ab / (S(a * a) + eps), 0, 1)
    return np.sqrt(a1 * a2)


def structure_prediction_filter(data, sigma, K=6, zeta=2.0,
                                use_similarity=True, sim_smooth=10.0):
    """Predict each trace from K neighbours per side, weight, and stack."""
    nt, nx = data.shape
    fwd, bwd = _predictions(data, sigma, K)
    num = data.copy()
    den = np.ones_like(data)
    cols = np.arange(nx)
    for k in range(1, K + 1):
        wk = np.exp(-(k ** 2) / zeta ** 2)           # eq. 12
        for P, valid in ((fwd[k], cols >= k), (bwd[k], cols <= nx - 1 - k)):
            s = local_similarity(data, P, sim_smooth) if use_similarity else 1.0
            w = wk * s * valid[None, :]
            num += w * P
            den += w
    return num / den


# --------------------------------------------------------------------------
# Palm / generic image: smoothing along the local line direction (v2)
# --------------------------------------------------------------------------
def oriented_smooth(img, sigma_d=1.0, sigma_w=4.0, half=None,
                    length=8, sigma_l=4.0):
    """Smooth along the direction of least variation (eigenvector of lambda_2)."""
    half = half or int(np.ceil(3 * sigma_w))
    Jxx, Jtt, Jxt = structure_tensor(img, sigma_d, sigma_w, half)
    theta = 0.5 * np.arctan2(2 * Jxt, Jxx - Jtt)     # gradient direction (x, y)
    dx, dy = -np.sin(theta), np.cos(theta)           # line direction (perp.)
    rows, cols = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(float)
    out = np.zeros_like(img, dtype=float)
    wsum = 0.0
    for o in range(-length, length + 1):
        w = np.exp(-(o ** 2) / (2 * sigma_l ** 2))
        out += w * map_coordinates(img, [rows + o * dy, cols + o * dx],
                                   order=1, mode="nearest")
        wsum += w
    return out / wsum


# --------------------------------------------------------------------------
# Synthetic test data (folded layers + noise), known slope = h'(x)
# --------------------------------------------------------------------------
def synthetic(nt=200, nx=200, noise=0.5, seed=0):
    t, x = np.mgrid[0:nt, 0:nx].astype(float)
    h = 40 + 25 * np.sin(2 * np.pi * x / 100.0)
    img = np.sin(2 * np.pi * (t - h) / 12.0)
    true_slope = 25 * 2 * np.pi / 100.0 * np.cos(2 * np.pi * x / 100.0)
    rng = np.random.default_rng(seed)
    return img + noise * rng.standard_normal(img.shape), img, true_slope


# --------------------------------------------------------------------------
# Plotting helpers + CLI
# --------------------------------------------------------------------------
def _show(panels, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    axs = np.atleast_1d(axs)
    for ax, (title, im, cmap, lim) in zip(axs, panels):
        kw = dict(vmin=-lim, vmax=lim) if lim else {}
        ax.imshow(im, cmap=cmap, aspect="auto", **kw)
        ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print("saved", out)


def est_noise(img):
    """Rough noise level: robust std of the Laplacian (white noise: var = 20 s^2)."""
    from scipy.ndimage import laplace
    L = laplace(img)
    return 1.4826 * np.median(np.abs(L - np.median(L))) / np.sqrt(20)


def run_denoise(data, a):
    """Generic 'give me any image, remove the noise' pipeline."""
    from PIL import Image
    lo, hi, mu, sd = data.min(), data.max(), data.mean(), data.std() + 1e-12
    x = (data - mu) / sd
    if a.method == "prediction":                  # paper's method (needs layered data)
        slope = estimate_slopes(x)[0]
        y = structure_prediction_filter(x, slope, K=a.K, zeta=a.zeta)
    else:                                         # smooth along structures, any image
        y = x
        for i in range(a.iters):
            print("pass", i + 1, "of", a.iters)
            y = oriented_smooth(y, sigma_d=1.5, sigma_w=4.0,
                                length=a.length, sigma_l=a.length / 2.0)
    y = y * sd + mu
    print("estimated noise level: before %.3f -> after %.3f"
          % (est_noise(data), est_noise(y)))
    base = os.path.splitext(a.out)[0]
    img8 = np.clip((y - lo) / (hi - lo + 1e-12) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img8).save(base + "_denoised.png")
    print("saved", base + "_denoised.png")
    # --- comparison with a plain Gaussian blur of user-chosen strength ---
    blur = gaussian_filter(data, a.blur_sigma)
    sharp = lambda z: np.mean(np.hypot(*np.gradient(gaussian_filter(z, 1.0))))
    print("\nComparison (estimated noise level / mean edge strength):")
    print("  noisy input              : %.2f / %.2f" % (est_noise(data), sharp(data)))
    print("  Gaussian blur (sigma=%.1f): %.2f / %.2f"
          % (a.blur_sigma, est_noise(blur), sharp(blur)))
    print("  structure tensor         : %.2f / %.2f" % (est_noise(y), sharp(y)))
    print("  (noise estimate is rough; judge edges visually in the figure)")
    _show([("input (noisy)", data, "gray", None),
           ("plain Gaussian blur (sigma=%.1f)" % a.blur_sigma, blur, "gray", None),
           ("structure-tensor denoised", y, "gray", None),
           ("noise removed by structure tensor", data - y, "gray", None)], a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["demo", "seismic", "palm", "denoise"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--dt", type=float, default=0.004, help="time sampling (s)")
    ap.add_argument("--dx", type=float, default=1.0, help="trace spacing")
    ap.add_argument("--zeta", type=float, default=2.0)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--no-agc", action="store_true")
    ap.add_argument("--out", default="result.png")
    ap.add_argument("--method", choices=["oriented", "prediction"],
                    default="oriented", help="denoise mode: filter type")
    ap.add_argument("--blur-sigma", type=float, default=2.0,
                    help="denoise: strength of the Gaussian-blur comparison")
    ap.add_argument("--iters", type=int, default=2, help="denoise: passes")
    ap.add_argument("--length", type=int, default=8,
                    help="denoise: smoothing half-length along structure (px)")
    a = ap.parse_args()

    if a.mode == "demo":
        noisy, clean, true_s = synthetic()
        slope, l1, l2, coh = estimate_slopes(noisy)
        sl = slope[20:-20, 20:-20]
        ts = true_s[20:-20, 20:-20]
        print("median |slope error| (samples/trace): %.3f"
              % np.median(np.abs(sl - ts)))
        filt = structure_prediction_filter(noisy, slope, K=a.K, zeta=a.zeta)
        rms = lambda z: np.sqrt(np.mean((z - clean) ** 2))
        print("RMS error vs clean: noisy %.3f -> filtered %.3f"
              % (rms(noisy), rms(filt)))
        lim = np.percentile(np.abs(slope), 98)
        _show([("noisy", noisy, "gray", None), ("slope", slope, "jet", lim),
               ("filtered", filt, "gray", None),
               ("removed", noisy - filt, "gray", None)], a.out)
        return

    data = load_array(a.path)
    if a.mode == "denoise":
        run_denoise(data, a)
        return
    if a.mode == "seismic":
        if not a.no_agc:
            data = agc(data, int(round(0.5 / a.dt)) | 1)
        slope, l1, l2, coh = estimate_slopes(data)
        filt = structure_prediction_filter(data, slope, K=a.K, zeta=a.zeta)
        phys = slope * a.dt / a.dx          # s per unit x
        lim = np.percentile(np.abs(phys), 98)
        _show([("input", data, "gray", None), ("slope", phys, "jet", lim),
               ("filtered", filt, "gray", None),
               ("difference", data - filt, "gray", None)], a.out)
    else:  # palm
        data = (data - data.mean()) / (data.std() + 1e-12)
        slope, l1, l2, coh = estimate_slopes(data, 1.0, 4.0, 12)
        smooth = oriented_smooth(data)
        ang = np.degrees(np.arctan(slope))
        _show([("palm", data, "gray", None), ("line angle (deg)", ang, "hsv", 90),
               ("coherence", coh, "viridis", None),
               ("oriented smoothing", smooth, "gray", None)], a.out)


if __name__ == "__main__":
    main()
