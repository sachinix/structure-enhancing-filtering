"""Edit the settings below, then press the Run button (triangle) in VS Code."""
from types import SimpleNamespace
from structure_tensor import load_array, run_denoise

# ---------------- SETTINGS: edit these ----------------
IMAGE = "noisy_layers.png"      # your image file name (same folder as this script)
METHOD = "oriented"     # "oriented" (any image)  or  "prediction" (seismic/layered data)
ITERS = 2               # more passes = smoother
LENGTH = 8              # smoothing distance along lines (pixels)
BLUR_SIGMA = 2.0        # strength of the plain-blur comparison
# -------------------------------------------------------

args = SimpleNamespace(method=METHOD, iters=ITERS, length=LENGTH,
                       blur_sigma=BLUR_SIGMA, K=6, zeta=2.0, out="result.png")
run_denoise(load_array(IMAGE), args)
print("Finished. Open result_denoised.png and result.png in the file list on the left.")
