# Structure Enhancing Filtering with the Structure Tensor

A Python-based image filtering project that uses the **Structure Tensor** to reduce noise while preserving important image structures, edges, and patterns.

## 1. Project Overview

Traditional image filtering techniques, such as Gaussian Blur, reduce noise but can also blur important image details.

This project implements structure-aware filtering using the Structure Tensor. It estimates local image orientations and uses this information to perform smoothing along the detected structural directions.

The main objective is to achieve noise reduction while preserving important image details.

## 2. Objectives

- Understand and implement the Structure Tensor.
- Calculate image derivatives along X and Y directions.
- Estimate local image orientations using eigenvalues and eigenvectors.
- Implement structure-oriented smoothing.
- Implement structure prediction filtering.
- Compare structure-aware filtering with conventional Gaussian Blur.

## 3. Methodology

The project follows these steps:

1. **Input Image:** Load the noisy image and convert it into a NumPy array.
2. **Normalization:** Normalize pixel intensity values.
3. **Gaussian Derivatives:** Calculate image derivatives along X and Y directions.
4. **Structure Tensor:** Construct the Structure Tensor using image derivatives.
5. **Eigenvalue and Eigenvector Calculation:** Identify local structural information.
6. **Orientation Estimation:** Estimate the local direction of image structures.
7. **Structure-Aware Filtering:** Apply filtering using the estimated local orientation.
8. **Output:** Save and compare the filtered image with the original input.

## 4. Implemented Filtering Methods

### Method 1: Structure-Oriented Smoothing

- Estimate the local structure direction.
- Select neighboring samples along the detected direction.
- Assign Gaussian weights to the selected samples.
- Calculate the weighted average.
- Repeat the smoothing process for the specified number of iterations.

### Method 2: Structure Prediction Filtering

- Estimate local slopes using the Structure Tensor.
- Predict neighboring traces.
- Calculate local similarity.
- Apply Gaussian taper weights.
- Combine neighboring information using weighted stacking.

**Note:** The current palm-like image demonstration uses Structure-Oriented Smoothing. Structure Prediction Filtering is implemented as a separate filtering approach.

## 5. Structure Tensor

The Structure Tensor is represented as:

\[
M =
\begin{bmatrix}
\langle P_x^2\rangle & \langle P_xP_t\rangle\\
\langle P_xP_t\rangle & \langle P_t^2\rangle
\end{bmatrix}
\]

Where:

- \(P_x\): Image intensity change along the X-direction.
- \(P_t\): Image intensity change along the Y-direction.
- Eigenvalues represent the strength of intensity variation.
- Eigenvectors represent the corresponding directions.

This information helps identify local image structures and guide the filtering process.

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical and array operations |
| SciPy | Gaussian filtering, derivatives and interpolation |
| Matplotlib | Image visualization and comparison |
| Pillow (PIL) | Image loading and saving |

## 7. Project Structure

```text
structure-enhancing-filtering/
│
├── structure_tensor.py
├── run_denoise.py
├── noisy_layers.png
├── result.png
├── result_denoised.png
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/sachinix/structure-enhancing-filtering.git
```

### Step 2: Open the project directory

```bash
cd structure-enhancing-filtering
```

### Step 3: Install dependencies

```bash
pip install numpy scipy matplotlib pillow
```

### Step 4: Run the project

```bash
python run_denoise.py
```

The program processes the input image and saves the filtered output.

## 9. Experimental Results

The project compares three outputs:

- Original noisy input image.
- Conventional Gaussian Blur output.
- Structure Tensor-based filtering output.

The implementation also provides estimated noise levels and mean edge strength for comparison.

The current experiment demonstrates that structure-aware filtering can reduce noise while retaining more structural information than conventional Gaussian Blur in the tested example.

## 10. Limitations and Future Work

- Test the implementation on different types of images.
- Apply the filtering technique to real seismic datasets.
- Improve noise reduction while preserving fine details.
- Experiment with different smoothing parameters and iterations.
- Compare the performance of both implemented filtering methods.

## 11. Research Paper Reference

**Title:** Structure Enhancing Filtering with the Structure Tensor

**Authors:** Rodrigo Morelatto and Ricardo Biloti

**Year:** 2013

This project is based on the concepts presented in the above research paper.

## 12. Author

**Sachin Kumar**  
B.Tech – Computer Science and Engineering  
IIT (ISM) Dhanbad

**GitHub:** https://github.com/sachinix

---

*This project was developed as an academic project to study and implement structure-aware image filtering using the Structure Tensor.*
