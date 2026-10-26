# Comparative Study: PCA vs. Truncated SVD vs. LDA on the AT&T / ORL Face Database

**Course:** Advanced Biometrics Lab  
**Assignment:** Assignment 3 — Dimensionality Reduction & Facial Biometrics  
**Primary Dataset:** AT&T / ORL Database of Faces (400 images, 40 identities, 92 × 112 pixels)  
**Evaluation Protocol:** 10 Repeated Stratified Random Splits (5 Train / 5 Test per Identity)  

---

## 1. Executive Summary

This study delivers an empirical and theoretical investigation comparing **Principal Component Analysis (PCA)**, **Truncated Singular Value Decomposition (SVD)**, and **Linear Discriminant Analysis (LDA)** for grayscale face representation and biometric identity recognition. The benchmark utilizes the complete **AT&T / ORL Database of Faces**, comprising 400 grayscale PGM images across 40 distinct subjects.

### Central Research Question
> *How do unsupervised variance-oriented representations (PCA and Truncated SVD) compare with a supervised class-discriminative representation (LDA) when recognizing human identities from small grayscale face images?*

### Key Experimental Discoveries
1. **Subspace Equivalence of PCA and SVD:** On dense, centered image features, PCA (Eigenfaces) and Truncated SVD generate mathematically and empirically equivalent low-rank representations. Recognition accuracy curves and reconstruction error metrics across all component counts are virtually identical (e.g., at $k=39$, PCA achieves $94.55\% \pm 1.42\%$ vs. Truncated SVD's $94.60\% \pm 1.65\%$).
2. **Supervised Fisher Projections Dominate Identification:** LDA consistently outperforms both PCA and Truncated SVD at every evaluated component count ($k \in \{5, 10, 20, 30, 39\}$). At $k=5$, LDA achieves $83.15\%$ ($+2.8\%$ over PCA). At $k=39$, LDA attains **$95.20\% \pm 1.69\%$**, exceeding the uncompressed raw pixel baseline ($95.00\%$) while achieving a **264.2× compression ratio** (reducing 10,304 features down to 39).
3. **Reconstruction vs. Discrimination Dichotomy:** While PCA and SVD provide optimal low-rank Frobenius reconstructions (MSE dropping to $0.0070$ and PSNR reaching $21.58$ dB at $k=39$), LDA directions project along axes of maximum class separation, discarding intra-class variance. Consequently, LDA does not provide a faithful image reconstruction basis, confirming that optimal visual synthesis does not correlate with optimal biometric discrimination.
4. **Computational Real-Time Feasibility:** All three methods demonstrate rapid test-time feature projection: PCA takes $13.2\ \mu\text{s}$, Truncated SVD takes $12.7\ \mu\text{s}$, and LDA takes $22.9\ \mu\text{s}$ per image, ensuring real-time viability for biometric embedded devices.

---

## 2. Theoretical Background

### 2.1 Principal Component Analysis (PCA / Eigenfaces)
PCA is an unsupervised statistical technique that finds an orthogonal coordinate system where the variance of the projected data is maximized. Given a centered data matrix $\tilde{X} = X - \mu$, where each row is a flattened image vector of dimension $D = 10,304$, the sample covariance matrix is:
$$\Sigma = \frac{1}{N-1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{D \times D}$$
The principal directions $W = [w_1, w_2, \dots, w_k]$ correspond to the eigenvectors associated with the $k$ largest eigenvalues:
$$\Sigma w_i = \lambda_i w_i, \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_k$$
An image $x \in \mathbb{R}^D$ is projected into the $k$-dimensional subspace via:
$$z = W^T (x - \mu) \in \mathbb{R}^k$$
The low-rank reconstruction is recovered by:
$$\hat{x} = W z + \mu$$

### 2.2 Truncated Singular Value Decomposition (Truncated SVD)
SVD factorizes an arbitrary data matrix $X \in \mathbb{R}^{N \times D}$ into:
$$X = U \Sigma V^T$$
where $U \in \mathbb{R}^{N \times N}$ and $V \in \mathbb{R}^{D \times D}$ have orthonormal columns, and $\Sigma \in \mathbb{R}^{N \times D}$ contains singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. Truncated SVD retains the top $k$ components:
$$X \approx U_k \Sigma_k V_k^T$$
Projecting an image onto the leading right singular vectors yields $z = x V_k$. Reconstruction is given by $\hat{x} = z V_k^T$.
*Theoretical Equivalence:* When the data matrix is centered by subtracting column means, $X_{centered} = U \Sigma V^T$, the sample covariance matrix is:
$$\Sigma_{cov} = \frac{1}{N-1} X_{centered}^T X_{centered} = \frac{1}{N-1} V \Sigma^2 V^T$$
Thus, the right singular vectors $V$ of the centered data matrix are identical to the eigenvectors $W$ of the covariance matrix, and singular values relate to eigenvalues via $\lambda_i = \frac{\sigma_i^2}{N-1}$.

### 2.3 Linear Discriminant Analysis (LDA / Fisherfaces)
Unlike PCA and SVD, LDA is supervised. For $C$ classes (identities), LDA computes the between-class scatter matrix $S_B$ and within-class scatter matrix $S_W$:
$$S_B = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T$$
$$S_W = \sum_{c=1}^C \sum_{x \in \mathcal{C}_c} (x - \mu_c)(x - \mu_c)^T$$
LDA seeks projection vectors $W$ that maximize the Rayleigh quotient (Fisher criterion):
$$J(W) = \frac{\det(W^T S_B W)}{\det(W^T S_W W)}$$
Since $S_B$ has rank at most $C - 1$, the maximum number of discriminant components is bounded by $\min(D, C - 1)$. For the ORL dataset with $C=40$ classes, the upper bound is **$k_{max} = 39$**.

#### Small Sample Size (SSS) Problem and SVD Solver
In face recognition, the dimensionality $D = 10,304$ vastly exceeds the training sample size $N = 200$. Consequently, $S_W$ is rank-deficient and singular, preventing standard matrix inversion $S_W^{-1} S_B$. To prevent singularity and numerical instability, we utilize the **SVD solver** (`solver='svd'`), which computes within-class whitening via SVD without explicitly forming or inverting the $10,304 \times 10,304$ covariance matrix.

---

## 3. Dataset & Preprocessing

The benchmark is conducted on the official Cambridge **AT&T / ORL Database of Faces**:
- **Subjects:** 40 individuals ($C = 40$)
- **Samples per Subject:** 10 images ($N_c = 10$, total $N = 400$)
- **Resolution:** $92 \times 112$ pixels (grayscale PGM)
- **Flattened Dimensionality:** $D = 92 \times 112 = 10,304$ features
- **Variations:** Lighting, facial expressions (open/closed eyes, smiling/neutral), accessories (glasses/no glasses), and subtle facial tilt.
- **Normalization:** Raw 8-bit unsigned integers $[0, 255]$ are normalized to floating-point range $[0.0, 1.0]$.

### Fair Comparison & Zero Data Leakage Rules
To prevent data snooping:
1. Training and testing splits are strictly stratified: exactly 5 training images and 5 testing images per subject per repetition.
2. Centering means, singular vectors, projection bases, and classifier weights are computed **strictly on training partitions only**.
3. Held-out test images are evaluated solely using transformations learned during the training phase.

---

## 4. Experimental Setup (Table A)

| Parameter | Value |
| :--- | :--- |
| **Dataset** | AT&T / ORL Database of Faces |
| **Total Images / Identities** | 400 images / 40 people |
| **Images per Identity** | 10 images per subject |
| **Image Dimensions** | $92 \times 112$ pixels (Grayscale PGM) |
| **Flattened Features per Image** | $10,304$ dimensions |
| **Pixel Normalization Scale** | $[0.0, 1.0]$ float32 |
| **Train / Test Partition** | 5 train + 5 test per identity (200 train / 200 test) |
| **Split Repetitions** | 10 independent randomized stratified splits |
| **Random Seeds** | `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]` |
| **Primary Classifier** | 1-Nearest Neighbor (1-NN, Euclidean distance) |
| **Secondary Classifier** | Linear Support Vector Machine ($C=1.0$) |
| **Evaluated Components ($k$)** | $5, 10, 20, 30, 39$ (plus extended $50, 75, 100, 150$) |

---

## 5. Measured Experimental Results

### 5.1 Recognition Performance Benchmark (Table B)

The table below presents the empirically measured identification performance (Mean Accuracy $\pm$ Std Dev across 10 independent splits, Macro-F1 score, and Training Fit Time in milliseconds).

| Method | Components ($k$) | Mean Accuracy (%) | Std Dev (%) | Macro-F1 Score | Fit Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw pixels + 1-NN (Baseline)** | **10,304** | **95.00%** | **±1.16%** | **0.9486 ± 0.0123** | **0.00 ms (None)** |
| PCA + 1-NN | 5 | 80.35% | ±2.57% | 0.7975 ± 0.0271 | 33.08 ms |
| PCA + 1-NN | 10 | 91.35% | ±1.47% | 0.9097 ± 0.0165 | 35.10 ms |
| PCA + 1-NN | 20 | 93.30% | ±2.06% | 0.9296 ± 0.0219 | 34.47 ms |
| PCA + 1-NN | 30 | 94.50% | ±1.62% | 0.9422 ± 0.0175 | 42.05 ms |
| PCA + 1-NN | 39 | 94.55% | ±1.42% | 0.9434 ± 0.0150 | 56.43 ms |
| Truncated SVD + 1-NN | 5 | 80.40% | ±2.69% | 0.7960 ± 0.0284 | 23.46 ms |
| Truncated SVD + 1-NN | 10 | 91.50% | ±1.73% | 0.9113 ± 0.0195 | 25.90 ms |
| Truncated SVD + 1-NN | 20 | 93.35% | ±1.63% | 0.9303 ± 0.0172 | 38.90 ms |
| Truncated SVD + 1-NN | 30 | 94.35% | ±1.76% | 0.9407 ± 0.0188 | 52.76 ms |
| Truncated SVD + 1-NN | 39 | 94.60% | ±1.65% | 0.9440 ± 0.0172 | 68.74 ms |
| **LDA + 1-NN** | 5 | 83.15% | ±3.25% | 0.8235 ± 0.0333 | 202.67 ms |
| **LDA + 1-NN** | 10 | 92.60% | ±2.47% | 0.9218 ± 0.0276 | 200.42 ms |
| **LDA + 1-NN** | 20 | 94.65% | ±1.70% | 0.9440 ± 0.0190 | 199.98 ms |
| **LDA + 1-NN** | 30 | 95.10% | ±1.74% | 0.9489 ± 0.0189 | 209.00 ms |
| **LDA + 1-NN (Best)** | **39** | **95.20%** | **±1.69%** | **0.9499 ± 0.0177** | **211.15 ms** |

---

### 5.2 Reconstruction and Compression Footprint (Table C)

| Method | Components ($k$) | Reconstruction MSE | Relative Frobenius Error | PSNR (dB) | Feature Storage / Image | Model Parameter Storage | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PCA** | 10 | 0.010302 | 21.01% | 19.87 dB | 40 B (0.04 KB) | 442.75 KB | Subtracted mean face (centered). Exact orthogonal inverse transform. |
| **Truncated SVD** | 10 | 0.010372 | 21.08% | 19.84 dB | 40 B (0.04 KB) | 442.75 KB | Uncentered data matrix. Projects on leading right singular vectors. |
| **PCA** | 20 | 0.008387 | 18.95% | 20.76 dB | 80 B (0.08 KB) | 845.25 KB | Subtracted mean face (centered). Exact orthogonal inverse transform. |
| **Truncated SVD** | 20 | 0.008409 | 18.98% | 20.75 dB | 80 B (0.08 KB) | 845.25 KB | Uncentered data matrix. Projects on leading right singular vectors. |
| **PCA** | 39 | 0.006957 | 17.26% | 21.58 dB | 156 B (0.15 KB) | 1,610.00 KB | Subtracted mean face (centered). Exact orthogonal inverse transform. |
| **Truncated SVD** | 39 | 0.006944 | 17.25% | 21.58 dB | 156 B (0.15 KB) | 1,610.00 KB | Uncentered data matrix. Projects on leading right singular vectors. |
| **LDA** | 39 | N/A | N/A | N/A | 156 B (0.15 KB) | 1,610.00 KB | Not primary: LDA optimizes class discriminability, not image reconstruction. |

---

### 5.3 Computational Efficiency Benchmarks (Table D)

| Method | Evaluated $k$ | Fit Time (ms) | Transform Time / Image (µs) | Model Storage | Feature Compression Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PCA** | 39 | 56.43 ± 13.82 ms | 13.2 ± 2.7 µs | 1,610.00 KB | 264.2× |
| **Truncated SVD** | 39 | 68.74 ± 17.70 ms | 12.7 ± 2.0 µs | 1,610.00 KB | 264.2× |
| **LDA** | 39 | 211.15 ± 43.01 ms | 22.9 ± 4.1 µs | 1,610.00 KB | 264.2× |

---

### 5.4 Best Configurations Comparison Summary

| Method | Best Components ($k$) | Peak Mean Accuracy (%) | Macro-F1 Score | Training Fit Time | Compression vs. Raw |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PCA** | 100 | 94.95% ± 1.40% | 0.9482 ± 0.0145 | 190.28 ms | 103.0× |
| **Truncated SVD** | 100 | 94.95% ± 1.54% | 0.9481 ± 0.0158 | 191.67 ms | 103.0× |
| **LDA (Overall Champion)** | **39** | **95.20% ± 1.69%** | **0.9499 ± 0.0177** | **211.15 ms** | **264.2×** |

---

## 6. Visualizations & Graphical Analysis

### Plot 1: Recognition Accuracy vs. Number of Retained Components
![Plot 1: Recognition Accuracy vs Components](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/outputs/figures/plot1_accuracy_vs_components.png)
*Figure 1: Identification accuracy (%) vs number of subspace components ($k \in \{5, 10, 20, 30, 39\}$) across 10 randomized stratified splits with $\pm 1$ standard deviation error bands and Raw Pixel baseline.*

**Analysis:**
- **PCA and SVD Curves Overlap:** The trajectories for PCA and Truncated SVD are virtually indistinguishable across the entire sweep. Both start at $\approx 80.4\%$ at $k=5$, jump to $\approx 91.4\%$ at $k=10$, and level off at $\approx 94.6\%$ at $k=39$.
- **LDA Superiority:** The red curve representing LDA sits strictly above PCA and SVD at all component counts. At $k=30$ and $k=39$, LDA reaches $95.10\%$ and $95.20\%$, surpassing the raw pixel baseline of $95.00\%$.

---

### Plot 2: Original Faces and Low-Rank Reconstructions
![Plot 2: Original Face vs Reconstructions](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/outputs/figures/plot2_face_reconstructions.png)
*Figure 2: Low-rank face reconstruction for held-out test faces (Subject 1 and Subject 10) comparing PCA and Truncated SVD across $k \in \{5, 10, 20, 30\}$.*

**Analysis:**
- At $k=5$, reconstructions capture the general head outline and coarse illumination.
- At $k=10$ and $k=20$, distinct facial features (eyes, nose, lip structure, eyeglasses) emerge.
- At $k=30$, high-frequency biometric details are reconstructed faithfully.
- Visual differences between PCA and SVD reconstructions are imperceptible to human inspection, confirming empirical subspace equivalence.

---

### Plot 3: Reconstruction Error vs. Component Count
![Plot 3: Reconstruction Error](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/outputs/figures/plot3_reconstruction_error.png)
*Figure 3: Reconstruction Mean Squared Error (MSE) and Relative Frobenius Error (%) on held-out test images as a function of retained components.*

**Analysis:**
- Reconstruction MSE monotonically decays from $0.0128$ ($k=5$) to $0.0070$ ($k=39$).
- Relative Frobenius Error drops from $23.45\%$ down to $17.26\%$.
- Curves for PCA and Truncated SVD align perfectly, illustrating identical low-rank matrix energy preservation.

---

### Plot 4: PCA Cumulative Explained Variance
![Plot 4: PCA Cumulative Explained Variance](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/outputs/figures/plot4_explained_variance.png)
*Figure 4: PCA cumulative explained variance ratio on ORL faces, highlighting the 90% and 95% energy retention thresholds.*

**Analysis:**
- The top $39$ components retain $76.5\%$ of the total data variance.
- Reaching $90\%$ cumulative variance requires $k \approx 85$ components.
- Reaching $95\%$ cumulative variance requires $k \approx 135$ components.
- **Critical Insight:** High explained variance does not directly dictate classification accuracy: LDA achieves higher classification accuracy at $k=39$ ($95.20\%$) than PCA at $k=150$ ($94.65\%$), proving that biometric discrimination depends on inter-class variance rather than total variance.

---

### Plot 5: Two-Dimensional Subspace Embeddings (Visual Class Separation)
![Plot 5: 2D Subspace Embeddings](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/outputs/figures/plot5_2d_embeddings.png)
*Figure 5: 2D scatter embeddings of 8 subject identities projected onto the first two components of PCA, Truncated SVD, and LDA.*

**Analysis:**
- **PCA and SVD Embeddings (Panels 1 & 2):** In both unsupervised projections, identity clusters overlap significantly. The first two components predominantly capture global lighting and head pose rather than person identity.
- **LDA Embedding (Panel 3):** In the supervised Fisher space, individual identities form tight, isolated clusters with large inter-class margins. This visually validates why LDA yields higher accuracy even at extreme low dimensions.

---

### Plot 6: 40 × 40 Identity Confusion Matrices
![Plot 6: Confusion Matrices](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/outputs/figures/plot6_confusion_matrices.png)
*Figure 6: 40 × 40 test identity confusion matrices for best PCA, Truncated SVD, and LDA configurations.*

**Analysis:**
- All three confusion matrices show concentrated diagonal activation, reflecting strong baseline biometric recognition across all 40 subjects.
- Off-diagonal misclassifications are minimal and occur primarily between subjects with heavy overlapping traits (e.g. dark hair, dark-rimmed glasses, or pronounced smiling variations).

---

### Plot 7: Computational Efficiency Benchmarks
![Plot 7: Computational Efficiency](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/outputs/figures/plot7_computational_cost.png)
*Figure 7: Computational benchmark comparing Training Fit Time (ms) and Test Transform Time per Image (µs).*

**Analysis:**
- PCA and SVD require only $56.4$ ms and $68.7$ ms to fit 200 training images ($10,304$ dimensions each).
- LDA requires $211.2$ ms due to between-class and within-class scatter calculations across 40 classes.
- Test-time projection is virtually instantaneous across all models: $13.2\ \mu\text{s}$ (PCA), $12.7\ \mu\text{s}$ (SVD), and $22.9\ \mu\text{s}$ (LDA), confirming real-time embedded capability.

---

## 7. Deep Analytical Answers to Core Research Questions

### Q1: Why do PCA and Truncated SVD perform almost identically on ORL?
On a centered dense matrix $X_{proc} = X - \mu$, the right singular vectors of $X_{proc}$ are mathematically identical to the eigenvectors of the sample covariance matrix $\Sigma = \frac{1}{N-1} X_{proc}^T X_{proc}$. When SVD is run without centering, the first singular vector aligns closely with the mean illumination vector $\mu$, while subsequent singular vectors span the same subspace as the principal components. Thus, for biometric classification using nearest neighbors, Euclidean distance relationships between faces in PCA and SVD subspaces are practically invariant.

### Q2: Why does LDA outperform PCA and Truncated SVD for recognition?
PCA is unsupervised and seeks axes of maximal total variance $\max_W \text{Tr}(W^T S_T W)$. In face datasets, the directions of largest variance frequently represent non-biometric variations—such as lighting direction, shadows, and background contrast. In contrast, LDA is supervised and optimizes the Fisher ratio $\max_W \frac{\det(W^T S_B W)}{\det(W^T S_W W)}$. LDA systematically finds directions that amplify between-subject separation while shrinking intra-subject variation (expression and lighting).

### Q3: Why does LDA beat the raw pixel baseline with a 264× compression factor?
Raw pixel representations ($10,304$ features) suffer from the *curse of dimensionality* and include significant high-frequency noise, shadows, and redundant background pixels. By compressing each face to the 39 optimal discriminant Fisher coordinates, LDA acts as an intelligent denoising filter that retains purely discriminative biometric features while eliminating non-identity noise.

### Q4: Why is LDA unsuitable as a direct image reconstruction method?
Image reconstruction requires preserving both between-class and within-class pixel variations. Because LDA explicitly projects data along directions that minimize within-class variance, the reconstruction operator loses the fine personal details needed to reproduce the original face image. PCA and SVD remain the mathematically optimal linear techniques for reconstruction (by the Eckart-Young-Mirsky theorem).

---

## 8. Conclusion and Future Work

1. **For Biometric Recognition:** LDA (Fisherfaces) is the unambiguously superior subspace technique, delivering $95.20\%$ accuracy at $k=39$ with a 264.2× compression factor.
2. **For Compression and Synthesis:** PCA and Truncated SVD are optimal, reducing image storage from $41.2$ KB to $156$ B while retaining $>21.5$ dB PSNR.
3. **Future Extension:** The established pipeline can be directly evaluated on the **Extended Yale Face Database B** to benchmark robustness under severe directional illumination and shadows.

---

## 9. Verification & Codebase Artifacts

- **Pipeline Script:** [`scripts/run_pipeline.py`](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/scripts/run_pipeline.py)
- **Word Report Generator:** [`scripts/update_word_report.py`](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/scripts/update_word_report.py)
- **Completed Word Document:** [`PCA_SVD_LDA_ORL_Face_Dataset_Comparison_Report_Completed.docx`](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/PCA_SVD_LDA_ORL_Face_Dataset_Comparison_Report_Completed.docx)
- **Interactive Jupyter Notebook:** [`assignment3.ipynb`](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/assignment3.ipynb)
- **Test Suite:** [`tests/test_pipeline.py`](file:///c:/Users/04khu/Desktop/7th%20Sem/AdvanceBiometricsLab/Assignment%203/tests/test_pipeline.py) (All 6 test cases passed)
