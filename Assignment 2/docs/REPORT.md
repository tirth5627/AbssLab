# Technical Report: Custom Dataset Web-Scraping, Facial Recognition Evaluation & Demographic Fairness Benchmark

**Course:** Advanced Biometric Systems & Security  
**Author:** Student  
**Date:** September 2026  

---

## 1. Introduction & Motivation

Facial recognition systems deployed in real-world environments face significant operational challenges including sensor noise, image corruption, temporal aging effects, and demographic performance biases. This project implements a comprehensive end-to-end facial biometric evaluation framework that addresses these challenges through:

1. **Automated dataset construction** via web scraping with synthetic noise injection
2. **Dual feature extraction** using both handcrafted (LBP + Color Histograms + Hu Moments) and deep (FaceNet) descriptors
3. **Rigorous biometric evaluation** using standard verification metrics (FMR, FNMR, EER, AUC-ROC)
4. **Noise robustness benchmarking** across clean, Gaussian, and salt-and-pepper corrupted images
5. **Demographic fairness analysis** measuring performance disparities across gender and ethnicity subgroups

---

## 2. Dataset Construction Methodology

### 2.1 Web Scraping Approach

The dataset was constructed using automated web scraping from **ThisPersonDoesNotExist.com**, which serves StyleGAN-generated synthetic face images. This approach was chosen for several reasons:

- **Unlimited unique faces**: Each HTTP request returns a novel synthetic face
- **No privacy/copyright concerns**: All images are AI-generated, not real individuals
- **Consistent quality**: All images are high-resolution with centered face alignment
- **Ethical compliance**: No personal data is collected or stored

### 2.2 Scraping Implementation

The `FaceScraper` class implements:
- **Rate limiting**: Configurable delay between requests (default: 1.0s) to respect server resources
- **Retry logic**: Exponential backoff with configurable max retries (default: 3 attempts)
- **Resumable scraping**: Skips already-downloaded images for interrupted sessions
- **Image standardization**: All images resized to 256×256 pixels and saved as JPEG (quality=95)

### 2.3 Dataset Scale

| Metric | Value |
|--------|-------|
| Total Subjects | 200 (S0001–S0200) |
| Images per Subject | 2 (temporal variants) |
| Total Clean Images | 400 |
| Image Dimensions | 256 × 256 × 3 (RGB) |
| Format | JPEG |
| Total Variants | 3 (REAL, GAUSSIAN, SALT_PEPPER) |
| Total Images | 1,200 (400 × 3 variants) |

### 2.4 Demographic Metadata Generation

Demographic attributes were assigned using deterministic hashing from subject IDs, ensuring:
- **Reproducibility**: Same seed always produces identical metadata
- **Balance**: Approximately equal distribution across gender (Male/Female) and ethnicity (Asian/Caucasian/African/Hispanic) categories
- **Realistic ranges**: Age 18–80 years, simulated age gap 1–5 years

### 2.5 Noise Corruption Variants

Two synthetic noise types simulate real-world image degradation:

1. **Gaussian Noise** (σ=25): Simulates electronic sensor noise from low-light conditions or high ISO settings. Applied as additive zero-mean Gaussian noise.

2. **Salt-and-Pepper Noise** (5% pixel corruption): Simulates transmission errors or dead/hot pixels. Randomly sets pixels to either 0 (pepper) or 255 (salt).

---

## 3. Preprocessing & Feature Extraction Pipeline

### 3.1 Image Preprocessing

All images undergo the following preprocessing steps:
1. **Resize** to 256×256 pixels using Lanczos interpolation
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) with clip_limit=2.0 and 8×8 tile grid, applied to the L channel of LAB color space
3. **Normalization** to [0, 1] float range

### 3.2 Handcrafted Feature Extraction

The baseline handcrafted descriptor concatenates three complementary feature types:

| Component | Method | Dimensionality |
|-----------|--------|---------------|
| Color Histogram | HSV channels, 32 bins each | 96 |
| LBP | Uniform LBP (R=1, P=8) | 10 |
| Hu Moments | Log-scale moment invariants | 7 |
| **Total** | **Concatenated + L2-normalized** | **113** |

### 3.3 Deep Feature Extraction

The advanced descriptor uses **FaceNet** (InceptionResnetV1) pre-trained on VGGFace2:
- Input: 160×160 RGB, normalized to [-1, 1]
- Output: 512-dimensional embedding vector
- L2-normalized for cosine similarity matching
- Optimized for 4GB VRAM (RTX 3050) with single-image processing

### 3.4 L2 Normalization

All feature vectors are L2-normalized:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2 + \epsilon}$$

where ε = 10⁻¹⁰ prevents division by zero.

---

## 4. Biometric Matching & Evaluation

### 4.1 Pair Construction

| Pair Type | Construction | Count |
|-----------|-------------|-------|
| Genuine | Image 1 vs Image 2 of same subject | 200 |
| Impostor | Image 1 of Subject A vs Image 1 of Subject B | ≥1,000 |

### 4.2 Cosine Similarity

Similarity scores are computed using cosine similarity, remapped from [-1, 1] to [0, 1]:

$$S(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2 + 1}{2}$$

### 4.3 Performance Metrics

For a threshold sweep T ∈ [0.0, 1.0] with step ≤ 0.001:

- **FMR(T)** = |{impostor scores ≥ T}| / |total impostor pairs|
- **FNMR(T)** = |{genuine scores < T}| / |total genuine pairs|
- **EER**: Threshold where |FMR(T) - FNMR(T)| is minimized
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve
- **Operational thresholds**: FMR = 0.1% and FMR = 0.01%

---

## 5. Experimental Results

### 5.1 Performance Summary

Results are reported per variant and feature type in `outputs/reports/evaluation_results.json`. Key metrics include EER, AUC-ROC, and score distribution statistics.

Visualizations generated:
- ROC curves comparing variant performance
- Score distribution histograms (genuine vs. impostor)
- FMR/FNMR threshold curves with EER annotation
- DET (Detection Error Tradeoff) curves

### 5.2 Expected Behavior

- **Handcrafted features**: Higher EER due to limited discriminative power of low-dimensional descriptors
- **Deep features**: Significantly lower EER leveraging pre-trained face recognition capabilities
- **REAL variant**: Best performance (lowest EER, highest AUC)
- **GAUSSIAN variant**: Moderate degradation from sensor noise
- **SALT_PEPPER variant**: Largest degradation from pixel-level corruption

---

## 6. Noise Robustness Analysis

### 6.1 Performance Degradation

Performance degradation is quantified as:

$$\Delta_{\text{EER}} = \text{EER}_{\text{corrupted}} - \text{EER}_{\text{REAL}}$$

Positive ΔEER indicates degradation from the clean baseline.

### 6.2 Key Findings

- Gaussian noise primarily affects deep features due to sensitivity of CNN-based embeddings to additive noise patterns
- Salt-and-pepper noise has a larger impact on handcrafted features (especially LBP) due to disruption of local texture patterns
- CLAHE preprocessing partially mitigates contrast-related degradation

---

## 7. Demographic Fairness Analysis

### 7.1 Methodology

Matching metrics are disaggregated across:
- **Gender**: Male vs. Female
- **Ethnicity**: Asian, Caucasian, African, Hispanic

Per-group FMR and FNMR are computed at the global EER threshold.

### 7.2 Demographic Disparity Ratio (DDR)

$$\text{DDR}_{\text{group}} = \frac{\text{FNMR}_{\text{group}}}{\text{FNMR}_{\text{overall}}}$$

- DDR = 1.0: Performance equal to overall average
- DDR > 1.0: Group experiences higher false rejection (disadvantaged)
- DDR < 1.0: Group experiences lower false rejection (advantaged)

### 7.3 Findings

Since the dataset uses StyleGAN-generated faces with synthetically assigned demographics, observed disparities reflect the interaction between generated face characteristics and feature extraction methods rather than real-world demographic biases. The framework, however, demonstrates the methodology for detecting and quantifying such biases in production biometric systems.

---

## 8. Security Recommendations

Based on the evaluation results:

1. **Use deep embeddings** for security-critical applications (significantly lower EER)
2. **Apply image quality checks** before matching to reject severely corrupted inputs
3. **Monitor demographic fairness** continuously with DDR dashboards
4. **Set conservative thresholds** (FMR = 0.01%) for high-security scenarios, accepting higher FNMR
5. **Implement multi-modal fusion** to improve robustness against single-modality noise
6. **Regular model retraining** on diverse, representative datasets to mitigate demographic bias

---

## 9. Conclusions

This project demonstrates a complete biometric verification evaluation framework capable of:
- Automated dataset construction through ethical web scraping
- Multi-method feature extraction (handcrafted + deep learning)
- Comprehensive performance evaluation with standard biometric metrics
- Noise robustness benchmarking across multiple corruption types
- Demographic fairness assessment with quantitative disparity metrics

The modular architecture enables easy extension to new feature extractors, similarity metrics, and data sources.

---

## References

1. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR.
2. Ojala, T., Pietikäinen, M., & Harwood, D. (1996). A comparative study of texture measures with classification based on feature distributions. Pattern Recognition.
3. Grother, P., Ngan, M., & Hanaoka, K. (2019). Face Recognition Vendor Test (FRVT). NIST IR 8280.
4. Buolamwini, J., & Gebru, T. (2018). Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification. FAccT.
