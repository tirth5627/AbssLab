# Assignment 3: PCA vs. Truncated SVD vs. LDA on ORL Face Dataset

**Course:** Advanced Biometrics Lab  
**Domain:** Facial Biometrics, Dimensionality Reduction & Subspace Learning  
**Dataset:** AT&T / ORL Database of Faces (400 images, 40 subjects, $92 \times 112$ grayscale)

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd "Assignment 3"
python -m pip install -r requirements.txt
```

### 2. Run the Benchmark Pipeline
Run all 10 randomized stratified splits, calculate all metrics, generate summary tables, and export all 7 publication-grade figures:
```bash
python scripts/run_pipeline.py
```

### 3. Update / Regenerate the Completed Word Report (.docx)
Populate all tables, insert all measured values, embed figures, and compile the final Word report:
```bash
python scripts/update_word_report.py
```

### 4. Run the Unit Test Suite
```bash
python -m pytest tests/ -v
```

### 5. Launch the Interactive HTML Blog
Simply double-click `index.html` or start a local web server:
```bash
python -m http.server 8000
# Navigate to http://localhost:8000/ in your browser
```

### 6. Open the Interactive Jupyter Notebook
```bash
jupyter notebook assignment3.ipynb
```

---

## 📊 Summary of Experimental Results

### Recognition Performance across 10 Stratified Splits (1-NN Classifier)
- **Raw Pixels Baseline (10,304 dims):** $95.00\% \pm 1.16\%$ (Macro-F1: $0.9486 \pm 0.0123$)
- **PCA (Eigenfaces):**
  - $k=5$: $80.35\% \pm 2.57\%$
  - $k=10$: $91.35\% \pm 1.47\%$
  - $k=20$: $93.30\% \pm 2.06\%$
  - $k=30$: $94.50\% \pm 1.62\%$
  - $k=39$: $94.55\% \pm 1.42\%$ (Peak at $k=100$: $94.95\%$)
- **Truncated SVD:**
  - $k=5$: $80.40\% \pm 2.69\%$
  - $k=10$: $91.50\% \pm 1.73\%$
  - $k=20$: $93.35\% \pm 1.63\%$
  - $k=30$: $94.35\% \pm 1.76\%$
  - $k=39$: $94.60\% \pm 1.65\%$ (Peak at $k=100$: $94.95\%$)
- **LDA (Fisherfaces — Overall Winner):**
  - $k=5$: $83.15\% \pm 3.25\%$
  - $k=10$: $92.60\% \pm 2.47\%$
  - $k=20$: $94.65\% \pm 1.70\%$
  - $k=30$: $95.10\% \pm 1.74\%$
  - **$k=39$:** **$95.20\% \pm 1.69\%$** (Exceeds raw pixels with **264.2× feature compression**)

---

## 📁 Directory Architecture

```
Assignment 3/
├── config/
│   └── default_config.yaml                  # Experiment hyperparameters and paths
├── data/
│   ├── s1/ ... s40/                         # 40 subject folders with 10 PGM images each
│   └── att_faces.zip                        # Official archive
├── docs/
│   └── REPORT.md                            # Comprehensive 12-section technical report
├── outputs/
│   ├── figures/                             # 7 publication-grade 300 DPI figures
│   │   ├── plot1_accuracy_vs_components.png
│   │   ├── plot2_face_reconstructions.png
│   │   ├── plot3_reconstruction_error.png
│   │   ├── plot4_explained_variance.png
│   │   ├── plot5_2d_embeddings.png
│   │   ├── plot6_confusion_matrices.png
│   │   └── plot7_computational_cost.png
│   └── tables/                              # All measured metrics in CSV & Markdown
│       ├── best_configurations_summary.csv
│       ├── raw_benchmark_records.csv
│       ├── summary_benchmark_records.csv
│       ├── table_a_configuration.csv
│       ├── table_b_recognition_performance.csv
│       ├── table_c_reconstruction_compression.csv
│       └── table_d_computational_efficiency.csv
├── scripts/
│   ├── run_pipeline.py                      # End-to-end benchmark execution script
│   ├── update_word_report.py                # Injects real measurements & plots into DOCX
│   └── generate_notebook.py                 # Assembles assignment3.ipynb
├── src/
│   ├── __init__.py                          # Package exports
│   ├── data_loader.py                       # ORL loader & stratified split generator
│   ├── models.py                            # PCA, Truncated SVD, and LDA implementations
│   ├── evaluation.py                        # Metrics, MSE, Frobenius error, storage
│   └── visualization.py                     # High-res plotting functions
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                     # Pytest suite (6 passed tests)
├── assignment3.ipynb                        # Fully executed interactive Jupyter Notebook
├── PCA_SVD_LDA_ORL_Face_Dataset_Comparison_Report.docx           # Updated original DOCX
├── PCA_SVD_LDA_ORL_Face_Dataset_Comparison_Report_Completed.docx # Completed DOCX report
├── REPORT.md                                # Root technical markdown report
├── README.md                                # This documentation guide
└── requirements.txt                         # Python dependencies
```

---

## 🏆 Key Scientific Takeaways
1. **PCA vs. SVD Equivalence:** Empirical validation confirms that on dense centered face matrices, PCA and Truncated SVD span essentially identical low-rank subspaces.
2. **Supervised Fisher Optimization:** LDA systematically outperforms unsupervised variance-maximizing methods for biometric identity classification because it separates identities rather than lighting or non-facial variance.
3. **Reconstruction vs. Discrimination:** PCA and SVD minimize pixel reconstruction error (Eckart-Young theorem), making them ideal for compression and image synthesis, whereas LDA is strictly designed for class discrimination.
