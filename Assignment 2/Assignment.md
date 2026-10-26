# 🎓 Assignment: Custom Dataset Web-Scraping, Facial Recognition Evaluation & Demographic Fairness Benchmark

**Course:** Advanced Biometric Systems & Security  
**Topic:** Web Scraping & Dataset Generation, Facial Verification, Noise Corruption Robustness, Performance Metrics (FMR/FNMR/EER), & Demographic Disparity  
**Target Level:** Postgraduate / Advanced Undergraduate  

---

## 📌 Executive Overview

Facial recognition systems deployed in real-world environments face significant operational challenges: sensor noise, image corruption, temporal aging, and demographic performance biases.

> [!IMPORTANT]
> **Dataset Requirement Note:**  
> Students will **NOT** use any pre-existing dataset provided in this workspace. Instead, you must **build and construct your own facial dataset** via **web scraping** (or alternative automated data collection/generation methods). However, your scraped dataset must strictly satisfy the exact benchmark conditions (200 subjects, $\ge 2$ images per subject with temporal/aging gaps, demographic metadata, and 3 image corruption variants: REAL, GAUSSIAN, SALT_PEPPER).

---

## 🎯 1. What Must Be Done (Task Requirements)

You are required to implement a modular end-to-end Python framework that performs the following 6 core tasks:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PIPELINE WORKFLOW                                    │
│                                                                                        │
│  [1. Web Scraping & Noise Pipeline] ──► [2. Preprocessing] ──► [3. Feature Extract]   │
│                                                                       │                │
│                                                                       ▼                │
│  [6. Fairness Analysis] ◄── [5. Robustness Test] ◄── [4. Score Matching & Metrics]    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Task 1: Web Scraping, Dataset Construction & Metadata Generation
- **Automated Web Scraping:** Build a web crawler / scraper (e.g., using `BeautifulSoup`, `Selenium`, `Playwright`, `Scrapy`, or search engine APIs) to collect face images across public web sources or synthetic face repositories.
- **Dataset Scale Criteria:**
  - Collect images for **200 distinct subjects** ($S0001$ to $S0200$).
  - Obtain **at least 2 images per subject** ($SXXXX\_001.jpg$, $SXXXX\_002.jpg$) representing distinct temporal instances, lighting conditions, or poses (minimum target 1+ year simulated or actual age/time gap).
- **Metadata Tagging:** Extract or generate structured metadata (`metadata.json`) containing demographic variables for each subject:
  - **Age** (e.g., 18–80 years), **Gender** (Male / Female), **Ethnicity** (Asian, Caucasian, African, Hispanic), and **Age Gap** (1–5 years).
- **Synthetic Corruption Generation:** Implement automated noise injection functions to generate 3 image variants:
  1. `REAL`: Original clean scraped images ($256 \times 256$ pixels, JPEG format).
  2. `GAUSSIAN`: Programmatically added Gaussian sensor noise.
  3. `SALT_PEPPER`: Programmatically added Salt-and-Pepper transmission error noise.

### Task 2: Modular Preprocessing & Feature Extraction Engine
- **Preprocessing:** Standardize all images to $256 \times 256$ pixels RGB/Grayscale and apply contrast enhancement (e.g., CLAHE / histogram equalization).
- **Feature Extraction:** Implement at least **two** feature representation techniques:
  - *Baseline Handcrafted Descriptor:* Color Histograms + Spatial Image Moments / Local Binary Patterns (LBP).
  - *Advanced / Deep Descriptor:* Deep Facial Embeddings (e.g., FaceNet / ResNet / OpenFace / MobileFaceNet embeddings or normalized feature ensembles).
- L2-normalize all feature vectors:
  $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2 + \epsilon}$$

### Task 3: Biometric Pair Matching & Similarity Scoring
- Construct genuine and impostor comparison pairs:
  - **Genuine Pairs (200 pairs):** Image 1 vs Image 2 of the *same* scraped subject.
  - **Impostor Pairs ($\ge 1000$ pairs):** Image 1 of Subject $A$ vs Image 1 of Subject $B$ ($A \neq B$).
- Compute comparison scores using **Cosine Similarity**:
  $$S(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\|_2 \|\mathbf{v}_2\|_2}$$

### Task 4: Quantitative Performance Metrics Calculation
For a threshold sweep $T \in [0.0, 1.0]$ with step size $\le 0.001$, calculate:
1. **False Match Rate (FMR / FAR):**
   $$\text{FMR}(T) = \frac{\text{Number of Impostor Pairs with Score } \ge T}{\text{Total Impostor Pairs}}$$
2. **False Non-Match Rate (FNMR / FRR):**
   $$\text{FNMR}(T) = \frac{\text{Number of Genuine Pairs with Score } < T}{\text{Total Genuine Pairs}}$$
3. **Equal Error Rate (EER):** Find threshold $T_{\text{EER}}$ where $|\text{FMR}(T) - \text{FNMR}(T)|$ is minimized.
4. **Area Under Curve (AUC-ROC):** Calculate overall discriminative accuracy.
5. **Operational Thresholds:** Identify strict security operating thresholds at $\text{FMR} = 0.1\%$ and $\text{FMR} = 0.01\%$.

### Task 5: Robustness & Noise Corruption Impact Analysis
- Evaluate performance metrics ($EER$, $AUC$, score distributions) across all 3 scraped variants (`REAL`, `GAUSSIAN`, `SALT_PEPPER`).
- Compute performance degradation score: $\Delta_{\text{EER}} = \text{EER}_{\text{corrupted}} - \text{EER}_{\text{REAL}}$.
- Benchmark performance decay relative to temporal aging gaps.

### Task 6: Demographic Bias & Fairness Assessment
- Disaggregate matching metrics across demographic subgroups:
  - **Gender Disparity:** Male vs. Female.
  - **Ethnic Disparity:** Asian, Caucasian, African, Hispanic.
- Compute sub-group FMR and FNMR at global threshold $T_{\text{EER}}$ and calculate the **Demographic Disparity Ratio (DDR)**:
  $$\text{DDR}_{\text{group}} = \frac{\text{FNMR}_{\text{group}}}{\text{FNMR}_{\text{overall}}}$$

---

## 📦 2. Required Deliverables

Submissions must be packaged cleanly according to the new directory standard outlined in Section 3.

| # | Deliverable Item | Description & Technical Format |
|---|------------------|--------------------------------|
| **D0** | **Dataset Scraper & Generator (`src/dataset_generator/` & `scripts/scrape_dataset.py`)** | Automated web scraper crawler script and noise corruption generator used to build the custom 200-subject dataset and metadata (`metadata.json`). |
| **D1** | **Modular Source Code (`src/`)** | Fully documented Python package implementing preprocessing, feature extraction, score matching, evaluation metrics, and visualization modules. |
| **D2** | **Executable Pipeline Entrypoint (`scripts/run_pipeline.py`)** | Command-Line Interface (CLI) allowing execution of evaluation pipelines with configurable arguments (e.g. `--variant REAL`, `--feature-type deep`, `--export-report`). |
| **D3** | **Quantitative Results JSON (`outputs/reports/evaluation_results.json`)** | Machine-readable JSON output containing FMR, FNMR, EER, AUC-ROC, and score distribution metrics per variant and per demographic group. |
| **D4** | **High-Resolution Figures (`outputs/figures/`)** | Publication-quality visualizations including:<br>• ROC & DET curves per dataset variant<br>• Score distribution histograms (Genuine vs. Impostor)<br>• Demographic parity bar charts (FMR/FNMR by gender/ethnicity)<br>• Noise degradation comparison plots |
| **D5** | **Automated Test Suite (`tests/`)** | Unit and integration tests (using `pytest`) verifying vector dimensions, similarity metric bounds ($[0,1]$), scraper output integrity, and metric edge cases. |
| **D6** | **Technical & Analytical Report (`docs/REPORT.md` or PDF)** | A 4–6 page academic paper covering dataset collection methodology, web scraping ethics/rate-limiting, experimental results, noise sensitivity, demographic bias findings, and security recommendations. |

---

## 🗂️ 3. Directory Format Comparison & Proposed Structure

### ⚠️ Current Workspace Structure (Flat & Cluttered)

The existing project structure suffers from several anti-patterns:
- Mixes dataset archives, output images, root-level scripts (`main.py`, `analysis.py`), and documentation in a single flat directory.
- Lacks web scraper / dataset construction code.
- No separation between raw input data, feature code, metrics evaluation, and generated figures.

```
Current Structure (Anti-pattern):
├── COMPLETION_CHECKLIST.txt
├── DATASET_GUIDE.md
├── DELIVERY_SUMMARY.txt
├── DOWNLOAD_GUIDE.txt
├── DOWNLOAD_OPTIONS.txt
├── README.md
├── analysis.py                  <-- Monolithic 2000-line script
├── dataset_analysis.png         <-- Visual output dumped in root
├── evaluation_results.json      <-- Results dumped in root
├── evaluation_results.png       <-- Results dumped in root
├── face_dataset/                <-- Pre-existing dataset (NOT TO BE USED)
├── face_dataset_complete.tar.gz
├── main.py                      <-- 14-line temporary test script
├── metadata.json
└── statistics.json
```

---

### ✨ Proposed Modular Directory Structure (Industry & MLOps Standard)

Students **MUST** construct and refactor the project into the following modular architecture:

```
facial-biometrics-scraped-benchmark/
├── data/                               # Custom Scraped Dataset Storage (Git Ignored)
│   ├── raw/                            # Custom web-scraped image sets
│   │   ├── REAL/                       # Scraped pristine images (S0001_001.jpg, ...)
│   │   ├── GAUSSIAN/                   # Programmatically generated Gaussian noise images
│   │   └── SALT_PEPPER/                # Programmatically generated salt-and-pepper noise
│   └── metadata/                       # Scraped metadata & statistics
│       ├── metadata.json               # 200 subject demographic records
│       └── statistics.json             # Dataset statistics
│
├── src/                                # Core Modular Source Package
│   ├── __init__.py
│   ├── dataset_generator/              # Web Scraper & Synthetic Noise Generator
│   │   ├── __init__.py
│   │   ├── scraper.py                  # Web crawler & image download engine
│   │   ├── metadata_builder.py         # Demographic parser & age gap tagger
│   │   └── noise_injector.py           # Gaussian & Salt-and-Pepper noise applier
│   ├── preprocessing/                  # Data ingestion & image normalization
│   │   ├── __init__.py
│   │   ├── loader.py                   # Custom dataset loader
│   │   └── enhancer.py                 # CLAHE, resizing, normalization
│   ├── feature_extraction/             # Feature extractors
│   │   ├── __init__.py
│   │   ├── base.py                     # Feature extractor interface
│   │   ├── handcrafted.py              # Histograms & spatial moments
│   │   └── embeddings.py               # Deep feature extractors / descriptors
│   ├── matching/                       # Biometric matching & scoring engine
│   │   ├── __init__.py
│   │   ├── pair_generator.py           # Genuine & impostor pair sampling
│   │   └── similarity.py               # Cosine & Euclidean similarity functions
│   ├── evaluation/                     # Metric calculation logic
│   │   ├── __init__.py
│   │   ├── metrics.py                  # FMR, FNMR, EER, AUC calculations
│   │   └── fairness.py                 # Disaggregated demographic fairness metrics
│   └── visualization/                  # Plotting & Reporting engine
│       ├── __init__.py
│       ├── roc_plots.py                # ROC / DET curve generator
│       └── demographic_plots.py        # Fairness & score distribution charts
│
├── scripts/                            # CLI Scripts & Entrypoints
│   ├── scrape_dataset.py               # Dataset web scraper CLI entrypoint
│   ├── run_pipeline.py                 # Main evaluation pipeline entrypoint
│   └── evaluate_robustness.py          # Noise & corruption evaluation script
│
├── config/                             # Configuration Management
│   ├── default_config.yaml             # Hyperparameters, scraping rules, paths
│   └── logging.yaml                    # Logging layout configuration
│
├── outputs/                            # All Generated Results (Git Ignored)
│   ├── figures/                        # Publication-quality PNG/SVG plots
│   │   ├── roc_curves.png
│   │   ├── score_distributions.png
│   │   └── demographic_fairness.png
│   └── reports/                        # JSON / HTML report outputs
│       ├── evaluation_results.json
│       └── final_summary_report.html
│
├── tests/                              # Automated Test Suite
│   ├── test_scraper.py                 # Tests for image downloading & metadata generation
│   ├── test_preprocessing.py           # Tests for resizing & normalization
│   ├── test_features.py                # Tests for vector output shapes
│   ├── test_similarity.py              # Tests for cosine distance bounds
│   └── test_metrics.py                 # Tests for EER computation edge cases
│
├── docs/                               # Assignment Documentation & Student Report
│   ├── ASSIGNMENT_SPEC.md              # Assignment specification (this document)
│   └── REPORT.md                       # Student's final analytical report write-up
│
├── .gitignore                          # Clean git ignores for data/ & outputs/
├── requirements.txt                    # Exact Python dependency pinning
└── README.md                           # Quickstart & setup guide
```

---

### 💡 Key Benefits of the New Directory Format

1. **Integrated Web Scraping Pipeline:** Dedicated `src/dataset_generator/` isolates web crawling, metadata generation, and synthetic noise creation from model evaluation.
2. **Separation of Concerns:** Core algorithmic logic (`src/`) is isolated from execution scripts (`scripts/`), custom data (`data/`), and outputs (`outputs/`).
3. **Reproducibility & Scalability:** Hyperparameters, scraping URLs/rules, and configuration paths are centralized in `config/default_config.yaml`.
4. **Automated Testing Ready:** Dedicated `tests/` directory tests web scraping robustness, noise generation, and metric computations using `pytest`.
5. **Clean Workspace:** All scraped images, generated noise variants, and evaluation reports are cleanly contained in `data/` and `outputs/`.

---

## 📊 4. Evaluation Criteria & Grading Rubric

Submissions will be evaluated out of **100 Points** across the following dimensions:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                EVALUATION WEIGHTAGE                                   │
│                                                                                        │
│  [Web Scraping & Dataset Construction: 20%] │ [Code Architecture & Modular Layout: 20%]│
│  [Biometric Matching & Metrics: 20%]        │ [Robustness & Noise Analysis: 15%]       │
│  [Demographic Bias Assessment: 10%]         │ [Technical Report & Visuals: 15%]        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

| Criteria | Points | Requirements |
|----------|--------|--------------|
| **Web Scraping & Dataset Construction** | 20 | Successful automated scraping of $\ge 200$ subjects with $\ge 2$ images per subject, metadata tagging (`metadata.json`), and clean synthetic noise injection (`GAUSSIAN`, `SALT_PEPPER`). |
| **Code Architecture & Refactoring** | 20 | Strict adherence to the proposed modular `src/` directory layout, clean OOP abstractions, proper typing, docstrings, and config management. |
| **Biometric Performance Engine** | 20 | Correct implementation of genuine/impostor pairing, cosine similarity, FMR, FNMR, EER, and ROC-AUC metrics on custom scraped data. |
| **Noise Robustness Analysis** | 15 | Quantitative comparison of performance degradation across `REAL`, `GAUSSIAN`, and `SALT_PEPPER` variants with aging gap analysis. |
| **Demographic Fairness Assessment**| 10 | Disaggregated FMR/FNMR evaluation across gender and ethnicity groups; computation of Demographic Disparity Ratios (DDR). |
| **Technical Report & Visualizations** | 15 | Clear visual charts (`outputs/figures/`) and well-structured technical write-up in `docs/REPORT.md` covering web scraping methodology, ethics, rate-limiting, and biometric results. |

---

## 🚀 5. Getting Started & Execution Workflow

### Step 1: Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Build Custom Dataset via Web Scraping
```bash
# Scrape face images, build metadata.json, and generate GAUSSIAN + SALT_PEPPER variants
python scripts/scrape_dataset.py --subjects 200 --images-per-subject 2 --output-dir data/
```

### Step 3: Run Evaluation Pipeline
```bash
# Run evaluation on scraped clean images
python scripts/run_pipeline.py --variant REAL --config config/default_config.yaml

# Run noise robustness benchmark
python scripts/evaluate_robustness.py --all-variants
```

### Step 4: Run Automated Tests
```bash
pytest tests/
```
