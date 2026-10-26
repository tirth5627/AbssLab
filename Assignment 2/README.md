# Facial Biometrics Scraped Benchmark

**Assignment 2 — Advanced Biometric Systems & Security**

End-to-end facial recognition evaluation framework with custom web-scraped dataset, noise robustness benchmarking, and demographic fairness analysis.

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd "Assignment 2"
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Build Custom Dataset via Web Scraping
```bash
python scripts/scrape_dataset.py --subjects 200 --images-per-subject 2 --output-dir data/
```

This will:
- Scrape 400 face images (200 subjects × 2 images each) from ThisPersonDoesNotExist.com
- Generate `data/metadata/metadata.json` with demographic attributes
- Create Gaussian and Salt-and-Pepper noise variants in `data/raw/GAUSSIAN/` and `data/raw/SALT_PEPPER/`

### 3. Run Evaluation Pipeline
```bash
# Evaluate with handcrafted features on clean images
python scripts/run_pipeline.py --variant REAL --feature-type handcrafted --export-report

# Evaluate with deep (FaceNet) features
python scripts/run_pipeline.py --variant REAL --feature-type deep --export-report

# Evaluate both feature types
python scripts/run_pipeline.py --variant REAL --feature-type both --export-report
```

### 4. Run Noise Robustness Benchmark
```bash
python scripts/evaluate_robustness.py --all-variants --feature-type handcrafted
python scripts/evaluate_robustness.py --all-variants --feature-type deep
```

### 5. Run Tests
```bash
pytest tests/ -v
```

## 📁 Project Structure

```
Assignment 2/
├── config/                         # Configuration files
│   ├── default_config.yaml         # Hyperparameters & paths
│   └── logging.yaml                # Logging configuration
├── data/                           # Scraped dataset (git-ignored)
│   ├── raw/REAL/                   # Clean scraped images
│   ├── raw/GAUSSIAN/               # Gaussian noise variants
│   ├── raw/SALT_PEPPER/            # Salt-and-pepper noise variants
│   └── metadata/                   # metadata.json & statistics.json
├── src/                            # Core source package
│   ├── dataset_generator/          # Web scraper & noise injector
│   ├── preprocessing/              # Image loading & enhancement
│   ├── feature_extraction/         # Handcrafted & deep features
│   ├── matching/                   # Pair generation & similarity
│   ├── evaluation/                 # FMR/FNMR/EER/AUC metrics
│   └── visualization/             # ROC/DET curves & fairness plots
├── scripts/                        # CLI entrypoints
│   ├── scrape_dataset.py           # Dataset construction CLI
│   ├── run_pipeline.py             # Evaluation pipeline CLI
│   └── evaluate_robustness.py      # Noise robustness CLI
├── tests/                          # pytest test suite
├── outputs/                        # Generated results (git-ignored)
│   ├── figures/                    # Publication-quality plots
│   └── reports/                    # JSON evaluation reports
├── docs/                           # Documentation
│   └── REPORT.md                   # Technical report
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Configuration

All hyperparameters are centralized in `config/default_config.yaml`:
- **Scraping**: Source URL, rate limiting, retry logic
- **Noise**: Gaussian sigma, salt-and-pepper ratio
- **Features**: Histogram bins, LBP parameters, deep model selection
- **Matching**: Number of impostor pairs, random seed
- **Evaluation**: Threshold sweep resolution, FMR operating points

## 📊 Output Artifacts

| Output | Location |
|--------|----------|
| ROC Curves | `outputs/figures/roc_*.png` |
| DET Curves | `outputs/figures/det_*.png` |
| Score Distributions | `outputs/figures/score_dist_*.png` |
| FMR/FNMR Curves | `outputs/figures/fmr_fnmr_*.png` |
| Fairness Charts | `outputs/figures/demographic_fairness_*.png` |
| DDR Charts | `outputs/figures/ddr_chart_*.png` |
| Noise Degradation | `outputs/figures/noise_degradation_*.png` |
| Evaluation JSON | `outputs/reports/evaluation_results.json` |
| Robustness JSON | `outputs/reports/robustness_results_*.json` |
