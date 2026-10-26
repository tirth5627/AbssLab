"""
Populate and enhance the Word document report with measured experimental results,
tables, embedded plots, and in-depth discussions.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

doc_path = os.path.join(parent_dir, "PCA_SVD_LDA_ORL_Face_Dataset_Comparison_Report.docx")
out_completed_path = os.path.join(parent_dir, "PCA_SVD_LDA_ORL_Face_Dataset_Comparison_Report_Completed.docx")
tables_dir = os.path.join(parent_dir, "outputs", "tables")
figures_dir = os.path.join(parent_dir, "outputs", "figures")

# Load tables
table_b_df = pd.read_csv(os.path.join(tables_dir, "table_b_recognition_performance.csv"))
table_c_df = pd.read_csv(os.path.join(tables_dir, "table_c_reconstruction_compression.csv"))
table_d_df = pd.read_csv(os.path.join(tables_dir, "table_d_computational_efficiency.csv"))
best_summary_df = pd.read_csv(os.path.join(tables_dir, "best_configurations_summary.csv"))
summary_records_df = pd.read_csv(os.path.join(tables_dir, "summary_benchmark_records.csv"))

doc = docx.Document(doc_path)
print(f"Loaded document: {doc_path} ({len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables)")

# 1. Update Table 5 (Table B in text: Recognition performance)
# In original doc, Table 5 has header + 4 rows
tbl_b = doc.tables[5]
# Let's inspect rows
# Replace row 1: Raw pixels
r_raw = table_b_df[table_b_df["Method"].str.contains("Raw pixels")].iloc[0]
tbl_b.rows[1].cells[2].text = str(r_raw["Mean Accuracy (%)"])
tbl_b.rows[1].cells[3].text = str(r_raw["Std Dev (%)"])
tbl_b.rows[1].cells[4].text = str(r_raw["Macro-F1"])
tbl_b.rows[1].cells[5].text = str(r_raw["Fit Time (ms)"])

# For PCA, SVD, LDA: show the full sweep by clearing rows 2, 3, 4 and inserting all evaluated configurations
# Delete rows 2, 3, 4 from bottom up
for _ in range(3):
    tbl_b._tbl.remove(tbl_b.rows[-1]._tr)

# Now append all rows for PCA, SVD, LDA
for _, row in table_b_df.iterrows():
    if "Raw pixels" in row["Method"]:
        continue
    new_tr = tbl_b.add_row()
    new_tr.cells[0].text = str(row["Method"])
    new_tr.cells[1].text = str(row["Components"])
    new_tr.cells[2].text = str(row["Mean Accuracy (%)"])
    new_tr.cells[3].text = str(row["Std Dev (%)"])
    new_tr.cells[4].text = str(row["Macro-F1"])
    new_tr.cells[5].text = str(row["Fit Time (ms)"])

print("Table B updated with full component sweep.")

# 2. Update Table 6 (Table C: Reconstruction and compression)
tbl_c = doc.tables[6]
# Delete existing placeholder rows 1, 2, 3
for _ in range(3):
    tbl_c._tbl.remove(tbl_c.rows[-1]._tr)

# Append rows from table_c_df
for _, row in table_c_df.iterrows():
    new_tr = tbl_c.add_row()
    new_tr.cells[0].text = str(row["Method"])
    new_tr.cells[1].text = str(row["Components"])
    new_tr.cells[2].text = str(row["Relative Frobenius Error"]) if pd.notna(row["Relative Frobenius Error"]) else "N/A"
    new_tr.cells[3].text = str(row["Feature Storage / Image"])
    new_tr.cells[4].text = str(row["Notes"])

print("Table C updated with reconstruction metrics.")

# 3. Update Table 7 (Table D: Efficiency)
tbl_d = doc.tables[7]
# Delete existing placeholder rows 1, 2, 3
for _ in range(3):
    tbl_d._tbl.remove(tbl_d.rows[-1]._tr)

# Append rows from table_d_df
for _, row in table_d_df.iterrows():
    new_tr = tbl_d.add_row()
    new_tr.cells[0].text = str(row["Method"])
    new_tr.cells[1].text = str(row["Fit Time (ms)"])
    new_tr.cells[2].text = str(row["Transform Time / Image (µs)"])
    new_tr.cells[3].text = str(row["Model Parameter Storage"])

print("Table D updated with computational efficiency metrics.")

# 4. Append comprehensive results, embedded figures, and in-depth analytical discussion
doc.add_page_break()

# Heading: Experimental Findings & In-Depth Discussion
h1 = doc.add_heading("Experimental Results & In-Depth Discussion", level=1)
h1.style.font.name = "Arial"

p_intro = doc.add_paragraph()
p_intro.add_run(
    "All planned experiments were executed using 10 randomized stratified splits (5 training images, 5 test images per subject, "
    "totaling 200 train and 200 test images per split) across the 40 identities in the AT&T / ORL face database. "
    "Every evaluation used strictly isolated train-fit transformations to ensure zero data leakage. Below is the detailed analysis "
    "accompanied by the measured publication-grade visualizations."
)

# Plot 1
doc.add_heading("1. Recognition Accuracy vs. Number of Retained Components", level=2)
p_p1 = doc.add_paragraph()
p_p1.add_run(
    "Plot 1 compares identity classification accuracy (mean ± 1 std dev) for PCA, Truncated SVD, and LDA against the raw pixel baseline (95.00% ± 1.16%)."
)
p1_img_path = os.path.join(figures_dir, "plot1_accuracy_vs_components.png")
if os.path.exists(p1_img_path):
    doc.add_picture(p1_img_path, width=Inches(6.2))
    cap1 = doc.add_paragraph("Figure 1: Identification accuracy (%) vs number of components across 10 stratified splits.")
    cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap1.style.font.italic = True

p_disc1 = doc.add_paragraph()
p_disc1.add_run(
    "Key Findings & Observation on PCA vs. SVD Equivalence:\n"
    "• Why PCA and Truncated SVD Are the Same Everywhere: Across all evaluated metrics—recognition accuracy (80.35% vs 80.40% at k=5, 94.55% vs 94.60% at k=39), "
    "Linear SVM classification (95.80% for both), reconstruction MSE (0.012803 at k=5, 0.007421 at k=30 for both), 2D cluster topology, template storage (156 B/face), "
    "and test projection latency (~13 µs)—PCA and SVD perform virtually identically. This equivalence is governed by four core mathematical principles:\n"
    "  1. Algebraic Spectral Equivalence on Centered Data: For mean-centered matrix X_c, sample covariance is S = (1/(N-1)) X_c^T X_c. SVD factorizes X_c = U Σ V^T. "
    "Substituting yields S = V [Σ²/(N-1)] V^T. Because V is orthonormal, the right singular vectors V of SVD are mathematically identical to the eigenvectors of PCA covariance, with λ_j = σ_j²/(N-1).\n"
    "  2. DC Shift Cancellation in 1-NN: Uncentered SVD's leading singular vector v₁ absorbs the non-zero grand mean face illumination (σ₁ ≈ 660.06). In 1-NN Euclidean distance, "
    "(z_a - z_b)₁ ≈ (x_a - x_b)^T v₁ ≈ 0 cancels out in vector differences, making distance metrics isometric between PCA and SVD subspaces and yielding identical classification decisions.\n"
    "  3. Eckart-Young-Mirsky Theorem: Both methods find the globally optimal rank-k subspace minimizing Frobenius reconstruction error ||X - X̂||_F.\n"
    "  4. Sirovich-Kirby Snapshot Duality: Both solve the 200×200 Gram matrix K = X X^T in O(N² D) time and project test faces via a single matrix-vector multiply in ~13 µs.\n"
    "• Superiority of Supervised LDA: LDA achieves higher accuracy at every component count (83.15% at k=5; 92.60% at k=10; 95.20% at k=39). "
    "By explicitly maximizing between-class scatter over within-class scatter, LDA projects along directions that isolate individual identities.\n"
    "• Compression without Loss: At k=39, LDA attains 95.20% ± 1.69% accuracy, surpassing the uncompressed raw pixel baseline (95.00%) "
    "while achieving a 264.2x reduction in feature dimensionality (from 10,304 down to 39 features)."
)

# Plot 2 & Plot 3
doc.add_heading("2. Image Reconstruction & Reconstruction Error Analysis", level=2)
p_p2 = doc.add_paragraph()
p_p2.add_run(
    "Figure 2 illustrates low-rank face reconstructions for test faces under PCA and Truncated SVD across k = 5, 10, 20, and 30 components. "
    "Figure 3 presents the quantitative reconstruction Mean Squared Error (MSE) and Relative Frobenius Error."
)
p2_img_path = os.path.join(figures_dir, "plot2_face_reconstructions.png")
if os.path.exists(p2_img_path):
    doc.add_picture(p2_img_path, width=Inches(6.2))
    cap2 = doc.add_paragraph("Figure 2: Visual reconstruction of held-out test faces (Subjects 1 & 10) for PCA and Truncated SVD.")
    cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap2.style.font.italic = True

p3_img_path = os.path.join(figures_dir, "plot3_reconstruction_error.png")
if os.path.exists(p3_img_path):
    doc.add_picture(p3_img_path, width=Inches(6.2))
    cap3 = doc.add_paragraph("Figure 3: Quantitative test reconstruction error (MSE and Relative Frobenius Error %) vs component count.")
    cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap3.style.font.italic = True

p_disc2 = doc.add_paragraph()
p_disc2.add_run(
    "Key Findings:\n"
    "• Visual Progression: At k=5, reconstructed faces capture global lighting and head silhouette. By k=20 and k=30, fine facial features "
    "(eyes, nose, mouth, glasses) become clearly resolved.\n"
    "• Quantitative Error Decay: Reconstruction MSE steadily decreases from 0.0128 (k=5) to 0.0070 (k=39), with relative Frobenius error dropping from 23.45% to 17.26%.\n"
    "• Reconstruction vs. Recognition Trade-off: PCA and SVD minimize pixel reconstruction error by design (Eckart-Young-Mirsky theorem), "
    "whereas LDA directions do not optimize image reconstruction, focusing solely on inter-class separation."
)

# Plot 4 & Plot 5
doc.add_heading("3. Cumulative Explained Variance: Truncated SVD vs. PCA Comparison", level=2)
p4_img_path = os.path.join(figures_dir, "plot4_cumulative_variance_svd_vs_pca.png")
if not os.path.exists(p4_img_path):
    p4_img_path = os.path.join(figures_dir, "plot4_explained_variance.png")
if os.path.exists(p4_img_path):
    doc.add_picture(p4_img_path, width=Inches(6.2))
    cap4 = doc.add_paragraph("Figure 4: Cumulative explained variance comparison between PCA (centered) and Truncated SVD (uncentered) across all 200 components.")
    cap4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap4.style.font.italic = True

p_disc3 = doc.add_paragraph()
p_disc3.add_run(
    "Key Findings:\n"
    "• Initial Disparity: At k=5, PCA achieves 48.36% cumulative variance while Truncated SVD captures 46.64% (a +1.73% PCA lead). "
    "This occurs because uncentered SVD allocates its leading singular vector (σ₁ ≈ 660.06) to absorb the non-zero DC illumination mean face. "
    "In contrast, PCA centers the data first, allowing PC1 to immediately capture genuine facial structural variation.\n"
    "• Asymptotic Convergence: Beyond k=20, the variance gap shrinks below 0.3%, and both models reach the 90% threshold at k=75 and 95% at k=114/115.\n"
    "• Terminal Rank Bounds: Centering removes 1 degree of freedom, so PCA reaches 100.00% variance strictly at k=199 (N_train - 1). "
    "Truncated SVD retains full data rank and reaches 100.00% variance at k=200 (N_train)."
)

# Plot 8: Eigenspectra and Dimensionality Limits
doc.add_heading("4. Decomposition Spectra & Total Possible Subspace Components", level=2)
p8_img_path = os.path.join(figures_dir, "plot8_spectra_pca_svd_lda.png")
if os.path.exists(p8_img_path):
    doc.add_picture(p8_img_path, width=Inches(6.2))
    cap8 = doc.add_paragraph("Figure 5: Decomposition spectra: (A) Sorted PCA eigenvalues λ_i with log scree plot; (B) Sorted SVD singular values σ_i; (C) LDA Fisher discriminant ratios η_i.")
    cap8.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap8.style.font.italic = True

p_disc8 = doc.add_paragraph()
p_disc8.add_run(
    "Theoretical and Empirical Subspace Dimensionality Bounds:\n"
    "• PCA Maximum Components: k_max^{PCA} = min(N_train - 1, D) = min(200 - 1, 10,304) = 199 components. "
    "Because the sample covariance matrix C = (1/(N-1)) X_c^T X_c is computed from centered data, the rows of X_c sum to zero, removing exactly one degree of freedom. "
    "On the full ORL dataset (400 images), the theoretical bound is 399 components.\n"
    "• Truncated SVD Maximum Components: k_max^{SVD} = min(N_train, D) = min(200, 10,304) = 200 components. "
    "Uncentered SVD operates directly on the raw data matrix X of rank min(200, 10,304) = 200 without mean-centering constraint. "
    "On the full ORL dataset, the bound is 400 components.\n"
    "• LDA Maximum Components: k_max^{LDA} = min(C - 1, D) = min(40 - 1, 10,304) = 39 components. "
    "The between-class scatter matrix S_B is constructed from 40 class centroids minus the global mean, constraining rank(S_B) <= C - 1 = 39. "
    "All remaining dimensions reside in the null space of between-class variance."
)

# Plot 5: 2D Subspace Embeddings
doc.add_heading("5. 2D Subspace Embeddings", level=2)
p5_img_path = os.path.join(figures_dir, "plot5_2d_embeddings.png")
if os.path.exists(p5_img_path):
    doc.add_picture(p5_img_path, width=Inches(6.2))
    cap5 = doc.add_paragraph("Figure 6: Two-dimensional subspace embeddings for PCA, Truncated SVD, and LDA across 8 subject identities.")
    cap5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap5.style.font.italic = True

p_disc5 = doc.add_paragraph()
p_disc5.add_run(
    "Key Findings:\n"
    "• Visual Separation in 2D: In PCA and SVD, clusters for different individuals overlap heavily because the top 2 components capture overall lighting and silhouette variation rather than identity. "
    "In contrast, LDA's 2D projection shows tightly grouped, well-separated clusters for each individual identity, validating Fisher's criterion."
)

# Plot 6 & Plot 7
doc.add_heading("6. Confusion Matrix Analysis & Computational Efficiency", level=2)
p6_img_path = os.path.join(figures_dir, "plot6_confusion_matrices.png")
if os.path.exists(p6_img_path):
    doc.add_picture(p6_img_path, width=Inches(6.2))
    cap6 = doc.add_paragraph("Figure 7: 40 × 40 Identity Confusion Matrices for best PCA, Truncated SVD, and LDA models.")
    cap6.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap6.style.font.italic = True

p7_img_path = os.path.join(figures_dir, "plot7_computational_cost.png")
if os.path.exists(p7_img_path):
    doc.add_picture(p7_img_path, width=Inches(6.2))
    cap7 = doc.add_paragraph("Figure 8: Computational benchmarks: Training Fit Time (ms) and Test Transform Time per Image (µs).")
    cap7.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap7.style.font.italic = True

p_disc4 = doc.add_paragraph()
p_disc4.add_run(
    "Key Findings:\n"
    "• Confusion Patterns: The 40×40 confusion matrices exhibit strong, concentrated diagonal density across all 40 classes. "
    "Off-diagonal errors occur predominantly between subjects with similar facial hair or glasses.\n"
    "• Efficiency Trade-Offs: PCA and Truncated SVD require only ~33-68 ms to fit on 200 training images, while LDA requires ~211 ms due to generalized "
    "eigen-decomposition across 40 classes. However, once fitted, projection time per test image is blazingly fast across all methods: "
    "PCA takes ~13.2 µs, Truncated SVD ~12.7 µs, and LDA ~22.9 µs per image, making all three approaches suitable for real-time biometric verification."
)

# Best Configurations Table
doc.add_heading("7. Summary of Best Configurations", level=2)
tbl_best = doc.add_table(rows=1, cols=6)
tbl_best.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr_cells = tbl_best.rows[0].cells
hdr_cells[0].text = "Method"
hdr_cells[1].text = "Best k"
hdr_cells[2].text = "Peak Accuracy"
hdr_cells[3].text = "Macro-F1"
hdr_cells[4].text = "Fit Time"
hdr_cells[5].text = "Compression"

for _, row in best_summary_df.iterrows():
    r_cells = tbl_best.add_row().cells
    r_cells[0].text = str(row["Method"])
    r_cells[1].text = str(row["Best Components (k)"])
    r_cells[2].text = str(row["Peak Mean Accuracy (%)"])
    r_cells[3].text = str(row["Macro-F1"])
    r_cells[4].text = str(row["Fit Time (ms)"])
    r_cells[5].text = str(row["Compression vs Raw"])

# Conclusion
doc.add_heading("8. Conclusions and Practical Guidelines", level=2)
p_conc = doc.add_paragraph()
p_conc.add_run(
    "1. When identity recognition is the sole objective, LDA (Fisherfaces) is the superior method, achieving higher accuracy "
    "at lower dimensionalities (95.20% at k=39) and outperforming raw pixels with 264x compression.\n"
    "2. PCA and Truncated SVD produce equivalent representations on centered dense face matrices. When uncentered, SVD's leading vector "
    "aligns with the illumination centroid, but subsequent components span equivalent identity subspace variations.\n"
    "3. Total possible components are rigorously bounded: PCA is restricted to N_train - 1 = 199 components due to centering, "
    "Truncated SVD yields 200 components, and LDA is bounded by C - 1 = 39 components.\n"
    "4. PCA and SVD remain essential when image reconstruction, data compression, or unsupervised exploration is required, as LDA "
    "discards within-class variance that is critical for reconstruction but irrelevant for discrimination.\n"
    "5. In large-scale biometric applications, a hybrid PCA-LDA pipeline provides an optimal combination of dimensionality reduction, "
    "computational efficiency, and discriminative classification power."
)

# Save completed docx
doc.save(out_completed_path)
print(f"[OK] Completed document saved to: {out_completed_path}")

# Also overwrite original document
doc.save(doc_path)
print(f"[OK] Original document updated: {doc_path}")
