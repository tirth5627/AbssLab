import json
import os
from pathlib import Path

def generate_dashboard():
    # Load data
    reports_dir = Path("outputs/reports")
    figures_dir = Path("outputs/figures")
    
    with open(reports_dir / "evaluation_results.json", "r") as f:
        eval_data = json.load(f)
    
    with open(reports_dir / "robustness_results_deep.json", "r") as f:
        robustness_deep = json.load(f)
        
    with open(reports_dir / "robustness_results_handcrafted.json", "r") as f:
        robustness_hand = json.load(f)
        
    deep_eer = eval_data.get("REAL_deep", {}).get("metrics", {}).get("eer", 0.0) * 100
    hand_eer = eval_data.get("REAL_handcrafted", {}).get("metrics", {}).get("eer", 0.0) * 100
    deep_auc = eval_data.get("REAL_deep", {}).get("metrics", {}).get("auc_roc", 0.0)
    hand_auc = eval_data.get("REAL_handcrafted", {}).get("metrics", {}).get("auc_roc", 0.0)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Biometrics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --text-color: #f8fafc;
            --card-bg: rgba(30, 41, 59, 0.7);
            --primary-accent: #3b82f6;
            --secondary-accent: #8b5cf6;
            --success: #10b981;
            --danger: #ef4444;
            --glass-border: rgba(255, 255, 255, 0.1);
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
            background-attachment: fixed;
            min-height: 100vh;
        }}

        header {{
            padding: 2rem 5%;
            text-align: center;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--glass-border);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(to right, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .container {{
            padding: 2rem 5%;
            max-width: 1600px;
            margin: 0 auto;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}

        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
            border-color: rgba(255,255,255,0.2);
        }}

        .metric-title {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 0.5rem;
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
        }}

        .metric-value.good {{ color: var(--success); }}
        .metric-value.bad {{ color: var(--danger); }}
        .metric-value.neutral {{ color: #cbd5e1; }}

        .section-title {{
            font-size: 1.8rem;
            margin: 3rem 0 1.5rem;
            border-bottom: 2px solid var(--glass-border);
            padding-bottom: 0.5rem;
            display: inline-block;
        }}

        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }}

        .gallery-item {{
            background: var(--card-bg);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1rem;
            backdrop-filter: blur(12px);
            transition: all 0.3s ease;
            cursor: pointer;
        }}

        .gallery-item:hover {{
            transform: scale(1.02);
            border-color: var(--primary-accent);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
        }}

        .gallery-item img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}

        .gallery-item h3 {{
            font-size: 1.1rem;
            color: #e2e8f0;
            text-align: center;
        }}

        /* Modal styling for image expansion */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.9);
            backdrop-filter: blur(5px);
            align-items: center;
            justify-content: center;
        }}

        .modal.active {{
            display: flex;
            animation: fadeIn 0.3s ease;
        }}

        .modal-content {{
            max-width: 90%;
            max-height: 90vh;
            border-radius: 8px;
            box-shadow: 0 0 30px rgba(0,0,0,0.5);
            animation: zoomIn 0.3s ease;
        }}

        .close-btn {{
            position: absolute;
            top: 20px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            transition: 0.3s;
            cursor: pointer;
        }}

        .close-btn:hover,
        .close-btn:focus {{
            color: var(--danger);
            text-decoration: none;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        @keyframes zoomIn {{
            from {{ transform: scale(0.9); }}
            to {{ transform: scale(1); }}
        }}

        .tabs {{
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .tab-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--glass-border);
            color: var(--text-color);
            padding: 0.75rem 1.5rem;
            border-radius: 30px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }}

        .tab-btn:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}

        .tab-btn.active {{
            background: linear-gradient(135deg, var(--primary-accent), var(--secondary-accent));
            border-color: transparent;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        }}

        .tab-content {{
            display: none;
            animation: fadeIn 0.4s ease;
        }}

        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>

    <header>
        <h1>Facial Biometrics Benchmark Dashboard</h1>
        <p>Interactive evaluation results for Deep vs Handcrafted features</p>
    </header>

    <div class="container">
        
        <!-- High-level Metrics -->
        <h2 class="section-title">Performance Overview (Clean Data)</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">FaceNet (Deep) EER</div>
                <div class="metric-value good">{deep_eer:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Handcrafted EER</div>
                <div class="metric-value bad">{hand_eer:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">FaceNet AUC-ROC</div>
                <div class="metric-value good">{deep_auc:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Handcrafted AUC-ROC</div>
                <div class="metric-value bad">{hand_auc:.4f}</div>
            </div>
        </div>

        <!-- Tabs for Sections -->
        <div class="tabs">
            <button class="tab-btn active" onclick="openTab('deep')">FaceNet (Deep) Results</button>
            <button class="tab-btn" onclick="openTab('handcrafted')">Handcrafted Results</button>
            <button class="tab-btn" onclick="openTab('robustness')">Noise Robustness</button>
            <button class="tab-btn" onclick="openTab('fairness')">Demographic Fairness</button>
        </div>

        <!-- Deep Tab -->
        <div id="deep" class="tab-content active">
            <div class="gallery-grid">
                <div class="gallery-item" onclick="openModal('figures/roc_curve_REAL_deep.png')">
                    <img src="figures/roc_curve_REAL_deep.png" alt="ROC Deep">
                    <h3>ROC Curve (Deep)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/det_curve_REAL_deep.png')">
                    <img src="figures/det_curve_REAL_deep.png" alt="DET Deep">
                    <h3>DET Curve (Deep)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/score_distributions_REAL_deep.png')">
                    <img src="figures/score_distributions_REAL_deep.png" alt="Score Dist Deep">
                    <h3>Score Distribution (Deep)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/fmr_fnmr_REAL_deep.png')">
                    <img src="figures/fmr_fnmr_REAL_deep.png" alt="FMR FNMR Deep">
                    <h3>FMR / FNMR (Deep)</h3>
                </div>
            </div>
        </div>

        <!-- Handcrafted Tab -->
        <div id="handcrafted" class="tab-content">
            <div class="gallery-grid">
                <div class="gallery-item" onclick="openModal('figures/roc_curve_REAL_handcrafted.png')">
                    <img src="figures/roc_curve_REAL_handcrafted.png" alt="ROC Handcrafted">
                    <h3>ROC Curve (Handcrafted)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/det_curve_REAL_handcrafted.png')">
                    <img src="figures/det_curve_REAL_handcrafted.png" alt="DET Handcrafted">
                    <h3>DET Curve (Handcrafted)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/score_distributions_REAL_handcrafted.png')">
                    <img src="figures/score_distributions_REAL_handcrafted.png" alt="Score Dist Handcrafted">
                    <h3>Score Distribution (Handcrafted)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/fmr_fnmr_REAL_handcrafted.png')">
                    <img src="figures/fmr_fnmr_REAL_handcrafted.png" alt="FMR FNMR Handcrafted">
                    <h3>FMR / FNMR (Handcrafted)</h3>
                </div>
            </div>
        </div>

        <!-- Robustness Tab -->
        <div id="robustness" class="tab-content">
            <div class="gallery-grid">
                <div class="gallery-item" onclick="openModal('figures/roc_all_variants_deep.png')">
                    <img src="figures/roc_all_variants_deep.png" alt="ROC Deep Robustness">
                    <h3>ROC Across Noise (Deep)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/roc_all_variants_handcrafted.png')">
                    <img src="figures/roc_all_variants_handcrafted.png" alt="ROC Handcrafted Robustness">
                    <h3>ROC Across Noise (Handcrafted)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/noise_degradation_deep.png')">
                    <img src="figures/noise_degradation_deep.png" alt="Noise Deg Deep">
                    <h3>Noise Degradation (Deep)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/noise_degradation_handcrafted.png')">
                    <img src="figures/noise_degradation_handcrafted.png" alt="Noise Deg Handcrafted">
                    <h3>Noise Degradation (Handcrafted)</h3>
                </div>
            </div>
        </div>

        <!-- Fairness Tab -->
        <div id="fairness" class="tab-content">
            <div class="gallery-grid">
                <div class="gallery-item" onclick="openModal('figures/demographic_fairness_REAL_deep.png')">
                    <img src="figures/demographic_fairness_REAL_deep.png" alt="Fairness Deep">
                    <h3>Demographic Fairness (Deep)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/ddr_chart_REAL_deep.png')">
                    <img src="figures/ddr_chart_REAL_deep.png" alt="DDR Deep">
                    <h3>Disparity Ratio (DDR) (Deep)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/demographic_fairness_REAL_handcrafted.png')">
                    <img src="figures/demographic_fairness_REAL_handcrafted.png" alt="Fairness Handcrafted">
                    <h3>Demographic Fairness (Handcrafted)</h3>
                </div>
                <div class="gallery-item" onclick="openModal('figures/ddr_chart_REAL_handcrafted.png')">
                    <img src="figures/ddr_chart_REAL_handcrafted.png" alt="DDR Handcrafted">
                    <h3>Disparity Ratio (DDR) (Handcrafted)</h3>
                </div>
            </div>
        </div>

    </div>

    <!-- The Modal -->
    <div id="imageModal" class="modal" onclick="closeModal(event)">
        <span class="close-btn">&times;</span>
        <img class="modal-content" id="expandedImg">
    </div>

    <script>
        function openTab(tabName) {{
            // Hide all tab contents
            const contents = document.getElementsByClassName("tab-content");
            for (let i = 0; i < contents.length; i++) {{
                contents[i].classList.remove("active");
            }}

            // Remove active class from buttons
            const buttons = document.getElementsByClassName("tab-btn");
            for (let i = 0; i < buttons.length; i++) {{
                buttons[i].classList.remove("active");
            }}

            // Show selected tab
            document.getElementById(tabName).classList.add("active");
            
            // Set active button
            event.currentTarget.classList.add("active");
        }}

        function openModal(imgSrc) {{
            const modal = document.getElementById("imageModal");
            const img = document.getElementById("expandedImg");
            img.src = imgSrc;
            modal.classList.add("active");
            document.body.style.overflow = "hidden"; // Prevent scrolling
        }}

        function closeModal(event) {{
            const modal = document.getElementById("imageModal");
            if (event.target === modal || event.target.className === "close-btn") {{
                modal.classList.remove("active");
                document.body.style.overflow = "auto";
            }}
        }}

        // Add keyboard support for closing modal
        document.addEventListener('keydown', function(event) {{
            if (event.key === "Escape") {{
                const modal = document.getElementById("imageModal");
                if (modal.classList.contains("active")) {{
                    modal.classList.remove("active");
                    document.body.style.overflow = "auto";
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    with open("outputs/dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print("Dashboard generated at outputs/dashboard.html")

if __name__ == "__main__":
    generate_dashboard()
