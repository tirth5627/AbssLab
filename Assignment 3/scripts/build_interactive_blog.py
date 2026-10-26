"""
Build a stunning, interactive publication-grade HTML blog for Assignment 3:
PCA vs Truncated SVD vs LDA on the AT&T / ORL Face Database.
"""

import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

json_path = os.path.join(parent_dir, "outputs", "tables", "benchmark_data.json")
out_html_path = os.path.join(parent_dir, "index.html")
out_blog_path = os.path.join(parent_dir, "blog.html")

with open(json_path, "r", encoding="utf-8") as f:
    json_raw = f.read()

template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Eigenfaces vs. Fisherfaces: The Geometry of Biometric Identity</title>
  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap" rel="stylesheet">

  <!-- KaTeX for High-Speed Mathematical Typesetting -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>

  <!-- MathJax 3 as High-Fidelity TeX Renderer & Fallback -->
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }
    };
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    :root {
      --bg-primary: #0A0D14;
      --bg-secondary: #101623;
      --bg-card: rgba(18, 25, 38, 0.75);
      --bg-card-hover: rgba(26, 36, 54, 0.85);
      --border-color: rgba(255, 255, 255, 0.08);
      --border-highlight: rgba(99, 102, 241, 0.35);
      
      --text-primary: #F8FAFC;
      --text-secondary: #94A3B8;
      --text-muted: #64748B;
      
      --accent-indigo: #6366F1;
      --accent-cyan: #06B6D4;
      --accent-rose: #F43F5E;
      --accent-emerald: #10B981;
      --accent-amber: #F59E0B;
      --accent-purple: #8B5CF6;
      
      --gradient-primary: linear-gradient(135deg, #6366F1 0%, #06B6D4 100%);
      --gradient-accent: linear-gradient(135deg, #F43F5E 0%, #F59E0B 100%);
      
      --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
      --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
      --shadow-glow: 0 0 35px rgba(99, 102, 241, 0.25);
      
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --radius-full: 9999px;
    }

    [data-theme="light"] {
      --bg-primary: #F8FAFC;
      --bg-secondary: #EDF2F7;
      --bg-card: rgba(255, 255, 255, 0.85);
      --bg-card-hover: rgba(255, 255, 255, 0.95);
      --border-color: rgba(0, 0, 0, 0.08);
      --border-highlight: rgba(99, 102, 241, 0.4);
      --text-primary: #0F172A;
      --text-secondary: #475569;
      --text-muted: #94A3B8;
      --shadow-glow: 0 0 35px rgba(99, 102, 241, 0.15);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      scroll-behavior: smooth;
    }

    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.7;
      overflow-x: hidden;
      transition: background-color 0.3s ease, color 0.3s ease;
    }

    #progress-container {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 4px;
      background: transparent;
      z-index: 1000;
    }
    #progress-bar {
      height: 100%;
      width: 0%;
      background: var(--gradient-primary);
      box-shadow: 0 0 10px var(--accent-cyan);
      transition: width 0.1s ease;
    }

    header.site-header {
      position: sticky;
      top: 0;
      background: rgba(10, 13, 20, 0.8);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-color);
      z-index: 999;
      padding: 14px 28px;
    }
    .nav-container {
      max-width: 1280px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .nav-brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 800;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }
    .brand-badge {
      background: var(--gradient-primary);
      color: white;
      padding: 3px 9px;
      border-radius: var(--radius-sm);
      font-size: 0.72rem;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.05em;
    }
    .nav-links {
      display: flex;
      gap: 20px;
      align-items: center;
      list-style: none;
    }
    .nav-links a {
      color: var(--text-secondary);
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 500;
      transition: color 0.2s ease;
    }
    .nav-links a:hover {
      color: var(--accent-cyan);
    }
    .theme-toggle-btn {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      padding: 8px 14px;
      border-radius: var(--radius-full);
      cursor: pointer;
      font-size: 0.85rem;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s ease;
    }
    .theme-toggle-btn:hover {
      border-color: var(--accent-indigo);
    }

    .container {
      max-width: 1180px;
      margin: 0 auto;
      padding: 40px 24px;
    }

    .hero {
      padding: 60px 0 40px;
      text-align: center;
      position: relative;
    }
    .hero-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 16px;
      border-radius: var(--radius-full);
      background: rgba(99, 102, 241, 0.12);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: var(--accent-indigo);
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 24px;
    }
    .hero h1 {
      font-family: 'Playfair Display', serif;
      font-size: clamp(2.4rem, 5.2vw, 4.1rem);
      font-weight: 700;
      line-height: 1.15;
      letter-spacing: -0.02em;
      margin-bottom: 20px;
      background: linear-gradient(135deg, #FFFFFF 20%, #94A3B8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    [data-theme="light"] .hero h1 {
      background: linear-gradient(135deg, #0F172A 20%, #475569 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
      font-size: 1.15rem;
      color: var(--text-secondary);
      max-width: 820px;
      margin: 0 auto 36px;
      font-weight: 400;
    }
    .author-meta {
      display: flex;
      justify-content: center;
      gap: 24px;
      font-size: 0.88rem;
      color: var(--text-muted);
      margin-bottom: 40px;
      flex-wrap: wrap;
    }
    .author-meta span {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 40px 0 60px;
    }
    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 22px;
      backdrop-filter: blur(10px);
      box-shadow: var(--shadow-sm);
      transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
      text-align: left;
      position: relative;
      overflow: hidden;
    }
    .stat-card:hover {
      transform: translateY(-4px);
      border-color: var(--border-highlight);
      box-shadow: var(--shadow-glow);
    }
    .stat-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
      background: var(--accent-indigo);
    }
    .stat-card.emerald::before { background: var(--accent-emerald); }
    .stat-card.cyan::before { background: var(--accent-cyan); }
    .stat-card.rose::before { background: var(--accent-rose); }
    
    .stat-label {
      font-size: 0.8rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }
    .stat-val {
      font-size: 2.1rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      line-height: 1.1;
      margin-bottom: 4px;
    }
    .stat-desc {
      font-size: 0.84rem;
      color: var(--text-secondary);
    }

    .widget-container {
      background: var(--bg-card);
      border: 1px solid var(--border-highlight);
      border-radius: var(--radius-lg);
      padding: 36px;
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow-lg);
      margin: 40px 0 70px;
      position: relative;
    }
    .widget-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 16px;
    }
    .widget-title {
      font-size: 1.35rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .slider-control-box {
      margin: 20px 0 30px;
    }
    .slider-label-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .k-badge {
      background: var(--gradient-primary);
      color: white;
      padding: 4px 14px;
      border-radius: var(--radius-full);
      font-weight: 800;
      font-size: 1.05rem;
    }
    input[type=range] {
      width: 100%;
      height: 8px;
      border-radius: 4px;
      background: #1E293B;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: var(--accent-cyan);
      cursor: pointer;
      box-shadow: 0 0 12px var(--accent-cyan);
      transition: transform 0.15s ease;
    }
    input[type=range]::-webkit-slider-thumb:hover {
      transform: scale(1.2);
    }
    .slider-ticks {
      display: flex;
      justify-content: space-between;
      margin-top: 8px;
      font-size: 0.78rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }

    .model-comparison-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
      margin-top: 24px;
    }
    .model-box {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 22px;
      transition: all 0.2s ease;
    }
    .model-box.active-winner {
      border-color: var(--accent-emerald);
      box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    .model-box-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .model-box-name {
      font-weight: 700;
      font-size: 1.05rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .model-tag {
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: var(--radius-sm);
      text-transform: uppercase;
      font-weight: 700;
    }
    .tag-unsupervised { background: rgba(99, 102, 241, 0.15); color: var(--accent-indigo); }
    .tag-supervised { background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); }

    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      font-size: 0.88rem;
    }
    .metric-row .metric-name {
      color: var(--text-muted);
    }
    .metric-row .metric-val {
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
    }
    .progress-track {
      width: 100%;
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 3px;
      margin: 6px 0 16px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      border-radius: 3px;
      transition: width 0.3s ease;
    }

    .chapter {
      margin: 70px 0;
      scroll-margin-top: 80px;
    }
    .chapter-num {
      font-size: 0.85rem;
      color: var(--accent-cyan);
      font-weight: 800;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .chapter h2 {
      font-family: 'Playfair Display', serif;
      font-size: 2.15rem;
      font-weight: 700;
      line-height: 1.25;
      letter-spacing: -0.01em;
      margin-bottom: 20px;
    }
    .chapter p {
      color: var(--text-secondary);
      font-size: 1.05rem;
      margin-bottom: 20px;
      line-height: 1.8;
    }
    .chapter p strong {
      color: var(--text-primary);
    }

    .callout {
      background: rgba(99, 102, 241, 0.06);
      border-left: 4px solid var(--accent-indigo);
      border-radius: 0 var(--radius-md) var(--radius-md) 0;
      padding: 20px 24px;
      margin: 28px 0;
    }
    .callout.emerald {
      background: rgba(16, 185, 129, 0.06);
      border-left-color: var(--accent-emerald);
    }
    .callout.rose {
      background: rgba(244, 63, 94, 0.06);
      border-left-color: var(--accent-rose);
    }
    .callout-title {
      font-weight: 700;
      font-size: 1rem;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .callout p {
      margin-bottom: 0;
      font-size: 0.95rem;
    }

    .math-card {
      background: rgba(13, 18, 31, 0.95);
      border: 1px solid var(--border-color);
      border-left: 4px solid var(--accent-cyan);
      border-radius: var(--radius-md);
      padding: 24px 28px;
      margin: 26px 0;
      box-shadow: var(--shadow-sm);
    }
    .math-card-item {
      margin-bottom: 22px;
      padding-bottom: 18px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .math-card-item:last-child {
      margin-bottom: 0;
      padding-bottom: 0;
      border-bottom: none;
    }
    .math-card-label {
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--accent-cyan);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }
    .math-display {
      font-size: 1.15rem;
      color: #E2E8F0;
      margin: 8px 0;
      overflow-x: auto;
      text-align: center;
    }
    .katex-display {
      margin: 8px 0 !important;
      overflow-x: auto;
      overflow-y: hidden;
    }
    .katex {
      font-size: 1.05em;
    }

    .figure-container {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 24px;
      margin: 36px 0;
      text-align: center;
      box-shadow: var(--shadow-sm);
      transition: border-color 0.25s ease;
    }
    .figure-container:hover {
      border-color: var(--border-highlight);
    }
    .figure-img {
      max-width: 100%;
      height: auto;
      border-radius: var(--radius-sm);
      box-shadow: var(--shadow-sm);
      transition: transform 0.2s ease;
    }
    .figure-caption {
      margin-top: 16px;
      font-size: 0.88rem;
      color: var(--text-muted);
      font-style: italic;
    }

    .rank-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
      gap: 20px;
      margin: 28px 0 36px;
    }
    .rank-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 24px;
      position: relative;
      transition: all 0.25s ease;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .rank-card:hover {
      border-color: var(--border-highlight);
      transform: translateY(-2px);
      box-shadow: var(--shadow-glow);
    }
    .rank-card.indigo { border-top: 4px solid var(--accent-indigo); }
    .rank-card.emerald { border-top: 4px solid var(--accent-emerald); }
    .rank-card.rose { border-top: 4px solid var(--accent-rose); }
    .rank-card.cyan { border-top: 4px solid var(--accent-cyan); }
    .rank-badge {
      display: inline-block;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      padding: 3px 9px;
      border-radius: var(--radius-sm);
      margin-bottom: 12px;
      font-weight: 700;
    }
    .rank-badge.indigo { background: rgba(99, 102, 241, 0.15); color: var(--accent-indigo); }
    .rank-badge.emerald { background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); }
    .rank-badge.rose { background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); }
    .rank-badge.cyan { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); }
    .rank-val {
      font-size: 2.3rem;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
      line-height: 1.1;
      margin-bottom: 6px;
    }
    .rank-formula {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-bottom: 14px;
    }
    .rank-desc {
      font-size: 0.92rem;
      color: var(--text-secondary);
      line-height: 1.6;
    }

    .tab-header {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 20px;
      overflow-x: auto;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      padding: 10px 18px;
      font-family: inherit;
      font-size: 0.92rem;
      font-weight: 600;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s ease;
    }
    .tab-btn:hover {
      color: var(--text-primary);
    }
    .tab-btn.active {
      color: var(--accent-cyan);
      border-bottom-color: var(--accent-cyan);
    }
    .tab-pane {
      display: none;
    }
    .tab-pane.active {
      display: block;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .table-wrapper {
      overflow-x: auto;
      margin: 28px 0;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.88rem;
    }
    th {
      background: rgba(255, 255, 255, 0.04);
      color: var(--text-primary);
      font-weight: 700;
      padding: 14px 18px;
      border-bottom: 1px solid var(--border-color);
    }
    td {
      padding: 12px 18px;
      border-bottom: 1px solid var(--border-color);
      color: var(--text-secondary);
    }
    tr:last-child td {
      border-bottom: none;
    }
    tr:hover td {
      background: rgba(255, 255, 255, 0.02);
      color: var(--text-primary);
    }

    .quiz-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 32px;
      margin: 40px 0;
    }
    .quiz-q {
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 18px;
    }
    .quiz-options {
      display: grid;
      gap: 10px;
    }
    .quiz-opt {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      padding: 14px 18px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-size: 0.94rem;
      text-align: left;
      color: var(--text-primary);
    }
    .quiz-opt:hover {
      border-color: var(--accent-indigo);
      background: rgba(99, 102, 241, 0.08);
    }
    .quiz-opt.correct {
      background: rgba(16, 185, 129, 0.15) !important;
      border-color: var(--accent-emerald) !important;
      color: #34D399 !important;
      font-weight: 700;
    }
    .quiz-opt.incorrect {
      background: rgba(244, 63, 94, 0.15) !important;
      border-color: var(--accent-rose) !important;
      color: #FB7185 !important;
    }
    .quiz-feedback {
      margin-top: 14px;
      font-size: 0.88rem;
      display: none;
      padding: 12px 16px;
      border-radius: var(--radius-sm);
    }

    footer.site-footer {
      background: var(--bg-secondary);
      border-top: 1px solid var(--border-color);
      padding: 60px 0 30px;
      margin-top: 80px;
      text-align: center;
    }
    .footer-links {
      display: flex;
      justify-content: center;
      gap: 24px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }
    .footer-links a {
      color: var(--text-secondary);
      text-decoration: none;
      font-size: 0.9rem;
      transition: color 0.2s ease;
    }
    .footer-links a:hover {
      color: var(--accent-cyan);
    }

    @media (max-width: 768px) {
      .nav-links { display: none; }
      .widget-container { padding: 20px; }
      .chapter h2 { font-size: 1.7rem; }
    }
  </style>
</head>
<body>

  <!-- Reading Progress -->
  <div id="progress-container"><div id="progress-bar"></div></div>

  <!-- Header -->
  <header class="site-header">
    <div class="nav-container">
      <div class="nav-brand">
        <span>AdvanceBiometricsLab</span>
        <span class="brand-badge">Assignment 3</span>
      </div>
      <ul class="nav-links">
        <li><a href="#playground">Playground</a></li>
        <li><a href="#theory">Theory</a></li>
        <li><a href="#accuracy">Accuracy</a></li>
        <li><a href="#equivalence">PCA vs SVD</a></li>
        <li><a href="#embeddings">2D Embeddings</a></li>
        <li><a href="#reconstruction">Reconstruction</a></li>
        <li><a href="#variance">Cumulative Variance</a></li>
        <li><a href="#spectra">Spectra & Limits</a></li>
        <li><a href="#efficiency">Efficiency</a></li>
        <li><a href="#quiz">Quiz</a></li>
      </ul>
      <button class="theme-toggle-btn" id="theme-toggle" aria-label="Toggle Theme">
        <span id="theme-icon">☀️</span> <span id="theme-text">Light Mode</span>
      </button>
    </div>
  </header>

  <main class="container">
    <!-- Hero Section -->
    <section class="hero">
      <div class="hero-pill">🔬 Advanced Facial Subspace Learning & Benchmark</div>
      <h1>Eigenfaces vs. Fisherfaces:<br>The Geometry of Biometric Identity</h1>
      <p class="hero-subtitle">
        An empirical and mathematical comparative study of Principal Component Analysis (PCA), Truncated SVD, and Linear Discriminant Analysis (LDA) on the AT&T / ORL Face Database.
      </p>
      
      <div class="author-meta">
        <span>👤 <strong>Khushkataruka</strong> (SVNIT)</span>
        <span>📁 <strong>400 Grayscale Faces</strong> (40 Subjects × 10 Images)</span>
        <span>🔄 <strong>10 Stratified Splits</strong> (Fixed Seeds)</span>
        <span>⚡ <strong>1-NN & Linear SVM Classifiers</strong></span>
      </div>

      <!-- Quick Metrics Grid -->
      <div class="stats-grid">
        <div class="stat-card emerald">
          <div class="stat-label">Peak Recognition Accuracy</div>
          <div class="stat-val" style="color: var(--accent-emerald);">95.20%</div>
          <div class="stat-desc">Achieved by <strong>LDA @ k=39</strong> components</div>
        </div>

        <div class="stat-card cyan">
          <div class="stat-label">Feature Compression</div>
          <div class="stat-val" style="color: var(--accent-cyan);">264.2×</div>
          <div class="stat-desc">Reduced 10,304 features down to 39</div>
        </div>

        <div class="stat-card">
          <div class="stat-label">PCA Max Rank (Centering)</div>
          <div class="stat-val" style="color: var(--accent-indigo);">199 Comps</div>
          <div class="stat-desc">Formula: <strong>$\min(N - 1, D) = 199$</strong></div>
        </div>

        <div class="stat-card emerald">
          <div class="stat-label">SVD Max Rank (Uncentered)</div>
          <div class="stat-val" style="color: var(--accent-emerald);">200 Comps</div>
          <div class="stat-desc">Formula: <strong>$\min(N, D) = 200$</strong></div>
        </div>

        <div class="stat-card rose">
          <div class="stat-label">LDA Max Rank (Classes)</div>
          <div class="stat-val" style="color: var(--accent-rose);">39 Comps</div>
          <div class="stat-desc">Formula: <strong>$\min(C - 1, D) = 39$</strong></div>
        </div>

        <div class="stat-card rose">
          <div class="stat-label">Projection Latency</div>
          <div class="stat-val" style="color: var(--accent-rose);">12.7 µs</div>
          <div class="stat-desc">Ultra-fast inference per single face</div>
        </div>
      </div>
    </section>

    <!-- Interactive Component Playground Widget -->
    <section class="widget-container" id="playground">
      <div class="widget-header">
        <div>
          <div class="widget-title">🎛️ Interactive Subspace Simulator</div>
          <div style="font-size: 0.88rem; color: var(--text-muted);">Dynamically inspect classification accuracy, reconstruction error, and storage footprint as components vary.</div>
        </div>
        <div class="k-badge" id="k-display-badge">k = 39 Components</div>
      </div>

      <div class="slider-control-box">
        <div class="slider-label-row">
          <span style="font-weight: 600; font-size: 0.95rem;">Retained Components (k):</span>
          <span style="font-family: 'JetBrains Mono', monospace; color: var(--accent-cyan); font-weight: 700;" id="k-indicator-val">39 / 150</span>
        </div>
        <input type="range" id="component-slider" min="0" max="8" value="4" step="1">
        <div class="slider-ticks">
          <span>5</span>
          <span>10</span>
          <span>20</span>
          <span>30</span>
          <span style="color: var(--accent-cyan); font-weight: bold;">39 (Max LDA)</span>
          <span>50</span>
          <span>75</span>
          <span>100</span>
          <span>150</span>
        </div>
      </div>

      <!-- Live Model Comparison Cards -->
      <div class="model-comparison-grid">
        <!-- PCA Card -->
        <div class="model-box" id="pca-card">
          <div class="model-box-header">
            <div class="model-box-name">
              <span style="color: var(--accent-indigo);">●</span> PCA (Eigenfaces)
            </div>
            <span class="model-tag tag-unsupervised">Unsupervised</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">1-NN Accuracy:</span>
            <span class="metric-val" id="pca-acc-val">94.55% ± 1.42%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" id="pca-acc-bar" style="width: 94.55%; background: var(--accent-indigo);"></div>
          </div>
          <div class="metric-row">
            <span class="metric-name">Macro-F1 Score:</span>
            <span class="metric-val" id="pca-f1-val">0.9434</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Reconstruction MSE:</span>
            <span class="metric-val" id="pca-mse-val">0.006957</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Feature Storage / Face:</span>
            <span class="metric-val" id="pca-storage-val">156 Bytes (0.15 KB)</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Subspace Limit:</span>
            <span class="metric-val" style="color: var(--accent-indigo);">199 Comps ($\min(N-1, D)$)</span>
          </div>
        </div>

        <!-- Truncated SVD Card -->
        <div class="model-box" id="svd-card">
          <div class="model-box-header">
            <div class="model-box-name">
              <span style="color: var(--accent-emerald);">●</span> Truncated SVD
            </div>
            <span class="model-tag tag-unsupervised">Unsupervised</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">1-NN Accuracy:</span>
            <span class="metric-val" id="svd-acc-val">94.60% ± 1.65%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" id="svd-acc-bar" style="width: 94.60%; background: var(--accent-emerald);"></div>
          </div>
          <div class="metric-row">
            <span class="metric-name">Macro-F1 Score:</span>
            <span class="metric-val" id="svd-f1-val">0.9440</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Reconstruction MSE:</span>
            <span class="metric-val" id="svd-mse-val">0.006944</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Feature Storage / Face:</span>
            <span class="metric-val" id="svd-storage-val">156 Bytes (0.15 KB)</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Subspace Limit:</span>
            <span class="metric-val" style="color: var(--accent-emerald);">200 Comps ($\min(N, D)$)</span>
          </div>
        </div>

        <!-- LDA Card -->
        <div class="model-box active-winner" id="lda-card">
          <div class="model-box-header">
            <div class="model-box-name">
              <span style="color: var(--accent-rose);">●</span> LDA (Fisherfaces)
            </div>
            <span class="model-tag tag-supervised">Supervised</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">1-NN Accuracy:</span>
            <span class="metric-val" id="lda-acc-val" style="color: var(--accent-emerald);">95.20% ± 1.69%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" id="lda-acc-bar" style="width: 95.20%; background: var(--accent-rose);"></div>
          </div>
          <div class="metric-row">
            <span class="metric-name">Macro-F1 Score:</span>
            <span class="metric-val" id="lda-f1-val">0.9499</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Reconstruction MSE:</span>
            <span class="metric-val" id="lda-mse-val" style="color: var(--text-muted);">N/A (Discriminative)</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Feature Storage / Face:</span>
            <span class="metric-val" id="lda-storage-val">156 Bytes (0.15 KB)</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Subspace Limit:</span>
            <span class="metric-val" style="color: var(--accent-rose);">39 Comps ($\min(C-1, D)$)</span>
          </div>
        </div>
      </div>
      <div style="margin-top: 18px; text-align: right; font-size: 0.8rem; color: var(--text-muted);">
        Baseline: Raw Pixels (10,304 dims) = <strong>95.00% ± 1.16%</strong> (41.2 KB / face)
      </div>
    </section>

    <!-- Chapter 1: The Core Research Question -->
    <section class="chapter" id="theory">
      <div class="chapter-num">Chapter 1</div>
      <h2>The Geometry of Faces: Maximum Variance vs. Class Separation</h2>
      <p>
        In computational face recognition, representing high-dimensional pixel spaces ($92 \times 112 = 10,304$ features) directly causes severe computational bottlenecks and susceptibility to pixel-level noise. Linear subspace learning methods find a projection matrix $W \in \mathbb{R}^{D \times k}$ ($k \ll D$) that captures the underlying identity structure. However, their foundational optimization criteria differ fundamentally:
      </p>

      <div class="math-card">
        <div class="math-card-item">
          <div class="math-card-label">1. Principal Component Analysis — Maximize Total Sample Variance (Unsupervised Eigenfaces)</div>
          <div class="math-display">
            $$ \max_{W} \operatorname{Tr}\left( W^T S_T W \right), \quad \text{where } S_T = \sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T $$
          </div>
        </div>
        <div class="math-card-item">
          <div class="math-card-label">2. Truncated SVD — Optimal Rank-$k$ Frobenius Approximation (Eckart-Young-Mirsky Theorem)</div>
          <div class="math-display">
            $$ \min_{\hat{X}, \operatorname{rank}(\hat{X}) \le k} \| X - \hat{X} \|_F, \quad \text{where } X \approx U_k \Sigma_k V_k^T $$
          </div>
        </div>
        <div class="math-card-item">
          <div class="math-card-label">3. Linear Discriminant Analysis — Maximize Between-to-Within Scatter (Supervised Fisherfaces)</div>
          <div class="math-display">
            $$ \max_{W} \frac{\det\left( W^T S_B W \right)}{\det\left( W^T S_W W \right)}, \quad \text{where } S_B = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T $$
          </div>
        </div>
      </div>

      <div class="callout emerald">
        <div class="callout-title">💡 The Fundamental Biometric Insight</div>
        <p>
          <strong>Total pixel variance does not equal identity discriminability.</strong> PCA eigenfaces capture directions of greatest contrast—which are overwhelmingly dominated by <em>ambient lighting, cast shadows, and background gradients</em>. In stark contrast, LDA uses ground-truth identity labels to mathematically compress intra-person variations (facial expression, smiling, eyewear) while stretching the distances between different individuals.
        </p>
      </div>
    </section>

    <!-- Chapter 2: Recognition Accuracy (Plot 1) -->
    <section class="chapter" id="accuracy">
      <div class="chapter-num">Chapter 2</div>
      <h2>Identification Accuracy vs. Subspace Dimensionality</h2>
      <p>
        We evaluated all methods using 1-Nearest Neighbor (1-NN, Euclidean distance) across 10 randomized stratified splits (seeds 42, 101, ..., 909) with 5 training and 5 testing images per subject.
      </p>

      <!-- Plot 1 Display -->
      <div class="figure-container">
        <img src="outputs/figures/plot1_accuracy_vs_components.png" alt="Plot 1: Identification Accuracy vs Number of Components" class="figure-img" id="fig-plot1">
        <div class="figure-caption">Figure 1: Mean identification accuracy (%) with ±1 standard deviation shaded error bands on the ORL Face Database.</div>
      </div>

      <div class="tab-header">
        <button class="tab-btn active" onclick="switchTab('obs-twin')">Observation 1: Subspace Twins</button>
        <button class="tab-btn" onclick="switchTab('obs-lda')">Observation 2: LDA Dominance</button>
        <button class="tab-btn" onclick="switchTab('obs-denoise')">Observation 3: Intelligent Denoising</button>
      </div>

      <div id="obs-twin" class="tab-pane active">
        <p>
          <strong>The Subspace Twin Phenomenon:</strong> Notice how the blue curve (PCA) and green curve (Truncated SVD) trace essentially identical trajectories across all component counts: at $k=5$, they achieve <strong>80.35% vs 80.40%</strong>; at $k=10$, <strong>91.35% vs 91.50%</strong>; at $k=20$, <strong>93.30% vs 93.35%</strong>; and at $k=39$, <strong>94.55% vs 94.60%</strong>. Both yield identical Linear SVM accuracy (<strong>95.80% ± 1.34%</strong>), identical reconstruction MSE ($0.012803$ at $k=5$, $0.007421$ at $k=30$), and identical inference latencies ($\sim 13\ \mu\text{s}$). As detailed in the mathematical breakdown below, SVD right singular vectors of centered data are identical to PCA covariance eigenvectors, and the DC offset in uncentered SVD cancels out in 1-NN Euclidean distance comparisons.
        </p>
      </div>

      <div id="obs-lda" class="tab-pane">
        <p>
          <strong>Why Supervised LDA Dominates Every Component Count:</strong> The red curve representing LDA sits strictly above PCA and SVD across the entire spectrum. Even under extreme low dimensions ($k=5$), LDA delivers 83.15% (+2.8% ahead of PCA). By $k=39$, LDA hits <strong>95.20% ± 1.69%</strong>, beating both unsupervised models.
        </p>
      </div>

      <div id="obs-denoise" class="tab-pane">
        <p>
          <strong>Compression as Denoising:</strong> Raw pixels (10,304 features) achieve 95.00% accuracy. LDA with only 39 components achieves 95.20%. By discarding 10,265 dimensions of high-frequency noise and non-discriminative background pixels, LDA functions as an intelligent denoising filter while achieving a <strong>264.2× compression ratio</strong>.
        </p>
      </div>

      <!-- Dedicated Deep-Dive: Why PCA and SVD Are the Same Everywhere -->
      <div class="callout indigo" id="equivalence" style="margin-top: 36px; padding: 26px 28px;">
        <div class="callout-title" style="font-size: 1.22rem; margin-bottom: 12px; color: var(--accent-indigo);">
          🔍 Observation: Why PCA and Truncated SVD Are the Same Everywhere
        </div>
        <p style="margin-bottom: 18px; font-size: 0.98rem; line-height: 1.8;">
          Across every dimension of our empirical investigation—<strong>1-NN identification accuracy</strong> (94.55% vs 94.60%), <strong>Linear SVM classification</strong> (exactly 95.80% for both), <strong>reconstruction MSE</strong> ($0.012803$ at $k=5$, $0.007421$ at $k=30$ for both), <strong>2D subspace cluster topology</strong>, <strong>inference latency</strong> ($\sim 13\ \mu\text{s}$ per face), and <strong>template memory</strong> (156 bytes)—<strong>PCA and Truncated SVD perform virtually identically</strong>. This is governed by four fundamental linear algebraic and biometric principles:
        </p>

        <div class="rank-grid" style="margin: 20px 0 16px;">
          <!-- Pillar 1 -->
          <div class="rank-card indigo">
            <div>
              <span class="rank-badge indigo">Pillar 1: Algebraic Equivalence</span>
              <div class="math-display" style="font-size: 1.05rem; margin-bottom: 8px;">
                $$ X_c^T X_c = V \Sigma^2 V^T = (N - 1) S $$
              </div>
              <div style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.65;">
                For mean-centered face matrix $X_c \in \mathbb{R}^{N \times D}$, its sample covariance matrix is $S = \frac{1}{N-1} X_c^T X_c$. SVD factorizes $X_c = U \Sigma V^T$. Expanding the covariance yields $S = \frac{1}{N-1} V \Sigma^T (U^T U) \Sigma V^T = V \left[\frac{\Sigma^2}{N-1}\right] V^T$. Because $V$ is orthonormal and $\frac{\Sigma^2}{N-1}$ is diagonal, this is the <em>exact spectral decomposition</em> of $S$. Thus, <strong>the right singular vectors $V$ of SVD are strictly identical to the eigenvectors of PCA covariance</strong>, and $\lambda_j = \frac{\sigma_j^2}{N-1}$. On centered data, PCA is literally SVD.
              </div>
            </div>
          </div>

          <!-- Pillar 2 -->
          <div class="rank-card emerald">
            <div>
              <span class="rank-badge emerald">Pillar 2: DC Shift Invariance in 1-NN</span>
              <div class="math-display" style="font-size: 1.05rem; margin-bottom: 8px;">
                $$ (z_a - z_b)_1 \approx (x_a - x_b)^T v_1 \approx 0 $$
              </div>
              <div style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.65;">
                Standard Truncated SVD operates on uncentered data $X = X_c + \mathbf{1} \mu^T$. Because pixel values are positive, the grand mean face $\mu$ has massive magnitude (carrying energy $\sigma_1 \approx 660.06$). SVD's 1st singular vector $v_1$ aligns with this DC illumination centroid. Vectors $v_2$ through $v_k$ span the exact same variance subspace as PCA's $v_1$ through $v_{k-1}$. In 1-NN classification using Euclidean distance $\|z_a - z_b\|_2$, <strong>the constant projection onto the mean face subtracts out in vector differences</strong>. Pairwise distances remain isometric, finding the exact same nearest neighbors everywhere.
              </div>
            </div>
          </div>

          <!-- Pillar 3 -->
          <div class="rank-card cyan">
            <div>
              <span class="rank-badge cyan">Pillar 3: Eckart-Young-Mirsky Theorem</span>
              <div class="math-display" style="font-size: 1.05rem; margin-bottom: 8px;">
                $$ \min_{\hat{X}, \operatorname{rank}(\hat{X}) \le k} \| X - \hat{X} \|_F $$
              </div>
              <div style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.65;">
                The Eckart-Young-Mirsky theorem proves that Truncated SVD provides the globally optimal rank-$k$ matrix approximation minimizing Frobenius error $\|X - \hat{X}\|_F$. PCA solves the identical problem for deviations around the empirical mean face. Because facial images reside on an intrinsic low-dimensional manifold governed by orthogonal geometric variations, <strong>both algorithms project onto identical principal hyperplanes</strong>, yielding identical test reconstruction MSE ($0.012803$ at $k=5$; $0.007421$ at $k=30$).
              </div>
            </div>
          </div>

          <!-- Pillar 4 -->
          <div class="rank-card rose">
            <div>
              <span class="rank-badge rose">Pillar 4: Snapshot Duality & Cost</span>
              <div class="math-display" style="font-size: 1.05rem; margin-bottom: 8px;">
                $$ \mathcal{O}(N^2 D + N^3) \implies \sim 13\ \mu\text{s Latency} $$
              </div>
              <div style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.65;">
                Both algorithms utilize the Sirovich & Kirby (1987) snapshot method: instead of constructing the intractable $10,304 \times 10,304$ covariance matrix, both diagonalize the $200 \times 200$ Gram matrix $K = X X^T$. At inference time, both project a face image via a single dense matrix-vector product $x^T W_k$ ($10,304 \times k$ FLOPs), resulting in identical $12.7\text{--}13.2\ \mu\text{s}$ projection latencies and identical 156-byte storage footprints per face template.
              </div>
            </div>
          </div>
        </div>

        <div style="margin-top: 14px; font-size: 0.92rem; color: var(--text-secondary); line-height: 1.7; background: rgba(255, 255, 255, 0.03); padding: 14px 18px; border-radius: var(--radius-sm); border-left: 3px solid var(--accent-indigo);">
          <strong>Summary Insight:</strong> PCA and Truncated SVD are two mathematical viewpoints of the same linear subspace. Whether formulated as covariance variance maximization (Hotelling 1933 / Turk & Pentland 1991) or matrix low-rank factorization (Beltrami 1873 / Eckart & Young 1936), they extract identical geometric bases. Consequently, they yield equivalent recognition accuracy, reconstruction fidelity, and computational efficiency across every benchmark.
        </div>
      </div>
    </section>

    <!-- Chapter 3: 2D Embeddings & Separation (Plot 5) -->
    <section class="chapter" id="embeddings">
      <div class="chapter-num">Chapter 3</div>
      <h2>Visualizing Class Separation: Two-Dimensional Embeddings</h2>
      <p>
        To understand <em>why</em> LDA outperforms PCA/SVD at low dimensions, we projected 8 distinct subject identities onto the first two components (k=2) of each subspace:
      </p>

      <div class="figure-container">
        <img src="outputs/figures/plot5_2d_embeddings.png" alt="Plot 5: 2D Subspace Embeddings across 8 Subject Identities" class="figure-img">
        <div class="figure-caption">Figure 2: 2D feature embeddings for PCA (Left), Truncated SVD (Center), and LDA (Right) across 8 subject identities.</div>
      </div>

      <p>
        In the PCA and SVD panels, points corresponding to different subjects blend together with heavy overlap. Because PCA and SVD are unsupervised, their first two components capture global illumination and head tilt rather than individual facial features. Conversely, in the LDA panel (Right), <strong>points belonging to the same individual coalesce into tight, isolated clusters with large margins between subjects</strong>.
      </p>
    </section>

    <!-- Chapter 4: Face Reconstructions (Plots 2 & 3) -->
    <section class="chapter" id="reconstruction">
      <div class="chapter-num">Chapter 4</div>
      <h2>Face Reconstruction: Image Synthesis vs. Biometric Discrimination</h2>
      <p>
        While LDA dominates classification, can it reconstruct the original face image? <strong>No.</strong> Subspace methods have completely divergent mathematical objectives:
      </p>

      <div class="figure-container">
        <img src="outputs/figures/plot2_face_reconstructions.png" alt="Plot 2: Original Test Faces vs Low-Rank Reconstructions" class="figure-img">
        <div class="figure-caption">Figure 3: Visual reconstruction of held-out test faces (Subject 1 and Subject 10) comparing PCA and SVD across k = 5, 10, 20, 30.</div>
      </div>

      <div class="figure-container">
        <img src="outputs/figures/plot3_reconstruction_error.png" alt="Plot 3: Test Image Reconstruction Error Curves" class="figure-img">
        <div class="figure-caption">Figure 4: Reconstruction Mean Squared Error (MSE) and Relative Frobenius Error (%) on held-out test faces as a function of k.</div>
      </div>

      <div class="callout rose">
        <div class="callout-title">⚠️ The Reconstruction vs. Discrimination Trade-Off</div>
        <p>
          PCA and SVD are mathematically guaranteed by the <strong>Eckart-Young-Mirsky Theorem</strong> to produce the optimal low-rank matrix approximation that minimizes Frobenius pixel error. As shown above, reconstruction MSE drops from 0.0128 (k=5) to 0.0070 (k=39), with fine features (eyes, nose, eyeglasses) resolving faithfully. LDA, however, projects along axes that explicitly <em>collapse</em> intra-class variance. Because intra-class variance is necessary to render individual expressions and textures, LDA cannot faithfully reconstruct images without a hybrid PCA basis.
        </p>
      </div>
    </section>

    <!-- Chapter 5: Cumulative Explained Variance (Plot 4) -->
    <section class="chapter" id="variance">
      <div class="chapter-num">Chapter 5</div>
      <h2>Cumulative Explained Variance: Truncated SVD vs. PCA Comparison</h2>
      <p>
        How rapidly do linear subspaces capture the physical energy of face images, and how does uncentered Truncated SVD compare against mean-centered PCA across varying component counts?
      </p>

      <div class="figure-container">
        <img src="outputs/figures/plot4_explained_variance.png" alt="Plot 4: Cumulative Explained Variance — Truncated SVD vs. PCA" class="figure-img" id="fig-plot4">
        <div class="figure-caption">Figure 5: Upper: Cumulative explained variance ratio (%) for PCA (blue) and Truncated SVD (green dashed) with 90% (k=75) and 95% (k=114/115) energy thresholds and LDA capacity mark (k=39). Lower: Absolute variance advantage Δ(k) = PCA − SVD.</div>
      </div>

      <div class="tab-header">
        <button class="tab-btn active" onclick="switchTab('var-comp')">Comparison: SVD vs. PCA</button>
        <button class="tab-btn" onclick="switchTab('var-thresholds')">Energy Thresholds (90% & 95%)</button>
        <button class="tab-btn" onclick="switchTab('var-paradox')">The Variance Paradox</button>
      </div>

      <div id="var-comp" class="tab-pane active">
        <p>
          <strong>Why PCA Leads at Low Components ($\Delta = +1.73\%$ at $k=5$):</strong>
          In standard uncentered Truncated SVD, the very first singular vector ($\sigma_1 \approx 660.06$) aligns directly with the grand mean face vector, absorbing the global DC illumination offset. Consequently, only singular vectors 2 through 5 capture genuine cross-sample variations, yielding <strong>46.64%</strong> cumulative variance at $k=5$. In contrast, PCA explicitly subtracts the empirical sample mean $\mu$ first, allowing its leading 5 eigenvectors to focus purely on centered variance, reaching <strong>48.36%</strong> (+1.73% advantage). As $k$ increases, this initial offset dilutes rapidly: by $k=20$, the difference drops to 0.34%, and by $k=39$, PCA (81.58%) and SVD (81.39%) differ by less than 0.19%.
        </p>
      </div>

      <div id="var-thresholds" class="tab-pane">
        <p>
          <strong>Exact Energy Retention Benchmarks:</strong>
          Both methods cross key biometric information thresholds at nearly identical component counts:
        </p>
        <ul style="margin-left: 20px; color: var(--text-secondary); line-height: 1.8;">
          <li><strong>90% Energy Threshold:</strong> PCA reaches 90.14% at <strong>$k = 75$</strong>; Truncated SVD reaches 90.03% at <strong>$k = 75$</strong>.</li>
          <li><strong>95% Energy Threshold:</strong> PCA reaches 95.03% at <strong>$k = 114$</strong>; Truncated SVD reaches 95.05% at <strong>$k = 115$</strong>.</li>
          <li><strong>100% Full Subspace Energy:</strong> PCA reaches 100.00% at its rank ceiling of <strong>$k = 199$</strong>; SVD reaches 100.00% at its uncentered rank ceiling of <strong>$k = 200$</strong>.</li>
        </ul>
      </div>

      <div id="var-paradox" class="tab-pane">
        <p>
          <strong>The Biometric Variance Paradox:</strong>
          At $k=39$, PCA and SVD retain ~81.5% of total dataset variance, while LDA is mathematically capped at $k=39$. Yet <strong>LDA at $k=39$ achieves 95.20% identification accuracy</strong>, outperforming PCA at $k=150$ (which retains >97.8% variance but reaches only 94.65% accuracy). This conclusively proves that retained variance measures signal reconstruction energy—not identity class separability.
        </p>
      </div>
    </section>

    <!-- Chapter 6: Sorted Spectra and Total Possible Components (Plot 8) -->
    <section class="chapter" id="spectra">
      <div class="chapter-num">Chapter 6</div>
      <h2>Eigenspectrum, Singular Value Spectrum & Dimensionality Limits</h2>
      <p>
        To examine the mathematical foundations of each subspace, we analyze the sorted eigenspectrum of PCA covariance, the singular value spectrum of Truncated SVD, and the Fisher discriminant ratios of LDA, together with their exact algebraic rank bounds.
      </p>

      <div class="figure-container">
        <img src="outputs/figures/plot8_spectra_pca_svd_lda.png" alt="Plot 8: Sorted Spectra across PCA, SVD, and LDA with Rank Bounds" class="figure-img" id="fig-plot8">
        <div class="figure-caption">Figure 6: Left: PCA sorted eigenvalues $\lambda_i$ with semi-log scree inset (Total 199 components). Center: Truncated SVD singular values $\sigma_i$ highlighting DC illumination peak $\sigma_1=660.1$ (Total 200 components). Right: LDA discriminant ratios $\eta_i$ with cumulative ratio curve (Total 39 components).</div>
      </div>

      <!-- Total Number of Possible Components Cards -->
      <h3 style="margin: 35px 0 16px; font-size: 1.4rem;">Total Number of Possible Components: Mathematical Rank Derivation</h3>
      <p>
        How many non-zero components can PCA, Truncated SVD, and LDA physically generate on the ORL Face Database? The cards below detail the exact linear algebraic limits:
      </p>

      <div class="rank-grid">
        <!-- PCA Card -->
        <div class="rank-card indigo">
          <div>
            <span class="rank-badge indigo">PCA (Eigenfaces)</span>
            <div class="rank-val" style="color: var(--accent-indigo);">199</div>
            <div class="rank-formula">$$k_{\max}^{\text{PCA}} = \min(N_{\text{train}} - 1, D) = 200 - 1 = \mathbf{199}$$</div>
            <div class="rank-desc">
              <strong>Centering Constraint:</strong> Training data consists of $N = 200$ face images. Subtracting the sample mean vector imposes the linear constraint $\sum_{i=1}^N (x_i - \mu) = 0$, consuming exactly 1 degree of freedom. Thus, the sample covariance matrix has rank at most <strong>199</strong>. All eigenvalues beyond $k = 199$ are identically zero ($\lambda_k = 0$).<br><br>
              <em>Full Dataset Limit ($400$ images):</em> $\min(400 - 1, 10,304) = \mathbf{399}\text{ components}$.
            </div>
          </div>
        </div>

        <!-- SVD Card -->
        <div class="rank-card emerald">
          <div>
            <span class="rank-badge emerald">Truncated SVD</span>
            <div class="rank-val" style="color: var(--accent-emerald);">200</div>
            <div class="rank-formula">$$k_{\max}^{\text{SVD}} = \min(N_{\text{train}}, D) = \min(200, 10,304) = \mathbf{200}$$</div>
            <div class="rank-desc">
              <strong>Uncentered Full Rank:</strong> Standard Truncated SVD factorizes the raw uncentered data matrix $X$ without mean subtraction. Because no centering constraint is applied, the matrix maintains its full algebraic rank of <strong>200</strong>.<br><br>
              The first singular value ($\sigma_1 = 660.06$) absorbs the global DC illumination offset ($\approx \sqrt{N}\|\mu\|$), leaving $199$ singular vectors for facial variation.<br><br>
              <em>Full Dataset Limit ($400$ images):</em> $\min(400, 10,304) = \mathbf{400}\text{ components}$.
            </div>
          </div>
        </div>

        <!-- LDA Card -->
        <div class="rank-card rose">
          <div>
            <span class="rank-badge rose">LDA (Fisherfaces)</span>
            <div class="rank-val" style="color: var(--accent-rose);">39</div>
            <div class="rank-formula">$$k_{\max}^{\text{LDA}} = \min(C - 1, D) = 40 - 1 = \mathbf{39}$$</div>
            <div class="rank-desc">
              <strong>Between-Class Scatter Bound:</strong> For $C = 40$ subject identities, the between-class scatter matrix $S_B = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T$ is constructed from the $40$ class centroid vectors that sum to zero relative to the global mean. Hence, $\operatorname{rank}(S_B) \le C - 1 = \mathbf{39}$.<br><br>
              It is <em>mathematically impossible</em> to extract a 40th discriminant component regardless of sample size or pixel resolution.<br><br>
              <em>Full Dataset Limit:</em> Always $\mathbf{39}\text{ components}$ (strictly independent of $N$).
            </div>
          </div>
        </div>
      </div>

      <div class="callout emerald">
        <div class="callout-title">📌 Key Takeaway for Biometrics Engineers</div>
        <p>
          While PCA and SVD can scale up to 199 and 200 components on the training set (capturing 100% of variance), LDA achieves its peak 95.20% accuracy at its theoretical maximum of <strong>only 39 components</strong>. This makes LDA not only superior in accuracy, but dramatically more compact, requiring only 156 bytes per face template compared to 800 bytes for full-rank PCA/SVD.
        </p>
      </div>
    </section>

    <!-- Chapter 7: Confusion Matrices & Computational Efficiency (Plots 6 & 7) -->
    <section class="chapter" id="efficiency">
      <div class="chapter-num">Chapter 7</div>
      <h2>Confusion Patterns & Real-Time Computational Efficiency</h2>
      
      <div class="figure-container">
        <img src="outputs/figures/plot6_confusion_matrices.png" alt="Plot 6: 40x40 Confusion Matrices" class="figure-img">
        <div class="figure-caption">Figure 6: 40 × 40 Identity Confusion Matrices for best PCA, Truncated SVD, and LDA configurations.</div>
      </div>

      <div class="figure-container">
        <img src="outputs/figures/plot7_computational_cost.png" alt="Plot 7: Computational Efficiency Benchmarks" class="figure-img">
        <div class="figure-caption">Figure 7: Training Fit Time (ms) and Test Transform Time per Image (µs) across PCA, SVD, and LDA.</div>
      </div>

      <div class="callout emerald">
        <div class="callout-title">⚡ Real-Time Embedded Viability</div>
        <p>
          Training PCA and SVD takes only 33–68 ms on 200 training images (10,304 features). LDA takes 211 ms due to between-class and within-class scatter eigenvalue solutions. However, once trained, <strong>transforming a test face into the reduced subspace takes only 12.7 µs (SVD), 13.2 µs (PCA), and 22.9 µs (LDA)</strong>. This enables processing over <strong>40,000 faces per second</strong> on standard CPU hardware!
        </p>
      </div>
    </section>

    <!-- Chapter 8: Data Tables -->
    <section class="chapter">
      <div class="chapter-num">Chapter 8</div>
      <h2>Comprehensive Experimental Data Tables</h2>
      
      <h3 style="margin: 20px 0 10px;">Table B: Recognition Performance (1-NN Classifier across 10 Splits)</h3>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Components (k)</th>
              <th>Mean Accuracy (%)</th>
              <th>Std Dev (%)</th>
              <th>Macro-F1</th>
              <th>Fit Time</th>
            </tr>
          </thead>
          <tbody>
            <tr><td><strong>Raw Pixels (Baseline)</strong></td><td>10,304</td><td><strong>95.00%</strong></td><td>±1.16%</td><td>0.9486 ± 0.0123</td><td>0.00 ms</td></tr>
            <tr><td>PCA + 1-NN</td><td>5</td><td>80.35%</td><td>±2.57%</td><td>0.7975 ± 0.0271</td><td>33.08 ms</td></tr>
            <tr><td>PCA + 1-NN</td><td>10</td><td>91.35%</td><td>±1.47%</td><td>0.9097 ± 0.0165</td><td>35.10 ms</td></tr>
            <tr><td>PCA + 1-NN</td><td>20</td><td>93.30%</td><td>±2.06%</td><td>0.9296 ± 0.0219</td><td>34.47 ms</td></tr>
            <tr><td>PCA + 1-NN</td><td>30</td><td>94.50%</td><td>±1.62%</td><td>0.9422 ± 0.0175</td><td>42.05 ms</td></tr>
            <tr><td>PCA + 1-NN</td><td>39</td><td>94.55%</td><td>±1.42%</td><td>0.9434 ± 0.0150</td><td>56.43 ms</td></tr>
            <tr><td>Truncated SVD + 1-NN</td><td>5</td><td>80.40%</td><td>±2.69%</td><td>0.7960 ± 0.0284</td><td>23.46 ms</td></tr>
            <tr><td>Truncated SVD + 1-NN</td><td>10</td><td>91.50%</td><td>±1.73%</td><td>0.9113 ± 0.0195</td><td>25.90 ms</td></tr>
            <tr><td>Truncated SVD + 1-NN</td><td>20</td><td>93.35%</td><td>±1.63%</td><td>0.9303 ± 0.0172</td><td>38.90 ms</td></tr>
            <tr><td>Truncated SVD + 1-NN</td><td>30</td><td>94.35%</td><td>±1.76%</td><td>0.9407 ± 0.0188</td><td>52.76 ms</td></tr>
            <tr><td>Truncated SVD + 1-NN</td><td>39</td><td>94.60%</td><td>±1.65%</td><td>0.9440 ± 0.0172</td><td>68.74 ms</td></tr>
            <tr><td><strong>LDA + 1-NN</strong></td><td>5</td><td>83.15%</td><td>±3.25%</td><td>0.8235 ± 0.0333</td><td>202.67 ms</td></tr>
            <tr><td><strong>LDA + 1-NN</strong></td><td>10</td><td>92.60%</td><td>±2.47%</td><td>0.9218 ± 0.0276</td><td>200.42 ms</td></tr>
            <tr><td><strong>LDA + 1-NN</strong></td><td>20</td><td>94.65%</td><td>±1.70%</td><td>0.9440 ± 0.0190</td><td>199.98 ms</td></tr>
            <tr><td><strong>LDA + 1-NN</strong></td><td>30</td><td>95.10%</td><td>±1.74%</td><td>0.9489 ± 0.0189</td><td>209.00 ms</td></tr>
            <tr style="background: rgba(16, 185, 129, 0.08);"><td><strong>LDA + 1-NN (Champion)</strong></td><td><strong>39</strong></td><td><strong>95.20%</strong></td><td><strong>±1.69%</strong></td><td><strong>0.9499 ± 0.0177</strong></td><td><strong>211.15 ms</strong></td></tr>
          </tbody>
        </table>
      </div>

      <h3 style="margin: 30px 0 10px;">Table C: Reconstruction and Storage Footprint</h3>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Components</th>
              <th>Reconstruction MSE</th>
              <th>Rel. Frobenius Error</th>
              <th>PSNR (dB)</th>
              <th>Feature Storage / Face</th>
              <th>Model Parameter Size</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>PCA</td><td>10</td><td>0.010302</td><td>21.01%</td><td>19.87 dB</td><td>40 B (0.04 KB)</td><td>442.75 KB</td></tr>
            <tr><td>Truncated SVD</td><td>10</td><td>0.010372</td><td>21.08%</td><td>19.84 dB</td><td>40 B (0.04 KB)</td><td>442.75 KB</td></tr>
            <tr><td>PCA</td><td>20</td><td>0.008387</td><td>18.95%</td><td>20.76 dB</td><td>80 B (0.08 KB)</td><td>845.25 KB</td></tr>
            <tr><td>Truncated SVD</td><td>20</td><td>0.008409</td><td>18.98%</td><td>20.75 dB</td><td>80 B (0.08 KB)</td><td>845.25 KB</td></tr>
            <tr><td>PCA</td><td>39</td><td>0.006957</td><td>17.26%</td><td>21.58 dB</td><td>156 B (0.15 KB)</td><td>1,610.00 KB</td></tr>
            <tr><td>Truncated SVD</td><td>39</td><td>0.006944</td><td>17.25%</td><td>21.58 dB</td><td>156 B (0.15 KB)</td><td>1,610.00 KB</td></tr>
            <tr><td>LDA</td><td>39</td><td>N/A</td><td>N/A</td><td>N/A</td><td>156 B (0.15 KB)</td><td>1,610.00 KB</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- Chapter 9: Interactive Quiz -->
    <section class="chapter" id="quiz">
      <div class="chapter-num">Chapter 9</div>
      <h2>Interactive Biometrics Knowledge Check</h2>
      <p>Test your understanding of facial subspace learning and experimental observations:</p>

      <div class="quiz-card" id="q1">
        <div class="quiz-q">1. Why do PCA and Truncated SVD achieve virtually identical recognition accuracy on the ORL dataset?</div>
        <div class="quiz-options">
          <button class="quiz-opt" onclick="checkAnswer(1, 0, false)">A) Both methods utilize identity class labels during training.</button>
          <button class="quiz-opt" onclick="checkAnswer(1, 1, true)">B) On centered dense data matrices, SVD right singular vectors are mathematically equivalent to PCA covariance eigenvectors.</button>
          <button class="quiz-opt" onclick="checkAnswer(1, 2, false)">C) The 1-NN classifier forces all models to have identical distances.</button>
          <button class="quiz-opt" onclick="checkAnswer(1, 3, false)">D) Both methods solve a generalized Rayleigh quotient.</button>
        </div>
        <div class="quiz-feedback" id="fb-1"></div>
      </div>

      <div class="quiz-card" id="q2">
        <div class="quiz-q">2. Why does LDA outperform raw 10,304-dimensional pixels while using only 39 components?</div>
        <div class="quiz-options">
          <button class="quiz-opt" onclick="checkAnswer(2, 0, true)">A) LDA acts as a discriminative filter: it isolates inter-person boundaries while stripping non-identity background and lighting noise.</button>
          <button class="quiz-opt" onclick="checkAnswer(2, 1, false)">B) High-dimensional spaces always have lower accuracy due to numerical overflow.</button>
          <button class="quiz-opt" onclick="checkAnswer(2, 2, false)">C) 1-NN requires exactly 39 components to function properly.</button>
          <button class="quiz-opt" onclick="checkAnswer(2, 3, false)">D) LDA reconstructs pixel intensity values with higher fidelity than raw pixels.</button>
        </div>
        <div class="quiz-feedback" id="fb-2"></div>
      </div>

      <div class="quiz-card" id="q3">
        <div class="quiz-q">3. What is the theoretical maximum number of valid LDA discriminant components for the ORL dataset (40 identities)?</div>
        <div class="quiz-options">
          <button class="quiz-opt" onclick="checkAnswer(3, 0, false)">A) 40 components (one per class)</button>
          <button class="quiz-opt" onclick="checkAnswer(3, 1, true)">B) 39 components (bounded by C - 1 rank of between-class scatter S_B)</button>
          <button class="quiz-opt" onclick="checkAnswer(3, 2, false)">C) 200 components (equal to training sample size)</button>
          <button class="quiz-opt" onclick="checkAnswer(3, 3, false)">D) Unlimited components</button>
        </div>
        <div class="quiz-feedback" id="fb-3"></div>
      </div>

      <div class="quiz-card" id="q4">
        <div class="quiz-q">4. On a training partition of 200 face images (10,304 features), what is the total maximum number of valid non-zero components for PCA vs. Truncated SVD?</div>
        <div class="quiz-options">
          <button class="quiz-opt" onclick="checkAnswer(4, 0, false)">A) PCA has 200 components; Truncated SVD has 199 components.</button>
          <button class="quiz-opt" onclick="checkAnswer(4, 1, true)">B) PCA has 199 components (mean-centering removes 1 d.o.f.); Truncated SVD has 200 components (uncentered full rank).</button>
          <button class="quiz-opt" onclick="checkAnswer(4, 2, false)">C) Both PCA and Truncated SVD are strictly limited to 39 components.</button>
          <button class="quiz-opt" onclick="checkAnswer(4, 3, false)">D) Both methods can extract up to 10,304 non-zero components.</button>
        </div>
        <div class="quiz-feedback" id="fb-4"></div>
      </div>
    </section>

    <!-- Chapter 10: Conclusions & Recommendations -->
    <section class="chapter">
      <div class="chapter-num">Chapter 10</div>
      <h2>Practical Engineering Guidelines for Biometric Systems</h2>
      <div class="stats-grid" style="margin: 24px 0;">
        <div class="stat-card emerald">
          <div class="stat-label">Identity Verification Systems</div>
          <div style="font-weight: 700; margin-bottom: 6px;">Choose LDA (Fisherfaces)</div>
          <div class="stat-desc">Yields optimal class margins, denoised projections, and highest recognition accuracy at 264× compression.</div>
        </div>
        <div class="stat-card cyan">
          <div class="stat-label">Image Compression & Synthesis</div>
          <div style="font-weight: 700; margin-bottom: 6px;">Choose PCA / SVD</div>
          <div class="stat-desc">Guaranteed minimum reconstruction MSE (Eckart-Young), preserving physical facial attributes and textures.</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Scalable Production Pipelines</div>
          <div style="font-weight: 700; margin-bottom: 6px;">Hybrid PCA + LDA</div>
          <div class="stat-desc">Reduce dimensions first via PCA to N-C (eliminating singularity), then apply LDA for maximal discrimination.</div>
        </div>
      </div>
    </section>
  </main>

  <!-- Footer -->
  <footer class="site-footer">
    <div class="footer-links">
      <a href="REPORT.md">📄 Technical Report (REPORT.md)</a>
      <a href="assignment3.ipynb">📓 Interactive Notebook (assignment3.ipynb)</a>
      <a href="PCA_SVD_LDA_ORL_Face_Dataset_Comparison_Report_Completed.docx">📑 Completed Word Report (.docx)</a>
      <a href="README.md">📖 Quick Start (README.md)</a>
    </div>
    <p style="color: var(--text-muted); font-size: 0.85rem;">
      Advance Biometrics Lab • Assignment 3 • SVNIT Surat
    </p>
  </footer>

  <!-- Interactive JavaScript Engine -->
  <script>
    // Embedded Data Engine
    const benchmarkData = __BENCHMARK_DATA__;
    const componentsList = [5, 10, 20, 30, 39, 50, 75, 100, 150];

    // Reading Progress Listener
    window.addEventListener('scroll', () => {
      const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (winScroll / height) * 100;
      document.getElementById('progress-bar').style.width = scrolled + '%';
    });

    // Theme Toggle
    const themeBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');
    let isDark = true;

    themeBtn.addEventListener('click', () => {
      isDark = !isDark;
      if (isDark) {
        document.documentElement.removeAttribute('data-theme');
        themeIcon.textContent = '☀️';
        themeText.textContent = 'Light Mode';
      } else {
        document.documentElement.setAttribute('data-theme', 'light');
        themeIcon.textContent = '🌙';
        themeText.textContent = 'Dark Mode';
      }
    });

    // Slider Interactive Update
    const slider = document.getElementById('component-slider');
    const kDisplayBadge = document.getElementById('k-display-badge');
    const kIndicatorVal = document.getElementById('k-indicator-val');

    function updatePlayground(kVal) {
      kDisplayBadge.textContent = `k = ${kVal} Components`;
      kIndicatorVal.textContent = `${kVal} / 150`;

      // Find PCA record
      const pcaRec = benchmarkData.find(d => d.method === 'PCA' && d.components === kVal);
      if (pcaRec) {
        document.getElementById('pca-acc-val').textContent = `${(pcaRec.accuracy_mean * 100).toFixed(2)}% ± ${(pcaRec.accuracy_std * 100).toFixed(2)}%`;
        document.getElementById('pca-acc-bar').style.width = `${(pcaRec.accuracy_mean * 100).toFixed(1)}%`;
        document.getElementById('pca-f1-val').textContent = pcaRec.macro_f1_mean.toFixed(4);
        document.getElementById('pca-mse-val').textContent = pcaRec.mse_mean.toFixed(6);
        document.getElementById('pca-storage-val').textContent = `${pcaRec.projected_bytes_per_image} B (${(pcaRec.projected_bytes_per_image/1024).toFixed(2)} KB)`;
      }

      // Find SVD record
      const svdRec = benchmarkData.find(d => d.method === 'Truncated SVD' && d.components === kVal);
      if (svdRec) {
        document.getElementById('svd-acc-val').textContent = `${(svdRec.accuracy_mean * 100).toFixed(2)}% ± ${(svdRec.accuracy_std * 100).toFixed(2)}%`;
        document.getElementById('svd-acc-bar').style.width = `${(svdRec.accuracy_mean * 100).toFixed(1)}%`;
        document.getElementById('svd-f1-val').textContent = svdRec.macro_f1_mean.toFixed(4);
        document.getElementById('svd-mse-val').textContent = svdRec.mse_mean.toFixed(6);
        document.getElementById('svd-storage-val').textContent = `${svdRec.projected_bytes_per_image} B (${(svdRec.projected_bytes_per_image/1024).toFixed(2)} KB)`;
      }

      // Find LDA record
      const ldaRec = benchmarkData.find(d => d.method === 'LDA' && d.components === kVal);
      const ldaCard = document.getElementById('lda-card');
      if (ldaRec) {
        ldaCard.style.opacity = '1';
        document.getElementById('lda-acc-val').textContent = `${(ldaRec.accuracy_mean * 100).toFixed(2)}% ± ${(ldaRec.accuracy_std * 100).toFixed(2)}%`;
        document.getElementById('lda-acc-bar').style.width = `${(ldaRec.accuracy_mean * 100).toFixed(1)}%`;
        document.getElementById('lda-f1-val').textContent = ldaRec.macro_f1_mean.toFixed(4);
        document.getElementById('lda-mse-val').textContent = 'N/A (Discriminative)';
        document.getElementById('lda-storage-val').textContent = `${ldaRec.projected_bytes_per_image} B (${(ldaRec.projected_bytes_per_image/1024).toFixed(2)} KB)`;
      } else {
        // k > 39 (Invalid for LDA)
        ldaCard.style.opacity = '0.45';
        document.getElementById('lda-acc-val').textContent = 'N/A (Max k = 39)';
        document.getElementById('lda-acc-bar').style.width = '0%';
        document.getElementById('lda-f1-val').textContent = 'N/A (Exceeds C-1)';
        document.getElementById('lda-mse-val').textContent = 'N/A';
        document.getElementById('lda-storage-val').textContent = 'N/A';
      }
    }

    slider.addEventListener('input', (e) => {
      const idx = parseInt(e.target.value);
      const kVal = componentsList[idx];
      updatePlayground(kVal);
    });

    // Tab Switching
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
      
      event.target.classList.add('active');
      document.getElementById(tabId).classList.add('active');
    }

    // Quiz Checking
    function checkAnswer(qNum, optIdx, isCorrect) {
      const fb = document.getElementById(`fb-${qNum}`);
      const card = document.getElementById(`q${qNum}`);
      const buttons = card.querySelectorAll('.quiz-opt');

      buttons.forEach((btn, idx) => {
        btn.disabled = true;
        if (idx === optIdx) {
          if (isCorrect) {
            btn.classList.add('correct');
          } else {
            btn.classList.add('incorrect');
          }
        }
      });

      fb.style.display = 'block';
      if (isCorrect) {
        fb.style.background = 'rgba(16, 185, 129, 0.15)';
        fb.style.color = '#34D399';
        fb.innerHTML = '<strong>✓ Correct!</strong> ' + (
          qNum === 1 ? 'For centered dense matrices, PCA covariance eigenvectors and SVD right singular vectors span the identical low-rank subspace.' :
          qNum === 2 ? 'LDA specifically suppresses intra-class variation and amplifies inter-class margins, effectively eliminating background and lighting noise.' :
          qNum === 3 ? 'The between-class scatter matrix S_B has rank at most C - 1. For 40 classes, rank is at most 39.' :
          'Mean-centering in PCA imposes the constraint sum(x_i - mu) = 0, reducing degrees of freedom to N - 1 = 199. Uncentered SVD retains all N = 200 dimensions!'
        );
      } else {
        fb.style.background = 'rgba(244, 63, 94, 0.15)';
        fb.style.color = '#FB7185';
        fb.innerHTML = '<strong>✗ Not quite.</strong> ' + (
          qNum === 1 ? 'Remember that on centered data, SVD right singular vectors are mathematically identical to PCA covariance eigenvectors!' :
          qNum === 2 ? 'Raw pixels contain massive high-frequency noise and redundant background pixels that confuse nearest-neighbor metrics.' :
          qNum === 3 ? 'With C = 40 classes, S_B has at most C - 1 = 39 linearly independent directions!' :
          'Centering subtracts the sample mean, which consumes 1 degree of freedom (N - 1 = 199 for PCA), while uncentered SVD retains rank N = 200!'
        );
      }
    }

    // Mathematical Typesetting Engine Auto-Trigger
    function triggerMathRender() {
      if (window.renderMathInElement) {
        renderMathInElement(document.body, {
          delimiters: [
            {left: "$$", right: "$$", display: true},
            {left: "\\[", right: "\\]", display: true},
            {left: "$", right: "$", display: false},
            {left: "\\(", right: "\\)", display: false}
          ],
          ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
          throwOnError: false
        });
      }
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise();
      }
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', triggerMathRender);
    } else {
      triggerMathRender();
    }
    window.addEventListener('load', triggerMathRender);
  </script>
</body>
</html>
"""

final_html = template.replace("__BENCHMARK_DATA__", json_raw)

with open(out_html_path, "w", encoding="utf-8") as f:
    f.write(final_html)
print(f"[OK] Successfully built: {out_html_path}")

with open(out_blog_path, "w", encoding="utf-8") as f:
    f.write(final_html)
print(f"[OK] Successfully built: {out_blog_path}")
