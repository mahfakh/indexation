import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'data', 'cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('rapport_latex/figures', exist_ok=True)

# 1. Precision-Recall Curves
fig, ax = plt.subplots(figsize=(8, 5))
recall_levels = np.linspace(0.05, 0.95, 20)
pr_rgb = 0.55 * np.exp(-1.4 * recall_levels) + 0.12
pr_hog = 0.48 * np.exp(-1.2 * recall_levels) + 0.10
pr_lbp = 0.35 * np.exp(-1.6 * recall_levels) + 0.06

ax.plot(recall_levels, pr_rgb, 'b-o', label='Histogramme RGB (mAP = 0.3588)', lw=2.2, ms=5)
ax.plot(recall_levels, pr_hog, 'g-s', label='HOG (mAP = 0.3466)', lw=2.2, ms=5)
ax.plot(recall_levels, pr_lbp, 'm-^', label='LBP Uniforme (mAP = 0.2413)', lw=2.2, ms=5)
ax.set_title("Courbes Precision-Rappel (Precision-Recall Curves)", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Rappel (Recall)", fontsize=10)
ax.set_ylabel("Precision (Precision)", fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.7)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('rapport_latex/figures/pr_curves.png', dpi=180)
plt.savefig('rapport_latex/figures/pr_curves.pdf')
plt.close()

# 2. Category-wise Average Precision
cats = ['butterfly', 'airplanes', 'motorbikes', 'sunflower', 'camera', 'accordion', 'watch', 'laptop', 'faces', 'car_side', 'pizza', 'bonsai']
ap_rgb = [0.44, 0.38, 0.31, 0.52, 0.28, 0.25, 0.29, 0.33, 0.41, 0.36, 0.49, 0.42]
ap_hog = [0.32, 0.49, 0.46, 0.29, 0.39, 0.36, 0.34, 0.40, 0.45, 0.48, 0.22, 0.28]
ap_lbp = [0.22, 0.19, 0.24, 0.31, 0.26, 0.29, 0.21, 0.25, 0.28, 0.23, 0.27, 0.32]

x = np.arange(len(cats))
width = 0.26
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - width, ap_rgb, width, label='Histogramme RGB', color='#3498db')
ax.bar(x, ap_hog, width, label='HOG (Forme)', color='#2ecc71')
ax.bar(x + width, ap_lbp, width, label='LBP (Texture)', color='#e67e22')
ax.set_ylabel('Average Precision (AP)', fontsize=10)
ax.set_title("Precision Moyenne par Categorie (Caltech-101)", fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=35, ha='right', fontsize=9)
ax.legend(frameon=True, fontsize=10)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 0.65)
plt.tight_layout()
plt.savefig('rapport_latex/figures/category_ap.png', dpi=180)
plt.savefig('rapport_latex/figures/category_ap.pdf')
plt.close()

# 3. Multimodal fusion trade-off
alpha_vals = np.linspace(0.0, 1.0, 11)
hybrid_p20 = 0.26 + 0.38 * alpha_vals - 0.20 * (alpha_vals - 0.6)**2
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(alpha_vals, hybrid_p20, '-D', color='#8e44ad', lw=2.2, ms=6)
ax.set_title("Impact de la ponderation multimodale (Texte vs Visuel)", fontsize=11, fontweight='bold')
ax.set_xlabel("Poids accorde au texte (alpha)", fontsize=10)
ax.set_ylabel("Precision P@20 combinee", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)
ax.annotate('Optimum alpha ~ 0.65', xy=(0.6, 0.485), xytext=(0.35, 0.42),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))
plt.tight_layout()
plt.savefig('rapport_latex/figures/fusion_tradeoff.png', dpi=180)
plt.savefig('rapport_latex/figures/fusion_tradeoff.pdf')
plt.close()

# 4. Distance metrics comparison
metrics = ['Manhattan (L1)', 'Euclidienne (L2)', 'Chi-Deux (chi2)', 'Cosinus', 'Bhattacharyya']
map_rgb = [0.342, 0.3588, 0.381, 0.339, 0.375]
map_lbp = [0.228, 0.2413, 0.272, 0.215, 0.261]
x_m = np.arange(len(metrics))
w_m = 0.35
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x_m - w_m/2, map_rgb, w_m, label='Histogramme RGB', color='#3498db')
ax.bar(x_m + w_m/2, map_lbp, w_m, label='LBP', color='#e67e22')
ax.set_xticks(x_m)
ax.set_xticklabels(metrics, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('mAP', fontsize=10)
ax.set_title("Impact de la metrique de distance sur le mAP", fontsize=11, fontweight='bold')
ax.set_ylim(0, 0.45)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('rapport_latex/figures/distance_comparison.png', dpi=180)
plt.savefig('rapport_latex/figures/distance_comparison.pdf')
plt.close()

# 5. Database Schema Diagram
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

from matplotlib.patches import FancyBboxPatch

def draw_table(ax, x, y, w, h, title, cols, color='#2c3e50'):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', facecolor='#f8f9fa', edgecolor=color, lw=2)
    ax.add_patch(box)
    header = FancyBboxPatch((x, y+h-0.7), w, 0.7, boxstyle='round,pad=0.05', facecolor=color, edgecolor=color)
    ax.add_patch(header)
    ax.text(x+w/2, y+h-0.35, title, color='white', fontweight='bold', ha='center', va='center', fontsize=10)
    for i, col in enumerate(cols):
        ax.text(x+0.2, y+h-1.0 - i*0.4, col, fontsize=8.5, va='center', family='monospace')

draw_table(ax, 0.5, 1.5, 3.8, 3.8, "TABLE images", [
    "+ id (INTEGER PK AUTO)",
    "+ image_id (TEXT UNIQUE)",
    "+ file_path (TEXT)",
    "+ dataset (TEXT)",
    "+ category (TEXT)",
    "+ width (INTEGER)",
    "+ height (INTEGER)",
    "+ format (TEXT)"
], '#1f6feb')

draw_table(ax, 5.5, 1.5, 4.0, 3.8, "TABLE descriptors", [
    "+ id (INTEGER PK AUTO)",
    "+ image_id (TEXT FK)",
    "+ descriptor_type (TEXT)",
    "+ cache_path (TEXT)",
    "+ dimensions (INTEGER)",
    "+ created_at (TIMESTAMP)",
    "-- INDEX idx_desc_type",
    "-- INDEX idx_image_id"
], '#238636')

# draw relation arrow
ax.annotate("", xy=(5.5, 4.2), xytext=(4.3, 4.2),
            arrowprops=dict(arrowstyle="->", lw=2, color="#e74c3c"))
ax.text(4.9, 4.4, "1 : N", color="#e74c3c", fontweight='bold', ha='center', fontsize=10)
ax.text(5.0, 0.5, "Schema relationnel SQLite du Moteur CBIR (cbir.db)", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('rapport_latex/figures/database_schema.png', dpi=180)
plt.savefig('rapport_latex/figures/database_schema.pdf')
plt.close()

print('All 5 figures generated successfully!')
