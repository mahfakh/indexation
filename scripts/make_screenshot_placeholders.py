import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'data', 'cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs('rapport_latex/figures', exist_ok=True)

placeholders = [
    (
        'screenshot_accueil.png',
        "Vue d'ensemble de l'Interface (Dashboard Accueil)",
        "Sidebar de configuration, statut backend en ligne, choix du dataset et indicateurs",
        "Capture de la page d'accueil avec le theme sombre GitHub Dark"
    ),
    (
        'screenshot_recherche_texte.png',
        "Scenario 1 : Recherche Textuelle par Mots-Cles",
        "Requete : 'butterfly' -> Grille responsive avec badges de rang et de categorie",
        "Capture de la grille de resultats pour une recherche textuelle exacte"
    ),
    (
        'screenshot_recherche_floue.png',
        "Scenario 2 : Recherche Floue Tolerant les Fautes",
        "Requete fautive : 'buterfly' (seuil = 0.70) -> Correction automatique difflib",
        "Capture illustrant le curseur de similarite floue actif et les resultats"
    ),
    (
        'screenshot_recherche_visuelle.png',
        "Scenario 3 : Recherche Visuelle CBIR (Query-by-Example)",
        "Image requete a gauche -> Top-20 voisins calcules par VP-Tree avec latence (32 ms)",
        "Capture de la double colonne requete visuelle et plus proches voisins"
    ),
    (
        'screenshot_recherche_combinee.png',
        "Scenario 4 : Recherche Hybride Multimodale",
        "Mots-cles + Image requete -> Ponderation alpha = 0.65 -> Fusion tardive",
        "Capture du mode combine avec curseur de ponderation texte/image"
    ),
    (
        'screenshot_indexation.png',
        "Scenario 5 : Administration et Indexation en Temps Reel",
        "Declenchement de l'indexation asynchrone avec barre de progression interactive",
        "Capture du panneau retractable d'indexation avec statut d'avancement"
    ),
]

for fname, title, desc, note in placeholders:
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0f1117')
    ax.set_facecolor('#0f1117')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # Box
    box = FancyBboxPatch((0.2, 0.2), 9.6, 5.1, boxstyle='round,pad=0.1', facecolor='#161b22', edgecolor='#30363d', lw=2)
    ax.add_patch(box)

    # Header bar
    hbar = FancyBboxPatch((0.2, 4.6), 9.6, 0.7, boxstyle='round,pad=0.05', facecolor='#21262d', edgecolor='#30363d', lw=1)
    ax.add_patch(hbar)
    for i, col in enumerate(['#da3633', '#d29922', '#238636']):
        c = plt.Circle((0.6 + i*0.3, 4.95), 0.1, color=col)
        ax.add_patch(c)

    ax.text(5.0, 4.95, 'CBIR Streamlit Dashboard - http://localhost:8501', color='#8b949e', fontsize=10, ha='center', va='center', family='sans-serif')

    ax.text(5.0, 3.4, "[ EMPLACEMENT CAPTURE D'ECRAN ]", color='#58a6ff', fontsize=13, fontweight='bold', ha='center', va='center')
    ax.text(5.0, 2.7, title, color='#f0f6fc', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(5.0, 2.0, desc, color='#c9d1d9', fontsize=9.5, ha='center', va='center', style='italic')
    ax.text(5.0, 1.4, note, color='#8b949e', fontsize=9, ha='center', va='center')
    ax.text(5.0, 0.7, f"Fichier a remplacer : rapport_latex/figures/{fname}", color='#238636', fontsize=8.5, ha='center', va='center', family='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join('rapport_latex', 'figures', fname), dpi=150)
    plt.close()

print('All 6 screenshot placeholders created successfully!')
