# CBIR — Moteur d'Indexation et de Recherche d'Images

Application client-serveur de **recherche d'images par le contenu (CBIR)** avec descripteurs couleur, LBP, HOG et support MPEG-7.  
Développé dans le cadre du cours d'indexation multimédia (SUPCOM).

---

## Architecture

```
cbir/
├── config.py           # Configuration via .env
├── database.py         # Gestionnaire SQLite
├── search.py           # Moteur de recherche (texte, visuel, combiné)
├── api.py              # API REST FastAPI
├── evaluation.py       # Évaluation (Précision@K, MAP)
├── descriptors/
│   ├── color.py        # Histogrammes RGB / HSV
│   ├── lbp.py          # Local Binary Patterns
│   ├── hog.py          # Histogram of Oriented Gradients
│   └── mpeg7.py        # Parseurs MPEG-7 (prêts si Dataset A fourni)
├── distances/
│   └── metrics.py      # L1, L2, Minkowski, χ², Intersection, Cosine, Bhattacharyya
└── indexing/
    ├── vptree.py       # VP-Tree (Vantage-Point Tree)
    └── indexer.py      # Pipeline d'indexation
frontend/
└── app.py              # Interface Streamlit
scripts/
├── run_indexing.py     # Indexation en ligne de commande
└── run_evaluation.py   # Évaluation en ligne de commande
tests/
├── test_distances.py
└── test_vptree.py
```

---

## Installation

### 1. Prérequis

- Python 3.10+
- pip

### 2. Cloner / ouvrir le projet

```bash
cd c:\Users\USER\Documents\antigravity\lively-maxwell
```

### 3. Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

### 4. Configurer les chemins des datasets

Copiez `.env.example` vers `.env` et ajustez les chemins :

```bash
copy .env.example .env
```

Le fichier `.env` est déjà configuré avec vos chemins réels :

```ini
DATASET_B_OBJECTS_PATH=C:/Users/USER/Desktop/ter_indexation_p1_ateliers2_cours (1)/...
DATASET_B_TEXTURES_PATH=C:/Users/USER/Desktop/ter_indexation_p1_ateliers2_cours (1)/...
STATIC_DIR=C:/Users/USER/Desktop/ter_indexation_p1_ateliers2_cours (1)/.../idb
```

---

## Démarrage rapide

### Étape 1 — Indexer le dataset Objects

```bash
python scripts/run_indexing.py --dataset objects --descriptors color_rgb lbp hog
```

Options :
- `--dataset` : `objects` ou `textures`
- `--descriptors` : un ou plusieurs parmi `color_rgb`, `color_hsv`, `lbp`, `hog`
- `--build-index` : construit le VP-Tree après l'extraction
- `--force` : réindexe même les images déjà traitées

### Étape 2 — Lancer le backend

```bash
uvicorn cbir.api:app --reload --host 0.0.0.0 --port 8000
```

Vérification : [http://localhost:8000/health](http://localhost:8000/health)  
Documentation API : [http://localhost:8000/docs](http://localhost:8000/docs)

### Étape 3 — Lancer le frontend

Dans un **second terminal** :

```bash
streamlit run frontend/app.py
```

Interface disponible à : [http://localhost:8501](http://localhost:8501)

---

## Tests

```bash
pytest tests/ -v
```

---

## Descripteurs disponibles

| Descripteur | Dimensions | Espace | Remarques |
|-------------|-----------|--------|-----------|
| `color_rgb` | bins × 3 (192 par défaut) | RGB | Histogramme par canal, normalisé L1 |
| `color_hsv` | 18+16+4 = 38 | HSV | Sensible à la teinte, compact |
| `lbp`       | 26 (24 pts + 2) | Niveaux de gris | Texture locale, invariant rotation |
| `hog`       | 1764 (128×128, 9 orient.) | Niveaux de gris | Forme et structure |

---

## Métriques de distance

| Métrique | Clé API | Usage recommandé |
|----------|---------|-----------------|
| Manhattan (L1) | `l1` | Histogrammes |
| Euclidienne (L2) | `l2` | Descripteurs normalisés |
| Minkowski (p=3) | `minkowski` | Usage général |
| Chi-deux | `chi_squared` | Histogrammes |
| Intersection | `intersection` | Histogrammes (similarité) |
| Cosinus (distance) | `cosine_dist` | Vecteurs denses (HOG) |
| Bhattacharyya | `bhattacharyya` | Distributions normalisées |

---

## Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET`  | `/health` | Statut du serveur |
| `GET`  | `/api/datasets` | Liste des datasets indexés |
| `GET`  | `/api/datasets/{id}/stats` | Statistiques d'un dataset |
| `POST` | `/api/index` | Lancer l'indexation |
| `GET`  | `/api/index/status` | Progression de l'indexation |
| `POST` | `/api/search/text` | Recherche par mots-clés |
| `POST` | `/api/search/image` | Recherche visuelle (CBIR) |
| `POST` | `/api/search/combined` | Recherche combinée |
| `POST` | `/api/search/upload` | Recherche par image uploadée |
| `GET`  | `/api/images/{id}` | Détail d'une image |
| `GET`  | `/api/descriptors/types` | Descripteurs disponibles |
| `GET`  | `/api/metrics` | Métriques disponibles |

---

## Structure de l'index (VP-Tree)

Le VP-Tree (Vantage-Point Tree) permet une recherche des k plus proches voisins en O(log N) comparaisons en moyenne, contre O(N) pour la recherche exhaustive.

- Construit séparément par type de descripteur et métrique de distance.
- Sauvegardé sous `data/cache/features/*.vptree.pkl`.
- Rechargé automatiquement au démarrage.

---

## Datasets

| Dataset | Chemin | Images | Catégories | Descripteurs |
|---------|--------|--------|-----------|-------------|
| Objects (Caltech-101) | `idb/Objects/` | ~9 146 | 102 | Couleur, LBP, HOG |
| Textures | `idb/Texture/jpg/` | ~6 208 | 28 | Couleur, LBP, HOG |
| Dataset A (MPEG-7) | Non disponible | ~1M | — | MPEG-7 (parseurs prêts) |

---

## Évaluation

```bash
python scripts/run_evaluation.py --dataset objects --descriptor color_rgb --metric l2 --k 20
```

Métriques calculées (vérité terrain = même catégorie) :
- **Précision@K** 
- **Rappel@K**
- **Mean Average Precision (mAP)**
- Comparaison exhaustive vs VP-Tree
