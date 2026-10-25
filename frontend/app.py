"""Frontend Streamlit pour le moteur CBIR.
Lancement :  streamlit run frontend/app.py
"""
import io
import time
import requests
import streamlit as st
from PIL import Image

# ─────────────────────────── Configuration ────────────────────────────────
API_BASE = "http://localhost:8000"
DEFAULT_DESCRIPTOR = "color_rgb"
DEFAULT_METRIC = "l2"

st.set_page_config(
    page_title="CBIR — Moteur de Recherche d'Images",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── CSS moderne ──────────────────────────────────
st.markdown("""
<style>
    /* ── Base & typographie ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background: linear-gradient(160deg, #0f1117 0%, #161b22 40%, #0d1117 100%);
        color: #e6edf3;
    }

    /* ── Titres ── */
    h1, h2, h3 {
        color: #f0f6fc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9;
    }

    /* ── Cards résultats ── */
    .result-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 8px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .result-card:hover {
        border-color: #58a6ff;
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(88,166,255,0.15);
    }

    .result-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
        font-size: 0.85rem;
        color: #8b949e;
    }

    .rank-badge {
        background: #21262d;
        color: #58a6ff;
        padding: 2px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .category-tag {
        background: #1f6feb22;
        color: #79c0ff;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* ── Score badges ── */
    .score-good {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .score-mid {
        background: linear-gradient(135deg, #9e6a03, #d29922);
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .score-poor {
        background: linear-gradient(135deg, #da3633, #f85149);
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ── Info bar ── */
    .info-bar {
        background: linear-gradient(90deg, #1f6feb22, #23863622);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 14px 20px;
        margin: 16px 0 24px 0;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.95rem;
        color: #c9d1d9;
    }

    /* ── Query image box ── */
    .query-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
    }

    /* ── Boutons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(31,111,235,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(31,111,235,0.45) !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider {
        border-radius: 10px !important;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid #30363d;
        margin: 1.2rem 0;
    }

    /* ── Success / Error ── */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px !important;
    }

    /* ── Image containers ── */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Header accent ── */
    .main-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }
    .main-header span {
        background: linear-gradient(135deg, #58a6ff, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Helpers API ──────────────────────────────────
def api_get(endpoint: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        st.error(f"API {r.status_code}: {r.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Backend non disponible. Lancez : `uvicorn cbir.api:app --reload`")
    except Exception as e:
        st.error(f"Erreur API : {e}")
    return None

def api_post(endpoint: str, data: dict = None, files=None) -> dict | None:
    try:
        if files:
            r = requests.post(f"{API_BASE}{endpoint}", params=data, files=files, timeout=60)
        else:
            r = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=60)
        if r.status_code == 200:
            return r.json()
        st.error(f"API {r.status_code}: {r.text[:300]}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Backend non disponible. Lancez : `uvicorn cbir.api:app --reload`")
    except Exception as e:
        st.error(f"Erreur API : {e}")
    return None

def load_image_from_url(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content))
    except Exception:
        pass
    return None

def distance_badge(distance: float | None, score: float | None) -> str:
    if score is not None:
        val = score
        label = f"Score {val:.3f}"
        cls = "score-good" if val > 0.7 else ("score-mid" if val > 0.4 else "score-poor")
    elif distance is not None:
        val = distance
        label = f"Dist {val:.4f}"
        cls = "score-good" if val < 0.3 else ("score-mid" if val < 0.7 else "score-poor")
    else:
        return ""
    return f'<span class="{cls}">{label}</span>'

def display_results_grid(results: list, cols: int = 4):
    """Affiche les résultats en grille moderne avec bouton 'Recherche similaire'."""
    if not results:
        st.info("Aucun résultat trouvé.")
        return

    for row_start in range(0, len(results), cols):
        row = results[row_start: row_start + cols]
        columns = st.columns(cols)
        for col, res in zip(columns, row):
            with col:
                img = load_image_from_url(res.get("url", ""))
                if img:
                    st.image(img, use_column_width=True)
                else:
                    st.markdown("🖼️ *Image non disponible*")

                cat = res.get("category", "—")
                rank = res.get("rank", "?")
                badge = distance_badge(res.get("distance"), res.get("score"))

                st.markdown(
                    f"""
                    <div class="result-meta">
                        <span class="rank-badge">#{rank}</span>
                        <span class="category-tag">{cat}</span>
                    </div>
                    <div style="margin-top:8px; text-align:center;">{badge}</div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("🔍 Similaire", key=f"sim_{res['image_id']}_{rank}", use_container_width=True):
                    st.session_state["query_image_id"] = res["image_id"]
                    st.session_state["trigger_visual_search"] = True
                    st.rerun()

# ─────────────────────────── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")

    # Santé backend
    health = api_get("/health")
    if health:
        st.success(f"✅ Backend OK · v{health.get('version', '?')}", icon="🟢")
    else:
        st.error("❌ Backend hors ligne")

    st.divider()

    # Dataset
    datasets_info = api_get("/api/datasets") or []
    ds_names = [d["name"] for d in datasets_info] if datasets_info else ["objects", "textures"]
    selected_dataset = st.selectbox(
        "📁 Dataset",
        options=["all"] + ds_names,
        index=0,
        key="sel_dataset",
    )
    dataset_filter = None if selected_dataset == "all" else selected_dataset

    # Mode de recherche
    search_mode = st.radio(
        "🔎 Mode de recherche",
        ["Texte", "Visuel", "Combiné"],
        key="search_mode",
        horizontal=False,
    )

    st.divider()

    # Descripteurs (Visuel / Combiné)
    if search_mode in ("Visuel", "Combiné"):
        st.markdown("#### 🧮 Descripteur")
        desc_types = api_get("/api/descriptors/types") or ["color_rgb", "color_hsv", "lbp", "hog"]
        descriptor_type = st.selectbox(
            "Type",
            desc_types,
            index=desc_types.index(DEFAULT_DESCRIPTOR) if DEFAULT_DESCRIPTOR in desc_types else 0,
            key="descriptor_type",
        )

        metrics = api_get("/api/metrics") or ["l2", "l1", "chi_squared", "cosine_dist"]
        distance_metric = st.selectbox(
            "Métrique de distance",
            metrics,
            index=metrics.index(DEFAULT_METRIC) if DEFAULT_METRIC in metrics else 0,
            key="distance_metric",
        )

        use_index = st.checkbox("⚡ Utiliser VP-Tree", value=True, key="use_index")

    max_results = st.slider("Nombre de résultats", 5, 100, 20, key="max_results")

    st.divider()

    # Upload
    if search_mode in ("Visuel", "Combiné"):
        st.markdown("#### 📤 Uploader une image")
        uploaded_file = st.file_uploader(
            "Choisir une image",
            type=["jpg", "jpeg", "png"],
            key="uploaded_file",
            label_visibility="collapsed",
        )

    st.divider()

    # Statistiques
    with st.expander("📊 Statistiques des datasets", expanded=False):
        if datasets_info:
            for ds in datasets_info:
                st.markdown(f"**{ds['name']}**")
                st.caption(f"{ds['total_images']} images · {ds['categories_count']} catégories")
                for desc, cnt in ds.get("descriptor_coverage", {}).items():
                    st.markdown(f"• `{desc}` → {cnt}")
                st.markdown("---")
        else:
            st.info("Aucun dataset indexé.")

    # Indexation
    with st.expander("🗄️ Indexation", expanded=False):
        idx_dataset = st.selectbox("Dataset à indexer", ["objects", "textures"], key="idx_dataset")
        idx_descriptors = st.multiselect(
            "Descripteurs",
            ["color_rgb", "color_hsv", "lbp", "hog"],
            default=["color_rgb", "lbp", "hog"],
            key="idx_descriptors",
        )
        idx_force = st.checkbox("Forcer la réindexation", key="idx_force")
        idx_build = st.checkbox("Construire VP-Tree après", value=True, key="idx_build")

        if st.button("🚀 Lancer l'indexation", key="btn_index", use_container_width=True):
            if not idx_descriptors:
                st.warning("Sélectionnez au moins un descripteur.")
            else:
                with st.spinner("Indexation en cours…"):
                    res = api_post("/api/index", {
                        "dataset": idx_dataset,
                        "descriptor_types": idx_descriptors,
                        "force_reindex": idx_force,
                    })
                    if res:
                        stats = res.get("stats", {})
                        st.success(
                            f"✅ {stats.get('processed', '?')}/{stats.get('total', '?')} images "
                            f"en {stats.get('time_seconds', 0):.1f}s"
                        )

        status_list = api_get("/api/index/status") or []
        for s in status_list:
            pct = s.get("progress_percent", 0)
            st.progress(pct / 100, text=f"{s['dataset']}: {s['status']} ({pct:.0f}%)")

# ──────────────────────────── Zone principale ─────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <span>🔍 CBIR</span>
        <span style="font-size:1.1rem; color:#8b949e; font-weight:500; -webkit-text-fill-color:#8b949e;">
            Moteur de Recherche d'Images
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════ RECHERCHE TEXTUELLE ═════════════════════════
if search_mode == "Texte":
    st.markdown("### 🔤 Recherche par mots-clés")

    query_text = st.text_input(
        "Entrez vos mots-clés",
        placeholder="ex: butterfly, airplane, buterfly…",
        key="text_query",
        label_visibility="collapsed",
    )

    col_fuz1, col_fuz2 = st.columns([1, 2.5])
    with col_fuz1:
        fuzzy_on = st.checkbox("🔀 Recherche floue", value=False, key="fuzzy_on")
    with col_fuz2:
        if fuzzy_on:
            fuzzy_threshold = st.slider(
                "Seuil de similarité",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="fuzzy_threshold",
                help="0.6 = tolérant · 0.9 = strict · 1.0 = exact",
            )
        else:
            fuzzy_threshold = 0.0
            st.caption("Activez la recherche floue pour tolérer les fautes de frappe.")

    if st.button("🔍 Rechercher", key="btn_text", use_container_width=False) or query_text:
        if not query_text.strip():
            st.warning("Entrez au moins un mot-clé.")
        else:
            with st.spinner("Recherche en cours…"):
                res = api_post("/api/search/text", {
                    "query": query_text.strip(),
                    "dataset": dataset_filter,
                    "max_results": max_results,
                    "fuzzy_threshold": fuzzy_threshold,
                })
            if res:
                mode_label = "🔀 fuzzy" if res.get("fuzzy") else "✅ exact"
                extra = f" · seuil={fuzzy_threshold:.2f}" if res.get("fuzzy") else ""
                st.markdown(
                    f"""
                    <div class="info-bar">
                        📋 <strong>{res["total"]}</strong> résultat(s)
                        · ⏱ {res["time_ms"]:.1f} ms
                        · Mode : {mode_label}{extra}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                display_results_grid(res.get("results", []))

# ═══════════════════════════ RECHERCHE VISUELLE ══════════════════════════
elif search_mode == "Visuel":
    st.markdown("### 🖼️ Recherche visuelle (CBIR)")

    col_q, col_r = st.columns([1, 2.8], gap="large")

    with col_q:
        st.markdown("**Image requête**")
        query_image_id = st.session_state.get("query_image_id")

        if uploaded_file:
            query_img = Image.open(uploaded_file)
            st.image(query_img, caption="Image uploadée", use_column_width=True)
            query_image_id = None
        elif query_image_id:
            img_detail = api_get(f"/api/images/{query_image_id}")
            if img_detail:
                qimg = load_image_from_url(img_detail.get("url", ""))
                if qimg:
                    st.image(qimg, caption=img_detail.get("category", ""), use_column_width=True)
                st.markdown(
                    f"""
                    <div class="query-box" style="margin-top:12px; text-align:left;">
                        <div style="font-size:0.85rem; color:#8b949e;">Catégorie</div>
                        <div style="font-weight:600; color:#79c0ff;">{img_detail.get("category", "—")}</div>
                        <div style="font-size:0.85rem; color:#8b949e; margin-top:8px;">Descripteurs</div>
                        <div style="font-size:0.8rem; color:#c9d1d9;">
                            {", ".join(img_detail.get("available_descriptors", [])) or "—"}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Uploadez une image ou cliquez sur « Similaire » dans les résultats.")

        manual_id = st.text_input("Ou entrez un image_id", key="manual_image_id", placeholder="ex: objects_00123")
        if manual_id:
            st.session_state["query_image_id"] = manual_id
            query_image_id = manual_id

    with col_r:
        do_search = (
            st.button("🔍 Rechercher", key="btn_visual")
            or st.session_state.pop("trigger_visual_search", False)
        )

        if do_search:
            if not query_image_id and not uploaded_file:
                st.warning("Sélectionnez une image requête.")
            else:
                with st.spinner("Recherche visuelle en cours…"):
                    if uploaded_file and not query_image_id:
                        uploaded_file.seek(0)
                        res = api_post(
                            "/api/search/upload",
                            data={
                                "descriptor_type": descriptor_type,
                                "distance_metric": distance_metric,
                                "max_results": max_results,
                            },
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")},
                        )
                    else:
                        res = api_post("/api/search/image", {
                            "query_image_id": query_image_id,
                            "descriptor_type": descriptor_type,
                            "distance_metric": distance_metric,
                            "max_results": max_results,
                            "dataset": dataset_filter,
                            "use_index": use_index,
                        })

                if res:
                    search_type = res.get("search_type", "?")
                    comparisons = res.get("comparisons", "?")
                    st.markdown(
                        f"""
                        <div class="info-bar">
                            📋 <strong>{res["total"]}</strong> résultat(s)
                            · ⏱ {res["time_ms"]:.1f} ms
                            · Mode : {search_type}
                            · Comparaisons : {comparisons}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    display_results_grid(res.get("results", []))

# ═══════════════════════════ RECHERCHE COMBINÉE ══════════════════════════
elif search_mode == "Combiné":
    st.markdown("### 🔤🖼️ Recherche combinée texte + image")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        comb_text = st.text_input(
            "Mots-clés de filtrage",
            placeholder="ex: butterfly",
            key="comb_text",
        )
    with c2:
        comb_id = st.text_input(
            "Image requête (image_id)",
            value=st.session_state.get("query_image_id", ""),
            key="comb_image_id",
            placeholder="ex: objects_00123",
        )

    text_weight = st.slider("Poids du texte", 0.0, 1.0, 0.5, 0.05, key="text_weight")

    if st.button("🔍 Rechercher", key="btn_combined"):
        if not comb_text.strip():
            st.warning("Entrez des mots-clés.")
        elif not comb_id.strip():
            st.warning("Entrez un image_id requête.")
        else:
            with st.spinner("Recherche combinée…"):
                res = api_post("/api/search/combined", {
                    "text_query": comb_text.strip(),
                    "query_image_id": comb_id.strip(),
                    "descriptor_type": descriptor_type,
                    "distance_metric": distance_metric,
                    "text_weight": text_weight,
                    "max_results": max_results,
                })
            if res:
                st.markdown(
                    f"""
                    <div class="info-bar">
                        📋 <strong>{res["total"]}</strong> résultat(s)
                        · ⏱ {res["time_ms"]:.1f} ms
                        · {res.get("comparisons", "?")} comparaisons
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                display_results_grid(res.get("results", []))