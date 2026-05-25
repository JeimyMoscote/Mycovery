"""
Mycovery · Metabolite Recommender v2.0
---------------------------------------
Carga: mycovery_803_clean.csv  (803 metabolitos con análisis completo)
Scoring: 6 criterios FQS (formulación) + relevancia por estresor (SRS)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mycovery · Metabolite Recommender v2",
    page_icon="🍄",
    layout="wide",
)

# ── ESTILOS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #f5f4f0; }
  .stApp { font-family: 'Helvetica Neue', Arial, sans-serif; }

  .title-block { background: #0C2518; border-radius: 12px; padding: 22px 32px; margin-bottom: 20px; }
  .title-main  { color: #1D9E75; font-size: 2.1rem; font-weight: 800; margin: 0; }
  .title-sub   { color: #9FE1CB; font-size: 0.9rem; margin: 4px 0 0 0; }
  .tagline     { color: #EF9F27; font-size: 0.82rem; font-style: italic; margin-top: 6px; }

  .score-card  { background: white; border-radius: 10px; padding: 14px 18px;
                 border-left: 4px solid #1D9E75; margin-bottom: 10px;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.07); }
  .score-card h4 { margin: 0 0 4px 0; color: #0C2518; font-size: 1rem; }
  .score-card p  { margin: 2px 0; color: #444; font-size: 0.82rem; line-height: 1.5; }

  .fqs-bar-wrap { display: flex; gap: 3px; margin: 6px 0 4px 0; align-items: center; }
  .fqs-seg { height: 8px; border-radius: 3px; }

  .badge { display:inline-block; padding:2px 9px; border-radius:12px;
           font-size:0.72rem; font-weight:600; margin-right:3px; }
  .alert-badge { background:#A32D2D; color:white; }
  .ok-badge    { background:#1D9E75; color:white; }

  .section-title { color:#0C2518; font-size:1.05rem; font-weight:700;
                   margin:20px 0 10px 0; border-bottom:2px solid #1D9E75; padding-bottom:4px; }

  .legend-dot { display:inline-block; width:10px; height:10px; border-radius:50%;
                margin-right:4px; vertical-align:middle; }

  .info-box { background:#e8f5ef; border-radius:8px; padding:10px 16px;
              border-left:3px solid #1D9E75; margin-bottom:12px; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTES ──────────────────────────────────────────────────────────

CAT_LABELS = {
    "FIT": "Fitohormonas",
    "COF": "Cofactores / Vitaminas",
    "OSM": "Osmoprotectores",
    "AOX": "Antioxidantes",
    "DEF": "Defensa vegetal",
    "SOL": "Solubilizadores",
    "AMI": "Aminoácidos",
    "POL": "Señales / Polisac.",
    "OTHER": "Sin clasificar",
    "RD": "Línea I+D",
}

CAT_COLORS = {
    "FIT": "#1D9E75", "COF": "#534AB7", "OSM": "#0C447C", "AOX": "#BA7517",
    "DEF": "#A32D2D", "SOL": "#5DCAA5", "AMI": "#854F0B", "POL": "#3B8B6A",
    "OTHER": "#888888", "RD": "#AAAAAA",
}

FQS_DIMS = {
    "fqs_sol":  {"label": "Solubilidad",          "icon": "💧", "max": 10,
                 "tooltip": "Silicos-IT class → facilidad de formulación líquida"},
    "fqs_gi":   {"label": "Absorción foliar",      "icon": "🌿", "max": 10,
                 "tooltip": "GI absorption → penetración pasiva de cutícula"},
    "fqs_bbb":  {"label": "Seguridad operario",    "icon": "🛡️", "max": 10,
                 "tooltip": "BBB permeant=No → sin riesgo neurotóxico"},
    "fqs_pers": {"label": "Persistencia campo",    "icon": "⏱️", "max": 10,
                 "tooltip": "BioTransformer stability → semanas de protección"},
    "fqs_noec": {"label": "Seguridad lombrices",   "icon": "🪱", "max": 10,
                 "tooltip": "NOEC [mg/Kg] → sin daño a macrofauna del suelo"},
    "fqs_cin":  {"label": "Estabilidad microbiana","icon": "🦠", "max": 10,
                 "tooltip": "Productos BioTransformer → integridad en suelo"},
}

STRESSOR_MAP = {
    "Sequía / Déficit hídrico": {
        "cats": ["OSM", "FIT", "AOX", "COF"],
        "weights": {"OSM": 1.0, "FIT": 0.9, "AOX": 0.7, "COF": 0.5},
        "desc": "Osmoprotección · regulación hormonal · captación de ROS",
        "emoji": "🌵",
    },
    "Salinidad": {
        "cats": ["OSM", "SOL", "AMI", "AOX"],
        "weights": {"OSM": 1.0, "SOL": 0.8, "AMI": 0.7, "AOX": 0.6},
        "desc": "Osmoprotección · movilización iónica · aminoácidos compatibles",
        "emoji": "🧂",
    },
    "Calor / Estrés térmico": {
        "cats": ["AOX", "COF", "OSM", "FIT"],
        "weights": {"AOX": 1.0, "COF": 0.8, "OSM": 0.7, "FIT": 0.6},
        "desc": "Captación de ROS · cofactores de reparación · señalización HSP",
        "emoji": "🌡️",
    },
    "Plaga / Insectos": {
        "cats": ["DEF", "AOX", "POL", "FIT"],
        "weights": {"DEF": 1.0, "AOX": 0.8, "POL": 0.7, "FIT": 0.5},
        "desc": "Defensinas · señales de reconocimiento · activación SAR",
        "emoji": "🐛",
    },
    "Hongo patógeno": {
        "cats": ["DEF", "AOX", "POL", "SOL"],
        "weights": {"DEF": 1.0, "AOX": 0.8, "POL": 0.6, "SOL": 0.5},
        "desc": "Compuestos antifúngicos · activación ISR/SAR · señales MAMP",
        "emoji": "🍄",
    },
    "Deficiencia de nutrientes": {
        "cats": ["SOL", "AMI", "COF", "FIT"],
        "weights": {"SOL": 1.0, "AMI": 0.8, "COF": 0.7, "FIT": 0.5},
        "desc": "Solubilización de P/K/Fe · aminoácidos · cofactores metabólicos",
        "emoji": "🌱",
    },
    "Frío / Helada": {
        "cats": ["OSM", "AOX", "COF", "FIT"],
        "weights": {"OSM": 0.9, "AOX": 1.0, "COF": 0.7, "FIT": 0.6},
        "desc": "Crioprotección · antioxidantes membranales · señalización de frío",
        "emoji": "❄️",
    },
    "Estrés hídrico post-cosecha": {
        "cats": ["OSM", "AMI", "AOX", "FIT"],
        "weights": {"OSM": 1.0, "AMI": 0.7, "AOX": 0.6, "FIT": 0.9},
        "desc": "Osmoprotección · retardo de senescencia · regulación hormonal",
        "emoji": "📦",
    },
}

# ── CARGA DE DATOS ───────────────────────────────────────────────────────

CSV_FILENAME = "mycovery_803_clean.csv"

@st.cache_data
def load_data() -> pd.DataFrame:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, CSV_FILENAME),
        os.path.join(os.getcwd(), CSV_FILENAME),
        CSV_FILENAME,
        os.path.join("/mount/src/mycovery", CSV_FILENAME),
        os.path.join("/mount/src/Mycovery", CSV_FILENAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return pd.read_csv(path, sep=";")
    return None


# ── SCORING ─────────────────────────────────────────────────────────────

def compute_srs(df: pd.DataFrame, stressors: list[str]) -> pd.Series:
    """Stressor Relevance Score, normalizado a 0-40."""
    scores = pd.Series(0.0, index=df.index)
    for s in stressors:
        for cat, w in STRESSOR_MAP[s]["weights"].items():
            scores[df["cat"] == cat] += w * 10
    max_poss = sum(max(STRESSOR_MAP[s]["weights"].values()) * 10 for s in stressors)
    if max_poss > 0:
        scores = (scores / max_poss * 40).clip(0, 40)
    return scores


def combined_score(df: pd.DataFrame, stressors: list[str],
                   w_fqs: float = 0.55, w_srs: float = 0.45) -> pd.Series:
    fqs_pct = df["FQS"] / 60.0 * 100
    srs_raw = compute_srs(df, stressors)
    srs_pct = srs_raw / 40.0 * 100 if stressors else pd.Series(0.0, index=df.index)
    if stressors:
        return w_fqs * fqs_pct + w_srs * srs_pct
    else:
        return fqs_pct


def select_diverse_top(df: pd.DataFrame, top_n: int,
                       include_unclassified: bool = False) -> pd.DataFrame:
    """Selecciona top_n maximizando diversidad de categorías funcionales."""
    pool = df.copy()
    if not include_unclassified:
        classified = pool[~pool["cat"].isin(["OTHER", "RD"])]
        unclassified = pool[pool["cat"].isin(["OTHER", "RD"])]
        # Fill remainder with unclassified if needed
        pool = pd.concat([classified, unclassified])

    selected = []
    used_cats = {}
    MAX_PER_CAT = max(2, top_n // 4)

    for _, row in pool.iterrows():
        if len(selected) >= top_n:
            break
        c = row["cat"]
        if c not in ("OTHER", "RD"):
            if used_cats.get(c, 0) >= MAX_PER_CAT:
                continue
            used_cats[c] = used_cats.get(c, 0) + 1
        selected.append(row)

    return pd.DataFrame(selected).reset_index(drop=True)


# ── RENDER DE CARD ───────────────────────────────────────────────────────

FQS_COLORS = ["#1D9E75", "#0C447C", "#534AB7", "#BA7517", "#5DCAA5", "#A32D2D"]


def render_card(rank: int, row: pd.Series, total: float) -> None:
    cat = row["cat"]
    cat_color = CAT_COLORS.get(cat, "#888")
    cat_label = CAT_LABELS.get(cat, cat)

    # FQS mini bar
    fqs_keys = list(FQS_DIMS.keys())
    fqs_vals = [int(row.get(k, 0)) for k in fqs_keys]
    fqs_sum = sum(fqs_vals)

    # Mini bar HTML (proportional segments)
    seg_html = ""
    for i, (k, v) in enumerate(zip(fqs_keys, fqs_vals)):
        width_pct = (v / 60) * 100
        color = FQS_COLORS[i]
        dim_label = FQS_DIMS[k]["label"]
        seg_html += (
            f'<span class="fqs-seg" '
            f'style="width:{width_pct:.1f}%;background:{color};'
            f'display:inline-block;height:8px;border-radius:2px;margin-right:2px;" '
            f'title="{dim_label}: {v}/10"></span>'
        )

    # Badges
    bbb_badge = (
        '<span class="badge alert-badge">⚠️ BBB+</span>'
        if row.get("BBB") == "Yes" else
        '<span class="badge ok-badge">✅ Seguro operario</span>'
    )
    mut_badge = (
        '<span class="badge alert-badge">⚠️ Mutagénico</span>'
        if row.get("Mutagenicity") == "Mutagenic" else
        '<span class="badge ok-badge">✅ No mutagénico</span>'
    )
    strain_badge = f'<span class="badge" style="background:#0C2518;color:#9FE1CB">{row.get("Strain","?")}</span>'
    cat_badge = f'<span class="badge" style="background:{cat_color};color:white">{cat_label}</span>'

    # Score bar
    score_pct = int(total)
    bar_filled = "█" * (score_pct // 10)
    bar_empty  = "░" * (10 - score_pct // 10)
    fqs_display = f"{fqs_sum}/60"

    st.markdown(f"""
    <div class="score-card">
      <h4>{rank}. {row['Compound']}</h4>
      <p style="margin-bottom:5px">{cat_badge} {strain_badge} {bbb_badge} {mut_badge}</p>
      <p>
        <strong>Score combinado:</strong> {bar_filled}{bar_empty} {score_pct}% &nbsp;|&nbsp;
        <strong>FQS:</strong> {fqs_display} &nbsp;|&nbsp;
        <strong>MW:</strong> {row.get('MW','?')} Da &nbsp;|&nbsp;
        <strong>LogP:</strong> {row.get('LogP','?')} &nbsp;|&nbsp;
        <strong>GI:</strong> {row.get('GI_abs','?')}
      </p>
      <p style="margin-top:4px"><strong>Perfil FQS (6 criterios):</strong></p>
      <div class="fqs-bar-wrap">{seg_html}</div>
      <p style="font-size:0.75rem;color:#666">
        {'  ·  '.join([f"{FQS_DIMS[k]['icon']} {FQS_DIMS[k]['label']}: {int(row.get(k,0))}/10" for k in fqs_keys])}
      </p>
    </div>
    """, unsafe_allow_html=True)


# ── HEADER ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
  <p class="title-main">🍄 Mycovery</p>
  <p class="title-sub">Fungal metabolomics for next-gen bioinputs · v2.0 · 803 metabolitos</p>
  <p class="tagline">Nature encoded it. We decoded it.</p>
</div>
""", unsafe_allow_html=True)

# ── CARGA CSV ────────────────────────────────────────────────────────────

df_full = load_data()
if df_full is not None:
    st.success(f"✅ Base de datos cargada · **{len(df_full)} metabolitos** con análisis completo")
else:
    st.error(
        "❌ No se encontró `mycovery_803_clean.csv`. "
        "Asegurate de que el archivo esté en la misma carpeta que `app_mycovery_v2.py` en tu repositorio de GitHub."
    )
    st.stop()

# ── SIDEBAR ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configura tu recomendación")

    cultivo = st.selectbox("🌿 Cultivo objetivo", [
        "Tomate", "Vid", "Trigo", "Maíz", "Arándano", "Lechuga",
        "Papa", "Cebolla", "Pimiento", "Olivo", "Aguacate", "Otro",
    ])

    stressors = st.multiselect(
        "⚡ Estresores (uno o más)",
        list(STRESSOR_MAP.keys()),
        default=["Sequía / Déficit hídrico"],
    )

    top_n = st.slider("Nº de metabolitos a recomendar", 3, 15, 8)

    include_unclassified = st.checkbox(
        "Incluir metabolitos sin clasificar (I+D)", value=False,
        help="Los 624 metabolitos sin categoría funcional aún. "
             "Útil para exploración de candidatos nuevos."
    )

    st.markdown("---")
    st.markdown("**Pesos del score combinado**")
    w_fqs = st.slider("Peso FQS (formulación)", 0.1, 0.9, 0.55, 0.05)
    w_srs = round(1.0 - w_fqs, 2)
    st.caption(f"Peso SRS (relevancia estresor): {w_srs}")

    st.markdown("---")
    st.markdown("**Filtros de formulación**")
    max_mw = st.slider("MW máximo (Da)", 100, 700, 500)
    logp_range = st.slider("Rango LogP", -5.0, 5.0, (-4.0, 3.0))
    min_fqs = st.slider("FQS mínimo (0-60)", 0, 60, 45)

    only_nonmuta = st.checkbox("Solo no mutagénicos (Ames test)", value=True)
    only_safe_bbb = st.checkbox("Solo seguros para operario (BBB−)", value=False)

    st.markdown("---")
    run = st.button("🔍 Recomendar mezcla", use_container_width=True, type="primary")

# ── LÓGICA PRINCIPAL ─────────────────────────────────────────────────────

if not run and "results_df" not in st.session_state:
    st.info("👈 Configura los parámetros en el panel izquierdo y pulsa **Recomendar mezcla**.")
    # Muestra estadísticas de la BD
    st.markdown('<p class="section-title">Distribución de la base de datos</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_counts = df_full["cat"].value_counts()
        labels = [CAT_LABELS.get(c, c) for c in cat_counts.index]
        colors = [CAT_COLORS.get(c, "#888") for c in cat_counts.index]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(labels[::-1], cat_counts.values[::-1], color=colors[::-1])
        ax.set_xlabel("N° metabolitos")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f5f4f0")
        fig.patch.set_facecolor("#f5f4f0")
        ax.set_title("Por categoría funcional", fontsize=9, color="#0C2518")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        strain_counts = df_full["Strain"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        ax2.pie(strain_counts.values, labels=strain_counts.index,
                autopct='%1.0f%%', colors=["#1D9E75","#534AB7","#0C447C"],
                textprops={"fontsize": 9})
        ax2.set_title("Por cepa productora", fontsize=9, color="#0C2518")
        fig2.patch.set_facecolor("#f5f4f0")
        plt.tight_layout()
        st.pyplot(fig2)
    with col3:
        fqs_vals = df_full["FQS"].dropna()
        fig3, ax3 = plt.subplots(figsize=(4, 3.5))
        ax3.hist(fqs_vals, bins=15, color="#1D9E75", edgecolor="white", alpha=0.85)
        ax3.set_xlabel("FQS (0-60)")
        ax3.set_ylabel("N° metabolitos")
        ax3.set_title("Distribución FQS", fontsize=9, color="#0C2518")
        ax3.spines[["top","right"]].set_visible(False)
        ax3.set_facecolor("#f5f4f0")
        fig3.patch.set_facecolor("#f5f4f0")
        plt.tight_layout()
        st.pyplot(fig3)
    st.stop()

# ── APPLY FILTERS & SCORE ────────────────────────────────────────────────

df = df_full.copy()

# Filtros booleanos
if only_nonmuta:
    df = df[df["Mutagenicity"] == "NON-Mutagenic"]
if only_safe_bbb:
    df = df[df["BBB"] == "No"]

# Filtros numéricos
df = df[
    (df["MW"].fillna(9999) <= max_mw) &
    (df["LogP"].fillna(99).between(logp_range[0], logp_range[1])) &
    (df["FQS"].fillna(0) >= min_fqs)
]

if len(df) == 0:
    st.error("❌ Ningún metabolito pasa los filtros actuales. Relaja los criterios.")
    st.stop()

# Score combinado
df["score_total"] = combined_score(df, stressors, w_fqs=w_fqs, w_srs=w_srs)
df = df.sort_values("score_total", ascending=False)

# Selección diversa
results = select_diverse_top(df, top_n, include_unclassified=include_unclassified)

if len(results) == 0:
    st.warning("No se encontraron metabolitos clasificados para los estresores seleccionados. "
               "Activa 'Incluir metabolitos sin clasificar' para ver candidatos I+D.")
    st.stop()

st.session_state["results_df"] = results

# ── MÉTRICAS RESUMEN ─────────────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cultivo", cultivo)
m2.metric("Estresores", len(stressors) if stressors else "— (solo FQS)")
m3.metric("Candidatos en pool", f"{len(df):,}")
m4.metric("Metabolitos recomendados", len(results))
cat_set = set(results["cat"]) - {"OTHER", "RD"}
m5.metric("Categorías cubiertas", f"{len(cat_set)}/8")

st.markdown("---")

# ── ESTRESORES SELECCIONADOS ──────────────────────────────────────────────

if stressors:
    st.markdown('<p class="section-title">Estresores y categorías requeridas</p>',
                unsafe_allow_html=True)
    cols_st = st.columns(min(len(stressors), 4))
    for i, s in enumerate(stressors):
        info = STRESSOR_MAP[s]
        with cols_st[i % len(cols_st)]:
            cats_str = " · ".join(
                '<span style="color:{};font-weight:600">{}</span>'.format(
                    CAT_COLORS.get(c, "#888"), CAT_LABELS.get(c, c)
                )
                for c in info["cats"]
            )
            st.markdown(
                f'<div class="info-box"><strong>{info["emoji"]} {s}</strong><br>'
                f'{cats_str}<br><span style="color:#555">{info["desc"]}</span></div>',
                unsafe_allow_html=True
            )

st.markdown("---")

# ── MEZCLA RECOMENDADA ───────────────────────────────────────────────────

st.markdown(f'<p class="section-title">Mezcla recomendada para {cultivo}</p>',
            unsafe_allow_html=True)

# Leyenda FQS
legend_html = "".join(
    f'<span style="display:inline-flex;align-items:center;margin-right:12px;font-size:0.78rem">'
    f'<span style="width:10px;height:10px;border-radius:2px;background:{FQS_COLORS[i]};'
    f'display:inline-block;margin-right:4px"></span>{FQS_DIMS[k]["icon"]} {FQS_DIMS[k]["label"]}</span>'
    for i, k in enumerate(FQS_DIMS.keys())
)
st.markdown(
    f'<div style="margin-bottom:10px;font-size:0.82rem"><strong>Leyenda FQS:</strong> {legend_html}</div>',
    unsafe_allow_html=True
)

for rank, (_, row) in enumerate(results.iterrows(), 1):
    render_card(rank, row, row["score_total"])

st.markdown("---")

# ── GRÁFICOS ──────────────────────────────────────────────────────────────

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<p class="section-title">Score combinado por metabolito</p>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5.5, max(3, len(results) * 0.45)))
    names = [
        r["Compound"][:30] + "…" if len(r["Compound"]) > 30 else r["Compound"]
        for _, r in results.iterrows()
    ]
    scores = [r["score_total"] for _, r in results.iterrows()]
    colors = [CAT_COLORS.get(r["cat"], "#888") for _, r in results.iterrows()]
    ax.barh(names[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel("Score combinado (0-100)")
    ax.set_xlim(0, 105)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#f5f4f0")
    fig.patch.set_facecolor("#f5f4f0")
    # Leyenda de categorías
    seen_cats = list({r["cat"] for _, r in results.iterrows()})
    patches = [mpatches.Patch(color=CAT_COLORS.get(c,"#888"), label=CAT_LABELS.get(c,c))
               for c in seen_cats]
    ax.legend(handles=patches, fontsize=7, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)

with col_b:
    st.markdown('<p class="section-title">Perfil FQS promedio de la mezcla</p>',
                unsafe_allow_html=True)
    fqs_means = {k: results[k].mean() for k in FQS_DIMS.keys()}
    dim_labels = [f"{FQS_DIMS[k]['icon']} {FQS_DIMS[k]['label']}" for k in FQS_DIMS]
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
    bars2 = ax2.barh(dim_labels[::-1], [fqs_means[k] for k in reversed(FQS_DIMS.keys())],
                     color=FQS_COLORS[::-1])
    ax2.set_xlim(0, 11)
    ax2.axvline(10, color="#ccc", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Puntuación promedio (0-10)")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_facecolor("#f5f4f0")
    fig2.patch.set_facecolor("#f5f4f0")
    ax2.set_title("Promedio de la mezcla seleccionada", fontsize=9, color="#0C2518")
    # Value labels
    for bar, val in zip(bars2, [fqs_means[k] for k in reversed(FQS_DIMS.keys())]):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)

# Cobertura de categorías
st.markdown('<p class="section-title">Cobertura de categorías funcionales en la mezcla</p>',
            unsafe_allow_html=True)
cat_counts_mix = results["cat"].value_counts()
fig3, ax3 = plt.subplots(figsize=(10, 2.2))
cats_all = [c for c in CAT_LABELS if c not in ("OTHER", "RD")]
vals_all = [cat_counts_mix.get(c, 0) for c in cats_all]
colors_all = [CAT_COLORS[c] for c in cats_all]
bars3 = ax3.bar([CAT_LABELS[c] for c in cats_all], vals_all, color=colors_all)
ax3.set_ylabel("N° metabolitos")
ax3.spines[["top","right"]].set_visible(False)
ax3.set_facecolor("#f5f4f0")
fig3.patch.set_facecolor("#f5f4f0")
for bar, v in zip(bars3, vals_all):
    if v > 0:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.tight_layout()
st.pyplot(fig3)

st.markdown("---")

# ── TABLA EXPORTABLE ─────────────────────────────────────────────────────

st.markdown('<p class="section-title">Tabla para laboratorio / formulación</p>',
            unsafe_allow_html=True)

export_cols = {
    "ID": "ID",
    "Strain": "Cepa",
    "Compound": "Metabolito",
    "cat": "Categoría",
    "score_total": "Score (%)",
    "FQS": "FQS (0-60)",
    "fqs_sol": "Solubilidad",
    "fqs_gi": "Absorción Foliar",
    "fqs_bbb": "Seguridad Op.",
    "fqs_pers": "Persistencia",
    "fqs_noec": "Seg. Lombrices",
    "fqs_cin": "Estab. Micro.",
    "MW": "MW (Da)",
    "LogP": "LogP",
    "GI_abs": "GI Abs.",
    "BBB": "BBB",
    "Mutagenicity": "Mutagenicidad",
    "NOEC_num": "NOEC (mg/Kg)",
}

export = results[[c for c in export_cols if c in results.columns]].copy()
export = export.rename(columns={c: export_cols[c] for c in export_cols if c in export.columns})
export["Categoría"] = export["Categoría"].map(CAT_LABELS)
export["Score (%)"] = export["Score (%)"].apply(lambda x: f"{x:.1f}")
export["FQS (0-60)"] = export["FQS (0-60)"].apply(lambda x: f"{int(x)}")

st.dataframe(export, use_container_width=True, hide_index=True)

csv_bytes = export.to_csv(index=False).encode("utf-8")
fname = f"mycovery_{cultivo}_{'_'.join(s[:5].replace('/','') for s in stressors)}.csv"
st.download_button(
    "⬇️ Descargar tabla CSV",
    csv_bytes,
    fname,
    "text/csv",
    use_container_width=True,
)

st.markdown("---")
st.markdown(
    "_Mycovery v2.0 · Fungal metabolomics for next-gen bioinputs · "
    "803 metabolitos · Atacama Desert · Chile · 2026_"
)


st.markdown(
    '<div style="background:#0C2518;border-radius:10px;padding:18px 24px;margin-top:16px;text-align:center">'
    '<p style="color:#9FE1CB;font-size:0.82rem;margin:0 0 6px 0">'
    '&#169; 2026 <strong style="color:#1D9E75">Mycovery</strong> &middot; Todos los derechos reservados</p>'
    '<p style="color:#6db89a;font-size:0.75rem;margin:0 0 4px 0">'
    'La base de datos de metabolitos f&#250;ngicos, el algoritmo de puntuaci&#243;n'
    ' multidimensional (FQS + SRS) y la metodolog&#237;a de recomendaci&#243;n de mezclas'
    ' son propiedad intelectual exclusiva de Mycovery.</p>'
    '<p style="color:#6db89a;font-size:0.75rem;margin:0">'
    'Queda prohibida la reproducci&#243;n, distribuci&#243;n o uso comercial'
    ' sin autorizaci&#243;n expresa y por escrito.'
    ' &middot; Atacama Desert &middot; Chile'
    ' &middot; <em>Nature encoded it. We decoded it.</em></p></div>',
    unsafe_allow_html=True
)
