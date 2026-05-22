import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ──
st.set_page_config(
    page_title="Mycovery · Metabolite Recommender",
    page_icon="🍄",
    layout="wide"
)

# ── ESTILOS ──
st.markdown("""
<style>
    .main { background-color: #f5f4f0; }
    .stApp { font-family: 'Helvetica Neue', Arial, sans-serif; }
    .title-block { background: #0C2518; border-radius: 12px; padding: 24px 32px; margin-bottom: 24px; }
    .title-main { color: #1D9E75; font-size: 2.2rem; font-weight: 800; margin: 0; }
    .title-sub  { color: #9FE1CB; font-size: 0.95rem; margin: 4px 0 0 0; }
    .tagline    { color: #EF9F27; font-size: 0.85rem; font-style: italic; margin-top: 6px; }
    .score-card { background: white; border-radius: 10px; padding: 16px; border-left: 4px solid #1D9E75; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
    .score-card h4 { margin: 0 0 4px 0; color: #0C2518; font-size: 1rem; }
    .score-card p  { margin: 0; color: #555; font-size: 0.85rem; line-height: 1.5; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-right: 4px; }
    .section-title { color: #0C2518; font-size: 1.1rem; font-weight: 700; margin: 20px 0 10px 0; border-bottom: 2px solid #1D9E75; padding-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ── DATOS ──
@st.cache_data
def load_data():
    data = [
        ["Indole-3-acetic acid","FIT",175.18,2,2,53.09,2,1.08],
        ["Indolepyruvic acid","FIT",203.19,2,3,70.35,3,0.85],
        ["Methyl indole-3-acetate","FIT",189.21,1,2,39.86,3,1.62],
        ["Indoleacetyl glutamine","FIT",303.31,3,4,115.97,7,0.12],
        ["Phenylacetic acid","FIT",136.15,1,2,37.30,2,1.41],
        ["Indoleacrylic acid","FIT",187.19,2,2,53.09,3,1.45],
        ["trans-Zeatin riboside","FIT",351.37,5,7,139.66,7,-1.20],
        ["Salicylic Acid","FIT",138.12,2,3,57.53,1,2.24],
        ["Traumatin","FIT",198.22,2,3,66.76,8,1.10],
        ["Anthranilic acid","FIT",137.14,2,3,63.32,1,0.98],
        ["Mevalonic acid","FIT",148.16,3,4,77.76,4,-0.48],
        ["Adenine","FIT",135.13,2,5,98.96,0,-0.09],
        ["4-Coumaric acid","FIT",164.16,2,3,57.53,2,1.39],
        ["Salicylate beta-D-Glucose Ester","FIT",300.26,5,8,136.68,5,0.12],
        ["Hypoxanthine","FIT",136.11,2,4,79.91,0,-0.96],
        ["Biotin","COF",244.31,3,4,107.02,6,0.39],
        ["Folic acid","COF",441.40,5,9,210.56,9,-2.50],
        ["Pantothenic acid","COF",219.24,4,6,120.65,7,-0.98],
        ["Thiamine","COF",265.35,1,4,98.96,3,-0.43],
        ["Pyridoxal","COF",167.16,2,4,76.56,1,0.54],
        ["Pyridoxamine","COF",168.19,3,4,89.96,1,-0.65],
        ["Nicotinic acid","COF",123.11,1,3,50.19,1,0.36],
        ["4-Aminobenzoic acid","COF",137.14,2,3,63.32,1,0.83],
        ["Ribothymidine","COF",258.23,4,7,119.07,3,-1.08],
        ["Uridine","COF",244.20,4,7,119.07,2,-1.44],
        ["(R)-Pantoic acid","COF",134.13,3,4,77.76,3,-0.43],
        ["4-Pyridoxic Acid","COF",183.16,3,5,96.22,1,-0.15],
        ["Ectoine","OSM",142.16,1,3,54.13,1,-3.40],
        ["1-Pyrroline-5-carboxylic acid","OSM",113.11,1,3,46.53,1,-0.98],
        ["2-Hydroxybutyric acid","OSM",104.10,2,3,57.53,2,0.05],
        ["Lactic Acid","OSM",90.08,2,3,57.53,1,-0.72],
        ["Glycolic Acid","OSM",76.05,2,3,57.53,1,-1.11],
        ["Threonic Acid","OSM",136.10,4,5,97.99,3,-1.92],
        ["cis-Aconitic acid","OSM",174.11,3,5,94.83,3,-0.34],
        ["Isocitric acid","OSM",192.12,4,6,115.06,3,-0.97],
        ["2-Hydroxyglutaric Acid","OSM",148.12,3,5,94.83,3,-0.55],
        ["N4-Acetyl-Spermidine","OSM",187.28,3,3,67.52,8,0.18],
        ["Glutamine","OSM",146.15,3,4,106.36,4,-3.64],
        ["Creatine","OSM",131.13,3,4,108.24,2,-1.26],
        ["Hydroxyproline","OSM",131.13,3,4,83.55,2,-1.24],
        ["Theanine","OSM",174.20,3,4,99.38,6,-2.12],
        ["Gallic Acid","AOX",170.12,4,5,97.99,1,0.70],
        ["Rosmarinic acid","AOX",360.36,4,7,133.52,6,0.98],
        ["Catechol","AOX",110.11,2,2,40.46,0,0.88],
        ["3,4-Dihydroxybenzoic acid","AOX",154.12,3,4,77.76,1,0.70],
        ["Homogentisic acid","AOX",168.15,3,4,77.76,2,0.44],
        ["Ferulic acid","AOX",194.19,2,4,66.76,3,1.35],
        ["Cinnamic Acid","AOX",148.16,1,2,37.30,2,2.13],
        ["Piceatannol","AOX",244.24,4,4,80.92,2,2.10],
        ["Isoferulic acid","AOX",194.19,2,4,66.76,3,1.35],
        ["Vanillic Acid","AOX",168.15,2,4,66.76,2,1.35],
        ["Syringic Acid","AOX",198.17,2,5,76.30,3,0.99],
        ["Homovanillic acid","AOX",182.17,2,4,66.76,3,0.66],
        ["2,5-Dihydroxybenzoic acid","AOX",154.12,3,4,77.76,1,0.54],
        ["Plumbagin","AOX",188.18,1,3,54.37,1,1.83],
        ["Salicyluric Acid","DEF",195.17,3,4,90.75,3,0.54],
        ["Umbelliferone","DEF",162.14,1,3,46.53,1,1.52],
        ["Kojic acid","DEF",142.11,3,4,77.76,1,-0.90],
        ["Picolinic Acid","DEF",123.11,1,3,50.19,1,0.54],
        ["Cis,Cis-Muconic Acid","DEF",142.11,2,4,74.60,3,0.12],
        ["3-Hydroxybenzoic Acid","DEF",138.12,2,3,57.53,1,1.88],
        ["4-Hydroxybenzoic acid","DEF",138.12,2,3,57.53,1,1.58],
        ["Resorcinol","DEF",110.11,2,2,40.46,0,0.80],
        ["Hydroquinone","DEF",110.11,2,2,40.46,0,0.59],
        ["Orcinol","DEF",124.14,2,2,40.46,1,1.25],
        ["2,3-Dihydroxybenzoic acid","DEF",154.12,3,4,77.76,1,0.93],
        ["Kynurenine","DEF",208.21,3,5,115.39,5,-1.62],
        ["Xanthurenic acid","DEF",205.17,3,5,92.11,1,0.63],
        ["Acetic Acid","SOL",60.05,1,2,37.30,1,-0.17],
        ["Citraconic acid","SOL",130.10,2,4,74.60,2,0.25],
        ["Glutaric Acid","SOL",132.12,2,4,74.60,3,-0.12],
        ["2,6-Pyridinedicarboxylic acid","SOL",167.12,2,5,90.42,1,-0.56],
        ["Citramalic acid","SOL",148.12,3,5,94.83,2,-0.25],
        ["Oxamic acid","SOL",89.05,2,4,80.12,1,-1.10],
        ["Geranic acid","SOL",168.23,1,2,37.30,5,2.85],
        ["D-Glyceric acid","SOL",106.08,3,4,77.76,2,-1.56],
        ["Galacturonic Acid","SOL",194.14,5,7,130.14,3,-1.90],
        ["Glycine","AMI",75.03,3,3,63.32,1,-3.21],
        ["N-Acetyl-alanine","AMI",131.13,2,4,80.12,3,-0.97],
        ["N-Acetyl-Glycine","AMI",117.10,2,4,80.12,2,-1.32],
        ["N-Acetyl-Histidine","AMI",197.19,3,5,112.72,4,-2.10],
        ["N-Acetyl-Proline","AMI",157.17,1,4,66.76,3,-0.50],
        ["N-Acetyl-Valine","AMI",159.18,2,4,80.12,4,-0.43],
        ["N-Acetyl-Leucine","AMI",173.21,2,4,80.12,5,0.08],
        ["1-Methylhistidine","AMI",169.18,3,4,99.13,3,-3.12],
        ["5-Aminolevulinic acid","AMI",131.13,3,4,103.55,4,-1.98],
        ["2-Oxoadipic acid","AMI",160.13,2,5,94.83,4,-0.32],
        ["N-Formylmethionine","AMI",177.22,2,4,88.66,5,-0.10],
        ["N-Acetyl-Tyrosine","AMI",223.23,3,5,103.55,5,-0.54],
        ["Uridine","POL",244.20,4,7,119.07,2,-1.44],
        ["Xanthine","POL",152.11,3,4,92.74,0,-1.22],
        ["D-Glucosamine","POL",179.17,5,6,119.07,2,-2.98],
        ["D-Mannosamine","POL",179.17,5,6,119.07,2,-2.98],
        ["Neuraminic acid","POL",309.27,7,9,174.34,5,-3.20],
        ["5'-Methylthioadenosine","POL",297.33,3,6,113.46,4,-0.12],
        ["Melamine","POL",126.12,6,6,134.78,0,-1.37],
        ["Guanidine","POL",59.07,3,3,93.09,0,-1.55],
        ["Ethanolamine","POL",61.08,3,2,52.49,2,-1.31],
        ["Succinyladenosine","POL",383.32,5,9,176.58,7,-2.10],
    ]
    df = pd.DataFrame(data, columns=['nombre','categoria','MW','HBD','HBA','TPSA','RotBonds','LogP'])
    return df

# ── MAPA ESTRESOR → CATEGORÍAS ──
STRESSOR_MAP = {
    "Sequía / Déficit hídrico": {
        "cats": ["OSM","FIT","AOX","COF"],
        "weights": {"OSM":1.0,"FIT":0.9,"AOX":0.7,"COF":0.5},
        "desc": "Requiere osmoprotección, regulación hormonal y protección oxidativa"
    },
    "Salinidad": {
        "cats": ["OSM","SOL","AMI","AOX"],
        "weights": {"OSM":1.0,"SOL":0.8,"AMI":0.7,"AOX":0.6},
        "desc": "Requiere osmoprotección, movilización de nutrientes y aminoácidos compatibles"
    },
    "Calor / Estrés térmico": {
        "cats": ["AOX","COF","OSM","FIT"],
        "weights": {"AOX":1.0,"COF":0.8,"OSM":0.7,"FIT":0.6},
        "desc": "Requiere captación de ROS, cofactores de reparación y señalización"
    },
    "Plaga / Insectos": {
        "cats": ["DEF","AOX","POL","FIT"],
        "weights": {"DEF":1.0,"AOX":0.8,"POL":0.7,"FIT":0.5},
        "desc": "Requiere activación de defensas y señales de reconocimiento"
    },
    "Hongo patógeno": {
        "cats": ["DEF","AOX","POL","SOL"],
        "weights": {"DEF":1.0,"AOX":0.8,"POL":0.6,"SOL":0.5},
        "desc": "Requiere compuestos antimicrobianos y activación de SAR"
    },
    "Deficiencia de nutrientes": {
        "cats": ["SOL","AMI","COF","FIT"],
        "weights": {"SOL":1.0,"AMI":0.8,"COF":0.7,"FIT":0.5},
        "desc": "Requiere solubilizadores, aminoácidos y cofactores metabólicos"
    },
    "Frío / Helada": {
        "cats": ["OSM","AOX","COF","FIT"],
        "weights": {"OSM":0.9,"AOX":1.0,"COF":0.7,"FIT":0.6},
        "desc": "Requiere antioxidantes, osmoprotectores y señalización de frío"
    },
}

CAT_LABELS = {
    "FIT":"Fitohormonas","COF":"Cofactores/Vitaminas","OSM":"Osmoprotectores",
    "AOX":"Antioxidantes","DEF":"Defensa","SOL":"Solubilizadores","AMI":"Aminoácidos","POL":"Señales/Polisac."
}
CAT_COLORS = {
    "FIT":"#1D9E75","COF":"#534AB7","OSM":"#0C447C","AOX":"#BA7517",
    "DEF":"#A32D2D","SOL":"#5DCAA5","AMI":"#854F0B","POL":"#3B8B6A"
}

def score_metabolites(df, stressors, top_n=8):
    scores = pd.Series(0.0, index=df.index)
    cats_covered = pd.Series([set() for _ in df.index], index=df.index)

    for s in stressors:
        info = STRESSOR_MAP[s]
        for cat, w in info["weights"].items():
            mask = df['categoria'] == cat
            scores[mask] += w
            for i in df[mask].index:
                cats_covered[i].add(cat)

    # Bonus diversidad: más categorías distintas cubiertas = mejor mezcla
    diversity_bonus = cats_covered.apply(len) * 0.15
    scores += diversity_bonus

    # Normalizar score
    if scores.max() > 0:
        scores = scores / scores.max()

    df = df.copy()
    df['score'] = scores.round(3)
    df['cats_covered'] = cats_covered.apply(lambda x: list(x))
    df['n_cats'] = cats_covered.apply(len)

    # Filtrar solo los que tienen score > 0
    df = df[df['score'] > 0].sort_values('score', ascending=False)

    # Seleccionar top_n maximizando diversidad de categorías
    selected = []
    used_cats = set()
    for _, row in df.iterrows():
        if len(selected) >= top_n:
            break
        selected.append(row)
        used_cats.update(row['cats_covered'])

    return pd.DataFrame(selected)

# ── HEADER ──
st.markdown("""
<div class="title-block">
  <p class="title-main">🍄 Mycovery</p>
  <p class="title-sub">Fungal metabolomics for next-gen bioinputs</p>
  <p class="tagline">Nature encoded it. We decoded it.</p>
</div>
""", unsafe_allow_html=True)

df = load_data()

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ⚙️ Define tu problema")

    cultivo = st.selectbox("🌿 Cultivo", [
        "Tomate","Vid","Trigo","Maíz","Arándano","Lechuga",
        "Papa","Cebolla","Pimiento","Otro"
    ])

    stressors = st.multiselect(
        "⚡ Estresores (uno o más)",
        list(STRESSOR_MAP.keys()),
        default=["Sequía / Déficit hídrico"]
    )

    objetivo = st.selectbox("🎯 Objetivo principal", [
        "Germinación","Desarrollo radicular","Rendimiento",
        "Tolerancia general","Defensa sistémica"
    ])

    top_n = st.slider("Número de metabolitos a recomendar", 3, 12, 6)

    st.markdown("---")
    st.markdown("**Restricciones de formulación**")
    max_mw = st.slider("MW máximo (Da)", 100, 600, 400)
    logp_range = st.slider("Rango LogP", -5.0, 5.0, (-3.0, 3.0))

    run = st.button("🔍 Recomendar mezcla", use_container_width=True, type="primary")

# ── MAIN ──
if not stressors:
    st.info("👈 Selecciona al menos un estresor en el panel izquierdo.")
    st.stop()

if run or True:
    # Filtrar por restricciones
    df_filtered = df[
        (df['MW'] <= max_mw) &
        (df['LogP'] >= logp_range[0]) &
        (df['LogP'] <= logp_range[1])
    ]

    results = score_metabolites(df_filtered, stressors, top_n)

    # Métricas resumen
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cultivo", cultivo)
    col2.metric("Estresores", len(stressors))
    col3.metric("Metabolitos recomendados", len(results))
    cats_total = set()
    for c in results['cats_covered']: cats_total.update(c)
    col4.metric("Categorías cubiertas", f"{len(cats_total)}/8")

    st.markdown("---")

    # Estresores seleccionados
    st.markdown('<p class="section-title">Estresores y categorías funcionales requeridas</p>', unsafe_allow_html=True)
    for s in stressors:
        info = STRESSOR_MAP[s]
        cats_str = " · ".join([f"**{CAT_LABELS[c]}**" for c in info['cats']])
        st.markdown(f"**{s}** → {cats_str}  \n_{info['desc']}_")

    st.markdown("---")
    st.markdown('<p class="section-title">Mezcla recomendada para ' + cultivo + '</p>', unsafe_allow_html=True)

    # Cards de resultados
    for i, (_, row) in enumerate(results.iterrows(), 1):
        score_pct = int(row['score'] * 100)
        bar = "█" * (score_pct // 10) + "░" * (10 - score_pct // 10)
        cats_html = "".join([
            f'<span class="badge" style="background:{CAT_COLORS.get(c,"#888")};color:white">{CAT_LABELS.get(c,c)}</span>'
            for c in row['cats_covered']
        ])
        st.markdown(f"""
        <div class="score-card">
          <h4>{i}. {row['nombre']}</h4>
          <p>{cats_html}</p>
          <p style="margin-top:6px">
            <strong>Score:</strong> {bar} {score_pct}% &nbsp;|&nbsp;
            <strong>MW:</strong> {row['MW']} Da &nbsp;|&nbsp;
            <strong>LogP:</strong> {row['LogP']} &nbsp;|&nbsp;
            <strong>Categoría:</strong> {CAT_LABELS.get(row['categoria'], row['categoria'])}
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Gráfico cobertura por categoría
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-title">Cobertura por categoría funcional</p>', unsafe_allow_html=True)
        cat_counts = {}
        for _, row in results.iterrows():
            for c in row['cats_covered']:
                cat_counts[c] = cat_counts.get(c, 0) + 1

        if cat_counts:
            fig, ax = plt.subplots(figsize=(5,3))
            cats = list(cat_counts.keys())
            vals = [cat_counts[c] for c in cats]
            colors = [CAT_COLORS.get(c, "#888") for c in cats]
            labels = [CAT_LABELS.get(c, c) for c in cats]
            bars = ax.barh(labels, vals, color=colors)
            ax.set_xlabel("N° metabolitos que cubren esta categoría")
            ax.spines[['top','right']].set_visible(False)
            ax.set_facecolor('#f5f4f0')
            fig.patch.set_facecolor('#f5f4f0')
            plt.tight_layout()
            st.pyplot(fig)

    with col_b:
        st.markdown('<p class="section-title">Score por metabolito recomendado</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5,3))
        nombres = [r['nombre'][:25] + '...' if len(r['nombre']) > 25 else r['nombre']
                   for _, r in results.iterrows()]
        scores_vals = [r['score'] for _, r in results.iterrows()]
        bar_colors = [CAT_COLORS.get(r['categoria'], '#888') for _, r in results.iterrows()]
        ax2.barh(nombres[::-1], scores_vals[::-1], color=bar_colors[::-1])
        ax2.set_xlabel("Score de relevancia")
        ax2.spines[['top','right']].set_visible(False)
        ax2.set_facecolor('#f5f4f0')
        fig2.patch.set_facecolor('#f5f4f0')
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")

    # Tabla exportable
    st.markdown('<p class="section-title">Tabla para laboratorio</p>', unsafe_allow_html=True)
    export = results[['nombre','categoria','score','MW','LogP','TPSA','HBD','HBA']].copy()
    export.columns = ['Metabolito','Categoría','Score','MW (Da)','LogP','TPSA','H-Bond Don.','H-Bond Ac.']
    export['Categoría'] = export['Categoría'].map(CAT_LABELS)
    export['Score'] = export['Score'].apply(lambda x: f"{x:.2f}")
    st.dataframe(export, use_container_width=True, hide_index=True)

    csv = export.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Descargar tabla CSV",
        csv,
        f"mycovery_{cultivo}_{'-'.join([s[:4] for s in stressors])}.csv",
        "text/csv",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown(
        "_Mycovery · Fungal metabolomics for next-gen bioinputs · "
        "Atacama Desert · Chile · 2025_",
        unsafe_allow_html=False
    )
