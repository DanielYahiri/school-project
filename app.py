# -*- coding: utf-8 -*-
"""
Application Streamlit de démonstration
Classification de l'occupation des sols — Yamoussoukro (Sentinel-2)
Projet de Deep Learning · Master 2 Data Science · UFHB

Modèle final : Random Forest (pipeline imblearn), déjà entraîné (non ré-entraîné ici).
Lancement : streamlit run app.py
"""

import base64
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

# =============================================================================
# 1. CONFIGURATION GÉNÉRALE
# =============================================================================
RACINE = Path(__file__).resolve().parent
CHEMIN_MODELE = RACINE / "resultats" / "modeles_finaux" / "random_forest_final.joblib"
CHEMIN_FICHE = RACINE / "resultats" / "modeles_finaux" / "fiche_modele_final.json"
CHEMIN_EXEMPLE = RACINE / "exemple_points.csv"
CHEMIN_BANNIERE = RACINE / "assets" / "banniere.jpg"

SEUIL_INCERTITUDE = 0.5  # en dessous : prédiction signalée « peu sûre »

# Palette de l'interface
VERT = "#1B5E3F"
OCRE = "#C8963E"

# =============================================================================
# 2. CLASSES : libellés français, couleurs fixes, icônes
#    (clé = nom anglais produit par le LabelEncoder, ordre alphabétique)
# =============================================================================
LIBELLES_FR = {
    "Bare Soil and Rock Outcrops": "Sol nu et affleurements",
    "Coffee and Cocoa": "Café et cacao",
    "Forest": "Forêt",
    "Human Infrastructure and Settlements": "Bâti et infrastructures",
    "Rubber and Other Crops": "Hévéa et autres cultures",
    "Savannas": "Savane",
    "Water Bodies": "Plans d'eau",
    "Wetlands": "Zones humides",
}

COULEURS_HEX = {
    "Hévéa et autres cultures": "#8FD175",  # vert clair
    "Café et cacao": "#8B5A2B",             # marron
    "Forêt": "#1E5631",                     # vert foncé
    "Savane": "#F2C94C",                    # jaune
    "Bâti et infrastructures": "#D7263D",   # rouge
    "Zones humides": "#2EC4B6",             # turquoise
    "Plans d'eau": "#1F5FBF",               # bleu
    "Sol nu et affleurements": "#9E9E9E",   # gris
}

ICONES = {
    "Hévéa et autres cultures": "🌱", "Café et cacao": "☕", "Forêt": "🌳",
    "Savane": "🌾", "Bâti et infrastructures": "🏘️", "Zones humides": "🪷",
    "Plans d'eau": "💧", "Sol nu et affleurements": "🪨",
}

# =============================================================================
# 3. RÉSULTATS DU PROJET (repris des fichiers de resultats/ : test, 5 graines)
# =============================================================================
SCORES_TEST = {"Macro-F1": 0.309, "Accuracy": 0.563, "MCC": 0.228, "Macro-F1 VC spatiale": 0.333}

# test_metriques_par_classe.csv + experience_E/distribution_classes.csv
PERF_PAR_CLASSE = pd.DataFrame({
    "Classe": ["Hévéa et autres cultures", "Savane", "Café et cacao", "Forêt",
               "Bâti et infrastructures", "Zones humides", "Plans d'eau", "Sol nu et affleurements"],
    "Précision": [0.69, 0.09, 0.34, 0.25, 0.80, 0.00, 0.68, 0.00],
    "Rappel":    [0.78, 0.06, 0.45, 0.22, 0.22, 0.00, 0.73, 0.00],
    "F1":        [0.73, 0.07, 0.39, 0.23, 0.35, 0.00, 0.70, 0.00],
    "Effectif total": [601, 102, 101, 78, 59, 27, 24, 8],
    "n test": [90, 16, 15, 11, 9, 4, 3, 2],
})

# test_matrice_confusion.png (normalisée par ligne, 5 graines cumulées)
ORDRE_MC = ["Sol nu et affleurements", "Café et cacao", "Forêt", "Bâti et infrastructures",
            "Hévéa et autres cultures", "Savane", "Plans d'eau", "Zones humides"]
MATRICE_CONFUSION = np.array([
    [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00, 0.00],
    [0.00, 0.45, 0.16, 0.00, 0.39, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.22, 0.00, 0.51, 0.27, 0.00, 0.00],
    [0.00, 0.04, 0.00, 0.22, 0.73, 0.00, 0.00, 0.00],
    [0.00, 0.13, 0.01, 0.00, 0.78, 0.06, 0.00, 0.01],
    [0.00, 0.05, 0.25, 0.00, 0.64, 0.06, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.73, 0.27],
    [0.00, 0.00, 0.00, 0.00, 0.50, 0.25, 0.25, 0.00],
])

# shap_importance_globale.csv
IMPORTANCE_SHAP = {
    "NDVI": 0.141, "NDRE": 0.1257, "EVI": 0.1216, "B3": 0.1185, "B10": 0.1079, "B6": 0.1052,
    "SAVI": 0.1047, "NDWI": 0.0945, "B4": 0.0926, "B9": 0.087, "B8": 0.0844, "MSI": 0.0806,
    "NDMI": 0.0793, "B5": 0.0761, "B7": 0.0689, "B1": 0.0643, "B2": 0.0619, "BSI": 0.0618, "BI": 0.0567,
}

# Un point réel par classe (tiré de donnees_propres.csv) pour pré-remplir la page « Prédire un point »
EXEMPLES_BANDES = {
    "Hévéa et autres cultures": [0.121, 0.1052, 0.093, 0.1113, 0.1595, 0.1813, 0.1743, 0.2051, 0.1837, 0.1002],
    "Café et cacao": [0.1223, 0.1072, 0.0926, 0.1164, 0.1765, 0.2052, 0.2001, 0.2319, 0.208, 0.1209],
    "Forêt": [0.1138, 0.0987, 0.081, 0.0997, 0.1627, 0.1958, 0.2025, 0.2209, 0.1608, 0.0875],
    "Savane": [0.1188, 0.1043, 0.0954, 0.1141, 0.1591, 0.1799, 0.1782, 0.2049, 0.2041, 0.124],
    "Bâti et infrastructures": [0.145, 0.1404, 0.1562, 0.1615, 0.1839, 0.1967, 0.1877, 0.2088, 0.2264, 0.1953],
    "Zones humides": [0.1126, 0.0907, 0.0738, 0.0831, 0.1118, 0.1265, 0.0888, 0.1297, 0.0759, 0.0408],
    "Plans d'eau": [0.1135, 0.0882, 0.0713, 0.0654, 0.0631, 0.0619, 0.0544, 0.0547, 0.0283, 0.0189],
    "Sol nu et affleurements": [0.1239, 0.1101, 0.1021, 0.1174, 0.1593, 0.179, 0.1804, 0.2004, 0.2118, 0.1508],
}

BANDES = {  # colonne : (libellé, groupe)
    "B1": ("B1 · aérosols", "Visible"), "B2": ("B2 · bleu", "Visible"),
    "B3": ("B3 · vert", "Visible"), "B4": ("B4 · rouge", "Visible"),
    "B5": ("B5 · RE 1", "Red-edge et proche infrarouge"),
    "B6": ("B6 · RE 2", "Red-edge et proche infrarouge"),
    "B7": ("B7 · RE 3", "Red-edge et proche infrarouge"),
    "B8": ("B8 · PIR", "Red-edge et proche infrarouge"),
    "B9": ("B9 · SWIR 1", "Infrarouge moyen (SWIR)"), "B10": ("B10 · SWIR 2", "Infrarouge moyen (SWIR)"),
}


# =============================================================================
# 4. STYLE (CSS) ET PETITS COMPOSANTS HTML
# =============================================================================
def injecter_style():
    """Feuille de style + blocage de la traduction automatique du navigateur
    (la traduction modifie la page et provoque l'erreur React « removeChild »)."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
    html, body, .stMarkdown, p, li, label {{ font-family: 'Inter', sans-serif; }}
    h1, h2, h3, h4, .titre-hero {{ font-family: 'Fraunces', Georgia, serif !important; color: {VERT}; }}
    .block-container {{ padding-top: 2rem; max-width: 1200px; }}
    [data-testid="stSidebar"] {{ background: {VERT}; }}
    [data-testid="stSidebar"] * {{ color: #F1EEE4 !important; }}
    [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.2); }}

    .hero {{ position: relative; border-radius: 18px; overflow: hidden; min-height: 230px;
             background-size: cover; background-position: center; margin-bottom: 1.5rem;
             box-shadow: 0 10px 30px rgba(27,94,63,.18); }}
    .hero::after {{ content: ""; position: absolute; inset: 0;
             background: linear-gradient(100deg, rgba(14,48,32,.93) 0%, rgba(14,48,32,.68) 55%, rgba(14,48,32,.2) 100%); }}
    .hero-contenu {{ position: relative; z-index: 1; padding: 2.2rem 2.4rem; color: #fff; max-width: 720px; }}
    .titre-hero {{ color: #fff !important; font-size: 2.3rem; line-height: 1.15; margin: .3rem 0 .6rem; }}
    .surtitre {{ text-transform: uppercase; letter-spacing: .14em; font-size: .75rem; color: {OCRE}; font-weight: 600; }}
    .hero p {{ color: #E8E4D8; font-size: 1.02rem; margin: 0; }}

    .carte {{ background: #fff; border: 1px solid #E6E1D3; border-radius: 14px; padding: 1.1rem 1.2rem;
              height: 100%; box-shadow: 0 2px 8px rgba(0,0,0,.04); }}
    .carte h4 {{ margin: .3rem 0 .4rem; font-size: 1.08rem; padding: 0; }}
    .carte p {{ margin: 0; color: #4A554E; font-size: .93rem; }}
    .pastille {{ display:inline-flex; align-items:center; justify-content:center; width: 28px; height: 28px;
                 border-radius: 50%; background: {VERT}; color: #fff; font-weight: 600; font-size: .85rem; }}

    .kpi {{ background: #fff; border-radius: 14px; padding: 1rem 1.1rem; border-left: 5px solid {OCRE};
            box-shadow: 0 2px 8px rgba(0,0,0,.05); min-height: 150px; }}
    .kpi .valeur {{ font-family: 'Fraunces', serif; font-size: 2rem; color: {VERT}; line-height: 1.05; }}
    .kpi .nom {{ font-weight: 600; color: #2B332E; margin-top: .35rem; }}
    .kpi .aide {{ color: #6B756F; font-size: .83rem; margin-top: .25rem; }}

    .resultat {{ border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: .8rem; }}
    .resultat .etiquette {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; opacity: .85; }}
    .resultat .classe {{ font-family: 'Fraunces', serif; font-size: 1.9rem; margin: .2rem 0; }}
    .jauge {{ border-radius: 99px; height: 10px; margin-top: .6rem; background: rgba(0,0,0,.15); }}
    .jauge > div {{ height: 10px; border-radius: 99px; }}

    .legende {{ display: flex; flex-wrap: wrap; gap: .4rem .9rem; font-size: .88rem; margin-top: .5rem; }}
    .legende .point {{ display:inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }}
    .note {{ background: #FFF8E6; border: 1px solid #F0DDA8; border-radius: 12px; padding: .8rem 1rem;
             color: #5C4A1A; font-size: .92rem; margin-top: .5rem; }}
    </style>
    """, unsafe_allow_html=True)

    # Empêche Chrome/Edge de traduire automatiquement la page
    script = """
    <script>
      const d = (window.parent && window.parent.document) || document;
      d.documentElement.setAttribute('translate', 'no');
      d.documentElement.setAttribute('lang', 'fr');
      if (!d.querySelector('meta[name="google"]')) {
        const m = d.createElement('meta'); m.name = 'google'; m.content = 'notranslate'; d.head.appendChild(m);
      }
    </script>"""
    try:
        st.html(script, unsafe_allow_javascript=True)  # Streamlit récent
    except TypeError:
        components.html(script, height=0)               # versions plus anciennes


@st.cache_data
def image_base64(chemin: str) -> str:
    """Encode une image pour l'utiliser en fond CSS (mise en cache)."""
    p = Path(chemin)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def bandeau(surtitre: str, titre: str, texte: str):
    """Grand bandeau d'en-tête avec l'image satellite de Yamoussoukro en fond."""
    fond = image_base64(str(CHEMIN_BANNIERE))
    style = f"background-image:url('data:image/jpeg;base64,{fond}');" if fond else f"background:{VERT};"
    st.markdown(f"""
    <div class="hero" style="{style}">
      <div class="hero-contenu">
        <div class="surtitre">{surtitre}</div>
        <div class="titre-hero">{titre}</div>
        <p>{texte}</p>
      </div>
    </div>""", unsafe_allow_html=True)


def carte(titre: str, texte: str, entete: str = ""):
    st.markdown(f'<div class="carte">{entete}<h4>{titre}</h4><p>{texte}</p></div>', unsafe_allow_html=True)


def kpi(valeur: str, nom: str, aide: str):
    st.markdown(f'<div class="kpi"><div class="valeur">{valeur}</div><div class="nom">{nom}</div>'
                f'<div class="aide">{aide}</div></div>', unsafe_allow_html=True)


def legende_classes():
    elements = "".join(f'<span><span class="point" style="background:{h}"></span>{n}</span>'
                       for n, h in COULEURS_HEX.items())
    st.markdown(f'<div class="legende">{elements}</div>', unsafe_allow_html=True)


def note(texte: str):
    st.markdown(f'<div class="note">{texte}</div>', unsafe_allow_html=True)


def echelle_couleurs():
    """Échelle Altair : couleur fixe pour chaque classe."""
    return alt.Scale(domain=list(COULEURS_HEX), range=list(COULEURS_HEX.values()))


def hex_vers_rgb(code_hex: str) -> list:
    code_hex = code_hex.lstrip("#")
    return [int(code_hex[i:i + 2], 16) for i in (0, 2, 4)]


def fmt(x: float, d: int = 3) -> str:
    """Nombre au format français (virgule décimale)."""
    return f"{x:.{d}f}".replace(".", ",")


# =============================================================================
# 5. MODÈLE : chargement (cache), indices, prédiction
# =============================================================================
@st.cache_resource(show_spinner="Chargement du modèle…")
def charger_modele():
    """Charge le dictionnaire {'pipeline', 'variables', 'classes'} sauvegardé avec joblib."""
    contenu = joblib.load(CHEMIN_MODELE)
    return contenu["pipeline"], list(contenu["variables"]), list(contenu["classes"])


@st.cache_data
def charger_fiche():
    return json.loads(CHEMIN_FICHE.read_text(encoding="utf-8")) if CHEMIN_FICHE.exists() else None


def calculer_indices(b: dict) -> dict:
    """Calcule les 9 indices à partir des 10 bandes (réflectances 0–1).
    Formules vérifiées sur les données d'entraînement :
      B2 = bleu, B3 = vert, B4 = rouge, B5 = red-edge 1, B8 = PIR, B9 = SWIR1 (B11 Sentinel-2)."""
    bleu, vert, rouge = b["B2"], b["B3"], b["B4"]
    red_edge, pir, swir1 = b["B5"], b["B8"], b["B9"]

    def nd(a, c):
        """Différence normalisée, protégée contre la division par zéro."""
        return (a - c) / (a + c) if (a + c) != 0 else np.nan

    return {
        "NDVI": nd(pir, rouge),
        # « + 7,5 × bleu » : formule présente dans les données d'entraînement
        # (la version standard est « − 7,5 × bleu »). On la reproduit pour rester cohérent avec le modèle.
        "EVI": 2.5 * (pir - rouge) / (pir + 6 * rouge + 7.5 * bleu + 1),
        "NDRE": nd(pir, red_edge),
        "SAVI": 1.5 * (pir - rouge) / (pir + rouge + 0.5),
        "NDMI": nd(pir, swir1),
        "NDWI": nd(vert, pir),
        "MSI": swir1 / pir if pir != 0 else np.nan,
        "BSI": nd(swir1 + rouge, pir + bleu),
        "BI": np.sqrt((bleu**2 + vert**2 + rouge**2 + pir**2) / 4),
    }


def predire(pipeline, classes, X: pd.DataFrame) -> pd.DataFrame:
    """Classe prédite (FR), confiance (= probabilité max), alerte et probabilités par classe."""
    # Le modèle a été entraîné sur un tableau numpy : valeurs passées dans l'ordre exact des variables.
    # Le RandomOverSampler n'intervient pas à la prédiction.
    probas = pipeline.predict_proba(X.to_numpy(dtype=float))
    noms_fr = [LIBELLES_FR[c] for c in classes]
    idx = probas.argmax(axis=1)

    res = pd.DataFrame(index=X.index)
    res["classe_predite"] = [noms_fr[i] for i in idx]
    res["confiance"] = probas.max(axis=1).round(3)
    res["prediction_peu_sure"] = res["confiance"] < SEUIL_INCERTITUDE
    df_probas = pd.DataFrame(probas, columns=[f"P({n})" for n in noms_fr], index=X.index).round(3)
    return pd.concat([res, df_probas], axis=1)


# =============================================================================
# 6. GRAPHIQUES
# =============================================================================
def barres_classes(df: pd.DataFrame, x: str, titre_x=None, domaine=None, format_x=None, hauteur=300):
    """Barres horizontales colorées par classe (gabarit commun aux graphiques)."""
    axe_x = alt.Axis(format=format_x) if format_x else alt.Axis()
    echelle_x = alt.Scale(domain=domaine) if domaine else alt.Scale()
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(f"{x}:Q", title=titre_x or x, scale=echelle_x, axis=axe_x),
        y=alt.Y("Classe:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("Classe:N", scale=echelle_couleurs(), legend=None),
        tooltip=list(df.columns),
    ).properties(height=hauteur)


def graphique_probas(probas: pd.Series) -> alt.Chart:
    df = pd.DataFrame({"Classe": [c[2:-1] for c in probas.index], "Probabilité": probas.values.astype(float)})
    ordre = df.sort_values("Probabilité", ascending=False)["Classe"].tolist()  # même ordre pour les 2 couches
    barres = barres_classes(df, "Probabilité", domaine=[0, 1], format_x="%").encode(
        y=alt.Y("Classe:N", sort=ordre, title=None, axis=alt.Axis(labelLimit=220)))
    texte = alt.Chart(df).mark_text(align="left", dx=4, color="#333").encode(
        x="Probabilité:Q", y=alt.Y("Classe:N", sort=ordre), text=alt.Text("Probabilité:Q", format=".0%"))
    seuil = alt.Chart(pd.DataFrame({"x": [SEUIL_INCERTITUDE]})).mark_rule(
        strokeDash=[4, 4], color="#888").encode(x="x:Q")
    return barres + texte + seuil


def graphique_confiance(resultats: pd.DataFrame) -> alt.Chart:
    return alt.Chart(resultats).mark_bar(color=VERT, opacity=.85).encode(
        x=alt.X("confiance:Q", bin=alt.Bin(step=0.1), title="Confiance (probabilité max)", axis=alt.Axis(format="%")),
        y=alt.Y("count():Q", title="Nombre de points"),
    ).properties(height=300)


# =============================================================================
# 7. CARTE
# =============================================================================
def extraire_coordonnees(df: pd.DataFrame):
    """(lon, lat) directement, ou conversion de x/y depuis UTM 30N (EPSG:32630). None sinon."""
    cols = {c.lower(): c for c in df.columns}
    if "lon" in cols and "lat" in cols:
        return pd.DataFrame({"lon": pd.to_numeric(df[cols["lon"]], errors="coerce"),
                             "lat": pd.to_numeric(df[cols["lat"]], errors="coerce")}, index=df.index)
    if "x" in cols and "y" in cols:
        from pyproj import Transformer  # utile seulement dans ce cas
        t = Transformer.from_crs("EPSG:32630", "EPSG:4326", always_xy=True)
        lon, lat = t.transform(pd.to_numeric(df[cols["x"]], errors="coerce").values,
                               pd.to_numeric(df[cols["y"]], errors="coerce").values)
        return pd.DataFrame({"lon": lon, "lat": lat}, index=df.index)
    return None


def afficher_carte(coords: pd.DataFrame, resultats: pd.DataFrame):
    df = coords.join(resultats[["classe_predite", "confiance"]]).dropna(subset=["lon", "lat"])
    df["couleur"] = df["classe_predite"].map(lambda c: hex_vers_rgb(COULEURS_HEX[c]) + [230])
    df["confiance_txt"] = (df["confiance"] * 100).round(0).astype(int).astype(str) + " %"

    couche = pdk.Layer(
        "ScatterplotLayer", data=df, get_position="[lon, lat]", get_fill_color="couleur",
        get_line_color=[255, 255, 255], line_width_min_pixels=1, stroked=True,
        get_radius=150, radius_min_pixels=5, radius_max_pixels=14, pickable=True,
    )
    vue = pdk.ViewState(latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()), zoom=10)
    st.pydeck_chart(pdk.Deck(
        layers=[couche], initial_view_state=vue,
        map_style=pdk.map_styles.CARTO_LIGHT,  # fond de carte gratuit, sans clé API
        tooltip={"html": "<b>{classe_predite}</b><br/>Confiance : {confiance_txt}"},
    ))
    legende_classes()


# =============================================================================
# 8. PAGES
# =============================================================================
def page_accueil():
    bandeau("Projet de Deep Learning · M2 Data Science · UFHB",
            "Que recouvre le sol autour de Yamoussoukro ?",
            "Une démonstration qui prédit l'occupation du sol (forêt, savane, cultures, bâti, eau…) "
            "à partir des mesures du satellite Sentinel-2.")

    st.markdown("### En bref")
    c1, c2, c3 = st.columns(3)
    with c1:
        carte("🛰️ Les données", "1 000 points au sol autour de Yamoussoukro. Pour chacun : 10 mesures "
              "Sentinel-2 et 9 indices calculés (NDVI, EVI…).")
    with c2:
        carte("🌲 Le modèle", "Une forêt aléatoire de 530 arbres de décision. Chaque arbre vote ; "
              "la classe qui recueille le plus de voix l'emporte.")
    with c3:
        carte("🗺️ Le résultat", "Pour chaque point : la classe la plus probable et la probabilité "
              "de chacune des 8 classes, avec le niveau de confiance.")

    st.markdown("### Comment utiliser l'application")
    e1, e2, e3 = st.columns(3)
    with e1:
        carte("Comprendre le modèle", "Les scores, les classes bien ou mal reconnues, "
              "les variables les plus utiles.", '<span class="pastille">1</span>')
    with e2:
        carte("Prédire un fichier", "Utilisez l'exemple fourni ou importez un CSV : carte, "
              "graphiques et export des résultats.", '<span class="pastille">2</span>')
    with e3:
        carte("Prédire un point", "Choisissez un exemple ou saisissez 10 valeurs : "
              "la prédiction s'affiche immédiatement.", '<span class="pastille">3</span>')
    st.caption("👈 Utilisez le menu à gauche pour naviguer.")

    st.markdown("### Les 8 classes")
    cols = st.columns(4)
    for i, (nom, couleur) in enumerate(COULEURS_HEX.items()):
        cols[i % 4].markdown(
            f'<div class="carte" style="border-top:5px solid {couleur}; margin-bottom:.8rem; padding:.8rem 1rem;">'
            f'<h4 style="margin:0">{ICONES[nom]} {nom}</h4></div>', unsafe_allow_html=True)

    note("⚠️ <b>Démonstration pédagogique, pas un outil opérationnel.</b> Le modèle classe correctement "
         "environ un point sur deux et peine sur les classes rares. Il n'est valable que pour la zone "
         "de Yamoussoukro et pour des données préparées exactement comme celles du projet.")


def page_modele():
    st.markdown("## 📊 Comprendre le modèle")
    st.write("Performances mesurées sur des données **jamais vues** pendant l'entraînement "
             "(150 points de test, moyenne sur 5 graines).")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi(fmt(SCORES_TEST["Accuracy"]), "Accuracy", "≈ 56 points sur 100 sont bien classés.")
    with k2:
        kpi(fmt(SCORES_TEST["Macro-F1"]), "Macro-F1", "Moyenne du F1 des 8 classes : chaque classe compte autant.")
    with k3:
        kpi(fmt(SCORES_TEST["MCC"]), "MCC", "0 = hasard, 1 = parfait. Robuste au déséquilibre.")
    with k4:
        kpi(fmt(SCORES_TEST["Macro-F1 VC spatiale"]), "Macro-F1 spatial",
            "Validation par blocs géographiques : test plus réaliste.")

    st.markdown("")
    onglets = st.tabs(["🎯 Par classe", "🔀 Confusions", "⚖️ Déséquilibre", "🔎 Variables utiles", "🧮 Indices"])

    with onglets[0]:  # --- Performance par classe
        st.write("Le **F1** combine précision et rappel : 0 = classe jamais reconnue, 1 = toujours reconnue.")
        st.altair_chart(barres_classes(PERF_PAR_CLASSE[["Classe", "F1", "Précision", "Rappel", "n test"]],
                                       "F1", domaine=[0, 1]), width="stretch")
        note("<b>Hévéa</b> et <b>Plans d'eau</b> sont bien reconnues. <b>Zones humides</b> et <b>Sol nu</b> "
             "ne le sont jamais : elles n'ont que 4 et 2 points de test.")

    with onglets[1]:  # --- Matrice de confusion
        st.write("Pour chaque **vraie classe** (ligne) : part des points attribués à chaque "
                 "**classe prédite** (colonne). La diagonale correspond aux bonnes réponses.")
        mc = pd.DataFrame(MATRICE_CONFUSION, index=ORDRE_MC, columns=ORDRE_MC).reset_index(names="Vraie classe")
        mc = mc.melt(id_vars="Vraie classe", var_name="Classe prédite", value_name="Part")
        base = alt.Chart(mc).encode(
            x=alt.X("Classe prédite:N", sort=ORDRE_MC, axis=alt.Axis(labelAngle=-35, labelLimit=160)),
            y=alt.Y("Vraie classe:N", sort=ORDRE_MC, axis=alt.Axis(labelLimit=200)),
        )
        cases = base.mark_rect().encode(
            color=alt.Color("Part:Q", scale=alt.Scale(scheme="greens", domain=[0, 1]), legend=None),
            tooltip=["Vraie classe", "Classe prédite", alt.Tooltip("Part:Q", format=".0%")])
        txt = base.transform_filter("datum.Part > 0").mark_text(fontSize=12).encode(
            text=alt.Text("Part:Q", format=".0%"),
            color=alt.condition("datum.Part > 0.5", alt.value("white"), alt.value("#333")))
        st.altair_chart((cases + txt).properties(height=430), width="stretch")
        note("La colonne <b>Hévéa et autres cultures</b> concentre la plupart des erreurs : c'est la classe "
             "majoritaire (60 % des données), le modèle s'y replie quand il hésite.")

    with onglets[2]:  # --- Déséquilibre des classes
        st.write("Nombre de points par classe dans l'ensemble des données (1 000 points).")
        st.altair_chart(barres_classes(PERF_PAR_CLASSE[["Classe", "Effectif total"]], "Effectif total",
                                       titre_x="Nombre de points"), width="stretch")
        st.write("Pour compenser, les classes rares ont été **suréchantillonnées** pendant l'entraînement "
                 "(RandomOverSampler). Cela ne crée pas d'information nouvelle quand une classe n'a que 8 points.")

    with onglets[3]:  # --- Importance des variables (SHAP)
        st.write("Importance moyenne (valeurs SHAP) : plus la barre est longue, plus la variable pèse dans les décisions.")
        imp = pd.DataFrame({"Variable": list(IMPORTANCE_SHAP), "Importance": list(IMPORTANCE_SHAP.values())})
        imp["Type"] = np.where(imp["Variable"].str.match(r"^B\d"), "Bande", "Indice")
        g = alt.Chart(imp).mark_bar(cornerRadiusEnd=4).encode(
            x="Importance:Q", y=alt.Y("Variable:N", sort="-x", title=None),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["Indice", "Bande"], range=[VERT, OCRE]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Variable", "Type", "Importance"]).properties(height=430)
        st.altair_chart(g, width="stretch")
        st.write("Les indices de végétation (**NDVI, NDRE, EVI**) sont les variables les plus utiles.")

    with onglets[4]:  # --- Formules des indices
        st.write("Correspondance des colonnes : B2 bleu, B3 vert, B4 rouge, B5 red-edge, "
                 "B8 proche infrarouge (PIR), B9 SWIR 1, B10 SWIR 2.")
        st.dataframe(pd.DataFrame({
            "Indice": ["NDVI", "EVI", "NDRE", "SAVI", "NDMI", "NDWI", "MSI", "BSI", "BI"],
            "Formule": ["(B8 − B4) / (B8 + B4)", "2,5 (B8 − B4) / (B8 + 6·B4 + 7,5·B2 + 1) *",
                        "(B8 − B5) / (B8 + B5)", "1,5 (B8 − B4) / (B8 + B4 + 0,5)", "(B8 − B9) / (B8 + B9)",
                        "(B3 − B8) / (B3 + B8)", "B9 / B8", "((B9 + B4) − (B8 + B2)) / ((B9 + B4) + (B8 + B2))",
                        "√((B2² + B3² + B4² + B8²) / 4)"],
            "Ce qu'il mesure": ["Vigueur de la végétation", "Végétation dense", "Chlorophylle (red-edge)",
                                "Végétation, corrigée du sol", "Humidité de la végétation", "Présence d'eau",
                                "Stress hydrique", "Sol nu", "Brillance du sol"],
        }), hide_index=True, width="stretch")
        st.caption("* Formule telle que présente dans les données du projet (la version standard utilise − 7,5·B2).")

    fiche = charger_fiche()
    if fiche:
        with st.expander("Fiche technique du modèle (JSON)"):
            st.json(fiche)


def page_fichier(pipeline, variables, classes):
    st.markdown("## 📄 Prédire un fichier")
    st.write("Chaque ligne du fichier est un point au sol ; le modèle prédit sa classe d'occupation du sol.")

    # --- Étape 1 : choix des données
    st.markdown("#### 1. Choisir les données")
    source = st.radio("Source", ["Utiliser le fichier d'exemple (48 points)", "Importer mon propre CSV"],
                      horizontal=True, label_visibility="collapsed")

    df = None
    if source.startswith("Utiliser"):
        if CHEMIN_EXEMPLE.exists():
            df = pd.read_csv(CHEMIN_EXEMPLE)
        else:
            st.error("Fichier d'exemple introuvable (exemple_points.csv).")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            fichier = st.file_uploader("Importer un CSV", type="csv", label_visibility="collapsed")
        with c2:
            modele_csv = pd.DataFrame(columns=variables + ["lon", "lat"]).to_csv(index=False)
            st.download_button("📋 Modèle de CSV vide", modele_csv, "modele_colonnes.csv", "text/csv",
                               width="stretch")
        st.caption(f"Colonnes obligatoires ({len(variables)}) : {', '.join(variables)}. "
                   "Facultatif : `lon`/`lat` ou `x`/`y` (EPSG:32630) pour la carte. Les autres colonnes sont ignorées.")
        if fichier is not None:
            try:
                df = pd.read_csv(fichier, sep=None, engine="python")  # virgule ou point-virgule
            except Exception as e:
                st.error(f"Impossible de lire le fichier : {e}")

    if df is None:
        return

    # --- Vérification des colonnes
    manquantes = [v for v in variables if v not in df.columns]
    if manquantes:
        st.error(f"**{len(manquantes)} colonne(s) manquante(s)** : {', '.join(manquantes)}")
        st.info("Téléchargez le « Modèle de CSV vide » pour voir les noms de colonnes attendus.")
        return

    X = df[variables].apply(pd.to_numeric, errors="coerce")
    invalides = X.isna().any(axis=1)
    if invalides.any():
        st.warning(f"{int(invalides.sum())} ligne(s) avec valeurs manquantes ou non numériques ignorée(s).")
    X = X[~invalides]
    if X.empty:
        st.error("Aucune ligne exploitable.")
        return

    resultats = predire(pipeline, classes, X)
    n, n_doute = len(resultats), int(resultats["prediction_peu_sure"].sum())
    classe_maj = resultats["classe_predite"].mode()[0]

    # --- Étape 2 : résultats
    st.markdown("#### 2. Résultats")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi(str(n), "Points analysés", "Lignes valides du fichier.")
    with k2:
        kpi(f"{resultats['confiance'].mean():.0%}", "Confiance moyenne", "Probabilité moyenne de la classe retenue.")
    with k3:
        kpi(str(n_doute), "Prédictions peu sûres", f"Confiance < {SEUIL_INCERTITUDE:.0%} : à vérifier.")
    with k4:
        kpi(ICONES[classe_maj], "Classe la plus fréquente", classe_maj)

    st.markdown("")
    coords = extraire_coordonnees(df.loc[resultats.index])
    noms_onglets = (["🗺️ Carte"] if coords is not None else []) + ["📊 Répartition", "📋 Tableau détaillé"]
    onglets = dict(zip(noms_onglets, st.tabs(noms_onglets)))

    if coords is not None:
        with onglets["🗺️ Carte"]:
            st.caption("Survolez un point pour voir sa classe prédite et la confiance du modèle.")
            afficher_carte(coords, resultats)

    with onglets["📊 Répartition"]:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Classes prédites**")
            rep = resultats["classe_predite"].value_counts().rename_axis("Classe").reset_index(name="Points")
            st.altair_chart(barres_classes(rep, "Points", titre_x="Nombre de points"), width="stretch")
        with g2:
            st.markdown("**Niveau de confiance**")
            st.altair_chart(graphique_confiance(resultats), width="stretch")

    with onglets["📋 Tableau détaillé"]:
        filtre = st.toggle("Afficher seulement les prédictions peu sûres", value=False)
        vue = resultats[resultats["prediction_peu_sure"]] if filtre else resultats
        cols_probas = [c for c in vue.columns if c.startswith("P(")]
        st.dataframe(
            vue, width="stretch", height=400,
            column_config={
                "classe_predite": st.column_config.TextColumn("Classe prédite"),
                "confiance": st.column_config.ProgressColumn("Confiance", min_value=0, max_value=1, format="percent"),
                "prediction_peu_sure": st.column_config.CheckboxColumn("Peu sûre ?"),
                **{c: st.column_config.NumberColumn(c[2:-1], format="percent") for c in cols_probas},
            },
        )

    export = df.loc[resultats.index].join(resultats)
    st.download_button("⬇️ Télécharger les résultats (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                       "predictions_occupation_sols.csv", "text/csv", type="primary")
    if coords is None:
        st.info("Ajoutez des colonnes `lon`/`lat` (ou `x`/`y` en EPSG:32630) pour afficher une carte.")


def page_point(pipeline, variables, classes):
    st.markdown("## 🎛️ Prédire un point")
    st.write("Saisissez les 10 mesures Sentinel-2 d'un point (réflectances entre 0 et 1). "
             "Les 9 indices sont calculés automatiquement, puis le modèle prédit la classe.")

    # --- Pré-remplissage avec un exemple réel
    choix = st.selectbox("Partir d'un exemple réel du jeu de données", list(EXEMPLES_BANDES),
                         format_func=lambda c: f"{ICONES[c]} {c}")
    st.caption("Ces exemples font partie des données d'entraînement : le modèle les a déjà vus, "
               "d'où une confiance très élevée. Modifiez les valeurs pour observer l'effet.")
    if st.session_state.get("exemple_charge") != choix:  # nouvel exemple choisi : on remplit les champs
        for i, v in enumerate(EXEMPLES_BANDES[choix]):
            st.session_state[f"B{i + 1}"] = v
        st.session_state["exemple_charge"] = choix

    gauche, droite = st.columns([1.05, 1], gap="large")

    with gauche:  # --- Saisie des bandes, regroupées par domaine spectral
        bandes = {}
        for groupe in ["Visible", "Red-edge et proche infrarouge", "Infrarouge moyen (SWIR)"]:
            st.markdown(f"**{groupe}**" + (" · RE = red-edge" if groupe.startswith("Red") else ""))
            noms = [b for b, (_, g) in BANDES.items() if g == groupe]
            for col, b in zip(st.columns(len(noms)), noms):
                bandes[b] = col.number_input(BANDES[b][0], min_value=0.0, max_value=1.0,
                                             step=0.001, format="%.4f", key=b)
        indices = calculer_indices(bandes)
        with st.expander("Voir les 9 indices calculés"):
            st.dataframe(pd.DataFrame([indices]).round(4), hide_index=True, width="stretch")

    with droite:  # --- Résultat
        if any(pd.isna(v) for v in indices.values()):
            st.error("Un indice n'est pas calculable (division par zéro) : vérifiez les valeurs saisies.")
            return

        X = pd.DataFrame([{**bandes, **indices}])[variables]  # ordre exact attendu par le modèle
        res = predire(pipeline, classes, X).iloc[0]
        classe, conf = res["classe_predite"], float(res["confiance"])
        fond = COULEURS_HEX[classe]
        encre = "#1F2A24" if classe in ("Savane", "Hévéa et autres cultures", "Sol nu et affleurements") else "#FFFFFF"

        st.markdown(f"""
        <div class="resultat" style="background:{fond}; color:{encre};">
          <div class="etiquette">Classe prédite</div>
          <div class="classe">{ICONES[classe]} {classe}</div>
          <div>Confiance : <b>{conf:.0%}</b></div>
          <div class="jauge"><div style="width:{conf * 100:.0f}%; background:{encre};"></div></div>
        </div>""", unsafe_allow_html=True)

        if conf < SEUIL_INCERTITUDE:
            st.warning(f"⚠️ **Prédiction peu sûre** : aucune classe ne dépasse {SEUIL_INCERTITUDE:.0%}, "
                       "le modèle hésite entre plusieurs classes.")
        elif conf < 0.7:
            st.info("Confiance moyenne : cette classe est probable, mais d'autres restent possibles.")
        else:
            st.success("Confiance élevée.")

        st.markdown("**Probabilité de chaque classe** (pointillés = seuil de 50 %)")
        st.altair_chart(graphique_probas(res[[c for c in res.index if c.startswith("P(")]]), width="stretch")


# =============================================================================
# 9. POINT D'ENTRÉE
# =============================================================================
def main():
    st.set_page_config(page_title="Occupation des sols · Yamoussoukro", page_icon="🛰️", layout="wide")
    injecter_style()

    if not CHEMIN_MODELE.exists():
        st.error(f"Modèle introuvable : `{CHEMIN_MODELE.relative_to(RACINE)}`")
        st.stop()
    pipeline, variables, classes = charger_modele()

    with st.sidebar:
        st.markdown("## 🛰️ Occupation des sols")
        st.caption("Yamoussoukro · Sentinel-2")
        st.markdown("---")
        page = st.radio("Navigation", ["🏠 Accueil", "📊 Comprendre le modèle",
                                       "📄 Prédire un fichier", "🎛️ Prédire un point"],
                        label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**Modèle** : Random Forest  \n530 arbres · 19 variables · 8 classes")
        st.markdown(f"**Accuracy test** : {fmt(SCORES_TEST['Accuracy'])}  \n"
                    f"**Macro-F1 test** : {fmt(SCORES_TEST['Macro-F1'])}")
        st.markdown("---")
        st.caption("Démonstration pédagogique — M2 Data Science, UFHB. "
                   "Ne pas utiliser hors de la zone de Yamoussoukro.")

    if page.endswith("Accueil"):
        page_accueil()
    elif page.endswith("modèle"):
        page_modele()
    elif page.endswith("fichier"):
        page_fichier(pipeline, variables, classes)
    else:
        page_point(pipeline, variables, classes)


if __name__ == "__main__":
    main()
