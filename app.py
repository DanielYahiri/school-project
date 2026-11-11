# -*- coding: utf-8 -*-
"""
Application Streamlit de démonstration
Classification de l'occupation des sols — Yamoussoukro (Sentinel-2)
Projet de Deep Learning · Master 2 Data Science · UFHB

Deux modèles, déjà entraînés (non ré-entraînés ici) :
  - Random Forest (modèle retenu) : resultats/modeles_finaux/random_forest_final.joblib
  - ANN 128-64-32 (comparaison)    : resultats/modeles_finaux/ann_final_poids.npz

Lancement : streamlit run app.py
"""

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
DOSSIER_MODELES = RACINE / "resultats" / "modeles_finaux"
CHEMIN_MODELE = DOSSIER_MODELES / "random_forest_final.joblib"
CHEMIN_FICHE = DOSSIER_MODELES / "fiche_modele_final.json"
CHEMIN_ANN = DOSSIER_MODELES / "ann_final_poids.npz"  # créé par scripts/sauvegarder_ann.py
CHEMIN_EXEMPLE = RACINE / "exemple_points.csv"
CHEMIN_BANNIERE = RACINE / "assets" / "banniere.jpg"
CHEMIN_LOGO = RACINE / "assets" / "logo.svg"

SEUIL_INCERTITUDE = 0.5  # en dessous : prédiction signalée « peu sûre »

# Palette de l'interface (sobre : la couleur ne sert qu'à l'essentiel)
VERT = "#1F5135"         # couleur principale + Random Forest
VERT_CLAIR = "#EEF4F0"
OCRE = "#D08C2E"         # ANN
TEXTE = "#111827"
TEXTE_2 = "#4B5563"
TEXTE_3 = "#6B7280"
BORDURE = "#E5E7EB"

# =============================================================================
# 2. CLASSES : libellés français et couleurs fixes
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

PART_CLASSES = {  # part de chaque classe dans les 1 000 points (distribution_classes.csv)
    "Hévéa et autres cultures": 60, "Café et cacao": 10, "Forêt": 8, "Savane": 10,
    "Bâti et infrastructures": 6, "Zones humides": 3, "Plans d'eau": 2, "Sol nu et affleurements": 1,
}

# =============================================================================
# 3. RÉSULTATS DU PROJET (repris des fichiers de resultats/)
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

# --- Comparaison ANN / Random Forest -----------------------------------------
# Validation (5 graines) et VC : modeles_classiques/comparaison/comparaison_finale.csv
# Test (5 graines) : RF = fiche_modele_final.json ; ANN = scripts/sauvegarder_ann.py (reconstruit à l'identique)
# Chaque score : (métrique, Random Forest, ANN)
ETAPES = {
    "Validation": {"sous_titre": "Sert à choisir le modèle", "badge": ("RF devant", "rf"),
                   "scores": [("Macro-F1", 0.450, 0.389), ("Accuracy", 0.623, 0.645), ("MCC", 0.353, 0.330)]},
    "Validation spatiale": {"sous_titre": "Blocs géographiques", "badge": ("Chevauchement", "neutre"),
                            "scores": [("Macro-F1", 0.333, 0.288)]},
    "Test": {"sous_titre": "Données jamais vues", "badge": ("ANN devant", "ann"),
             "scores": [("Macro-F1", 0.309, 0.412), ("Accuracy", 0.563, 0.629), ("MCC", 0.228, 0.304)]},
}
ECART_TRAIN_VAL = {"ANN": 0.054, "Random Forest": 0.550}   # surapprentissage (F1 train − F1 val)
TEMPS_ENTRAINEMENT = {"ANN": 35.2, "Random Forest": 3.4}  # secondes par graine

# F1 par classe sur le test, modèles de la graine 0 (ceux utilisés dans l'application) : (RF, ANN)
F1_TEST_CLASSES = {
    "Hévéa et autres cultures": (0.74, 0.76), "Café et cacao": (0.40, 0.35), "Forêt": (0.20, 0.36),
    "Bâti et infrastructures": (0.33, 0.33), "Savane": (0.08, 0.11), "Plans d'eau": (0.67, 1.00),
    "Zones humides": (0.00, 0.40), "Sol nu et affleurements": (0.00, 0.00),
}

COULEURS_MODELES = {"Random Forest": VERT, "ANN": OCRE}

# Un point réel par classe (donnees_propres.csv), bien classé par les DEUX modèles.
# Sol nu : aucun point n'est reconnu par les deux ; on garde un point reconnu par le Random Forest.
EXEMPLES_BANDES = {
    "Hévéa et autres cultures": [0.1207, 0.1109, 0.1064, 0.1224, 0.1842, 0.2104, 0.2048, 0.2352, 0.2208, 0.1417],
    "Café et cacao": [0.1158, 0.1042, 0.0839, 0.1071, 0.2099, 0.2596, 0.2501, 0.2894, 0.1621, 0.0741],
    "Forêt": [0.1135, 0.0963, 0.0777, 0.0944, 0.1643, 0.1975, 0.2079, 0.2201, 0.1466, 0.0696],
    "Savane": [0.1132, 0.0953, 0.0801, 0.0975, 0.1349, 0.1549, 0.1554, 0.1703, 0.1538, 0.0893],
    "Bâti et infrastructures": [0.1298, 0.1172, 0.1217, 0.1372, 0.1772, 0.1958, 0.1879, 0.2036, 0.2215, 0.1707],
    "Zones humides": [0.1122, 0.0895, 0.0695, 0.0701, 0.0764, 0.0792, 0.0736, 0.0768, 0.0428, 0.0242],
    "Plans d'eau": [0.1134, 0.0877, 0.0694, 0.0649, 0.062, 0.0615, 0.0527, 0.0557, 0.027, 0.0185],
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
# 4. ICÔNES AU TRAIT (SVG) ET STYLE
# =============================================================================
ICONES_SVG = {
    "couches": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    "base": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
            '<path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "grille": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>'
              '<rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    "balance": '<path d="M12 3v18"/><path d="M5 7h14"/><path d="m5 7-3 7a3.5 3.5 0 0 0 6 0z"/>'
               '<path d="m19 7-3 7a3.5 3.5 0 0 0 6 0z"/><path d="M8 21h8"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7.5h.01"/>',
    "alerte": '<path d="M12 3 2 21h20z"/><path d="M12 10v5"/><path d="M12 18h.01"/>',
    "valide": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
}


def svg(nom: str, taille: int = 20, couleur: str = "currentColor", largeur: float = 1.8) -> str:
    """Icône au trait (style « line icons ») en SVG inline."""
    return (f'<svg width="{taille}" height="{taille}" viewBox="0 0 24 24" fill="none" stroke="{couleur}" '
            f'stroke-width="{largeur}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{ICONES_SVG[nom]}</svg>')


def injecter_style():
    """Feuille de style + blocage de la traduction automatique du navigateur
    (la traduction modifie la page et provoque l'erreur React « removeChild »)."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, .stMarkdown, p, li, label, button, input {{ font-family: 'Inter', system-ui, sans-serif; }}
    h1, h2, h3 {{ font-family: 'Inter', system-ui, sans-serif !important; color: {TEXTE}; letter-spacing: -0.02em; }}
    .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}

    /* Barre latérale blanche et navigation sobre */
    [data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {BORDURE}; }}
    [data-testid="stNavSectionHeader"] {{ font-size: 11px !important; font-weight: 600; letter-spacing: .08em;
        text-transform: uppercase; color: {TEXTE_3} !important; }}
    [data-testid="stSidebarNavLink"] {{ border-radius: 8px; }}
    [data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {VERT_CLAIR}; }}
    [data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: {VERT} !important; font-weight: 600; }}

    /* Composants maison */
    .entete h1 {{ font-size: 28px; font-weight: 700; margin: 0 0 6px; padding: 0; }}
    .entete p {{ font-size: 15px; color: {TEXTE_2}; margin: 0 0 8px; }}
    .titre-section {{ font-size: 17px; font-weight: 600; color: {TEXTE}; margin: 22px 0 10px; }}
    .sous-titre-section {{ font-size: 13px; color: {TEXTE_3}; margin: -6px 0 12px; }}
    .carte {{ background: #FFFFFF; border: 1px solid {BORDURE}; border-radius: 12px; padding: 18px 20px;
              height: 100%; box-sizing: border-box; }}
    .chiffre {{ font-size: 28px; font-weight: 700; letter-spacing: -0.02em; color: {TEXTE}; margin: 8px 0 2px; line-height: 1.1; }}
    .legende-chiffre {{ font-size: 13px; color: {TEXTE_2}; }}
    .petit-titre {{ font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: {TEXTE_3}; }}
    .classe-ligne {{ display: flex; align-items: center; gap: 12px; }}
    .pastille {{ width: 14px; height: 14px; border-radius: 4px; flex: none; display: inline-block; vertical-align: -2px; }}
    .badge {{ display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600;
              border-radius: 999px; padding: 4px 10px; white-space: nowrap; }}
    .badge-rf {{ color: {VERT}; background: {VERT_CLAIR}; }}
    .badge-ann {{ color: #8A5312; background: #FBF1E3; }}
    .badge-neutre {{ color: {TEXTE_2}; background: #F1F2F0; }}
    .badge-alerte {{ color: #92400E; background: #FEF3C7; }}
    .barre-fond {{ flex: 1; height: 8px; background: #F1F2F0; border-radius: 4px; }}
    .barre {{ height: 8px; border-radius: 4px; }}
    .note {{ display: flex; gap: 14px; align-items: flex-start; background: #FFFFFF; border: 1px solid {BORDURE};
             border-radius: 12px; padding: 16px 18px; font-size: 14px; line-height: 1.6; color: #374151; margin-top: 10px; }}
    .note svg {{ flex: none; margin-top: 2px; }}
    .statut {{ display: flex; gap: 12px; align-items: center; border-radius: 10px; padding: 13px 16px;
               font-size: 14px; margin: 4px 0 12px; }}
    .statut svg {{ flex: none; }}
    .statut-ok {{ background: {VERT_CLAIR}; color: #1F3D2B; }}
    .statut-alerte {{ background: #FEF3C7; color: #78350F; }}
    .legende {{ display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 13px; color: #374151; }}

    /* Conteneurs à bordure (st.container(border=True)) : cartes blanches */
    [data-testid="stVerticalBlockBorderWrapper"] {{ background: #FFFFFF; border-color: {BORDURE} !important;
        border-radius: 12px !important; }}
    .st-key-hero {{ padding: 14px 10px; }}

    /* Liens de page utilisés comme boutons d'action */
    .st-key-action_principale a {{ background: {VERT}; border-radius: 8px; padding: 8px 16px; }}
    .st-key-action_principale a span, .st-key-action_principale a p {{ color: #FFFFFF !important; font-weight: 600; }}
    .st-key-action_principale a:hover {{ background: #163B27; }}
    .st-key-action_secondaire a {{ border: 1px solid #D1D5DB; border-radius: 8px; padding: 8px 16px; background: #FFFFFF; }}
    </style>
    """, unsafe_allow_html=True)

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


# --- Petits composants HTML ---------------------------------------------------
def html(contenu: str):
    st.markdown(contenu, unsafe_allow_html=True)


def entete(titre: str, sous_titre: str = ""):
    html(f'<div class="entete"><h1>{titre}</h1><p>{sous_titre}</p></div>')


def titre_section(numero, titre: str, sous_titre: str = ""):
    prefixe = f"{numero}. " if numero else ""
    html(f'<div class="titre-section">{prefixe}{titre}</div>'
         + (f'<div class="sous-titre-section">{sous_titre}</div>' if sous_titre else ""))


def carte_chiffre(icone: str, valeur: str, legende: str):
    html(f'<div class="carte">{svg(icone, 22, VERT)}<div class="chiffre">{valeur}</div>'
         f'<div class="legende-chiffre">{legende}</div></div>')


def carte_stat(etiquette: str, valeur: str, aide: str):
    html(f'<div class="carte"><div class="legende-chiffre" style="font-size:12px;color:{TEXTE_3}">{etiquette}</div>'
         f'<div class="chiffre" style="font-size:26px">{valeur}</div>'
         f'<div class="legende-chiffre" style="font-size:12px">{aide}</div></div>')


def note(texte: str, icone: str = "info", couleur: str = VERT):
    html(f'<div class="note">{svg(icone, 20, couleur)}<div>{texte}</div></div>')


def statut(texte: str, ok: bool = True):
    icone, classe, couleur = ("valide", "statut-ok", VERT) if ok else ("alerte", "statut-alerte", "#B45309")
    html(f'<div class="statut {classe}">{svg(icone, 18, couleur, 2)}<span>{texte}</span></div>')


def pastille(couleur: str, taille: int = 12) -> str:
    return f'<span class="pastille" style="background:{couleur};width:{taille}px;height:{taille}px"></span>'


def legende_modeles(taille: int = 12) -> str:
    return (f'<div class="legende"><span>{pastille(VERT, taille)} Random Forest (retenu)</span>'
            f'<span>{pastille(OCRE, taille)} ANN</span></div>')


def legende_classes():
    html('<div class="legende" style="margin-top:8px">'
         + "".join(f"<span>{pastille(h)} {n}</span>" for n, h in COULEURS_HEX.items()) + "</div>")


def barres_doubles(lignes, largeur_libelle: int = 190) -> str:
    """Barres groupées RF / ANN en HTML. lignes = [(libellé, valeur RF, valeur ANN), …] (valeurs 0–1)."""
    corps = ""
    for lib, rf, ann in lignes:
        corps += (f'<span style="font-size:13px">{lib}</span>'
                  f'<div style="display:flex;flex-direction:column;gap:3px">'
                  f'<div title="Random Forest : {rf:.0%}" style="width:{max(rf, .01) * 100:.0f}%;height:7px;'
                  f'background:{VERT};border-radius:3px"></div>'
                  f'<div title="ANN : {ann:.0%}" style="width:{max(ann, .01) * 100:.0f}%;height:7px;'
                  f'background:{OCRE};border-radius:3px"></div></div>')
    return (f'<div style="display:grid;grid-template-columns:{largeur_libelle}px minmax(0,1fr);gap:10px 18px;'
            f'align-items:center">{corps}</div>')


def echelle_couleurs():
    """Échelle Altair : couleur fixe pour chaque classe."""
    return alt.Scale(domain=list(COULEURS_HEX), range=list(COULEURS_HEX.values()))


def style_graphique(graphique):
    """Style commun des graphiques Altair : police Inter, grille discrète, sans cadre."""
    return (graphique.configure(font="Inter")
            .configure_view(strokeWidth=0)
            .configure_title(fontSize=14, fontWeight=600, color=TEXTE, anchor="start")
            .configure_axis(labelColor=TEXTE_2, titleColor=TEXTE_2, gridColor="#EEF0EC",
                            domainColor=BORDURE, tickColor=BORDURE, labelFontSize=12, titleFontSize=12))


def barres_classes(df: pd.DataFrame, x: str, titre_x: str, domaine=None, titre=None, hauteur: int = 290):
    """Barres horizontales colorées par classe (gabarit commun)."""
    graph = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=16).encode(
        x=alt.X(f"{x}:Q", title=titre_x, scale=alt.Scale(domain=domaine) if domaine else alt.Scale()),
        y=alt.Y("Classe:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
        color=alt.Color("Classe:N", scale=echelle_couleurs(), legend=None),
        tooltip=list(df.columns)).properties(height=hauteur)
    return graph.properties(title=titre) if titre else graph


def hex_vers_rgb(code_hex: str) -> list:
    code_hex = code_hex.lstrip("#")
    return [int(code_hex[i:i + 2], 16) for i in (0, 2, 4)]


def fmt(x: float, d: int = 3) -> str:
    """Nombre au format français (virgule décimale)."""
    return f"{x:.{d}f}".replace(".", ",")


# =============================================================================
# 5. MODÈLES : chargement (cache), indices, prédiction
# =============================================================================
@st.cache_resource(show_spinner="Chargement du modèle…")
def charger_modele():
    """Charge le dictionnaire {'pipeline', 'variables', 'classes'} sauvegardé avec joblib."""
    contenu = joblib.load(CHEMIN_MODELE)
    return contenu["pipeline"], list(contenu["variables"]), list(contenu["classes"])


@st.cache_data
def charger_fiche():
    return json.loads(CHEMIN_FICHE.read_text(encoding="utf-8")) if CHEMIN_FICHE.exists() else None


@st.cache_resource(show_spinner="Chargement de l'ANN…")
def charger_ann():
    """Charge les poids de l'ANN (format numpy .npz) : pas besoin de TensorFlow."""
    if not CHEMIN_ANN.exists():
        return None
    p = np.load(CHEMIN_ANN)
    n_couches = sum(1 for k in p.files if k.startswith("W"))
    return {"moyenne": p["moyenne"], "ecart": p["ecart"],
            "couches": [(p[f"W{i}"], p[f"b{i}"]) for i in range(n_couches)]}


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
        # (la version standard est « − 7,5 × bleu »). On la reproduit pour rester cohérent avec les modèles.
        "EVI": 2.5 * (pir - rouge) / (pir + 6 * rouge + 7.5 * bleu + 1),
        "NDRE": nd(pir, red_edge),
        "SAVI": 1.5 * (pir - rouge) / (pir + rouge + 0.5),
        "NDMI": nd(pir, swir1),
        "NDWI": nd(vert, pir),
        "MSI": swir1 / pir if pir != 0 else np.nan,
        "BSI": nd(swir1 + rouge, pir + bleu),
        "BI": np.sqrt((bleu**2 + vert**2 + rouge**2 + pir**2) / 4),
    }


def gelu(z: np.ndarray) -> np.ndarray:
    """GELU exacte (activation de Keras « gelu ») : 0,5·z·(1 + erf(z/√2))."""
    from scipy.special import erf  # scipy est installé avec scikit-learn
    return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))


def probas_ann(ann: dict, X: pd.DataFrame) -> np.ndarray:
    """Propagation avant de l'ANN : standardisation → 3 couches Dense + GELU → Dense + softmax.
    Résultat identique à Keras (écart < 1e-6, vérifié)."""
    h = (X.to_numpy(dtype=float) - ann["moyenne"]) / ann["ecart"]
    for i, (W, b) in enumerate(ann["couches"]):
        h = h @ W + b
        if i < len(ann["couches"]) - 1:
            h = gelu(h)
    e = np.exp(h - h.max(axis=1, keepdims=True))  # softmax numériquement stable
    return e / e.sum(axis=1, keepdims=True)


def probas_rf(pipeline, X: pd.DataFrame) -> np.ndarray:
    # Le RF a été entraîné sur un tableau numpy : valeurs passées dans l'ordre exact des variables.
    # Le RandomOverSampler n'intervient pas à la prédiction.
    return pipeline.predict_proba(X.to_numpy(dtype=float))


def predire(fonction_probas, classes, X: pd.DataFrame) -> pd.DataFrame:
    """Classe prédite (FR), confiance (= probabilité max), alerte et probabilités par classe."""
    probas = fonction_probas(X)
    noms_fr = [LIBELLES_FR[c] for c in classes]
    idx = probas.argmax(axis=1)
    res = pd.DataFrame(index=X.index)
    res["classe_predite"] = [noms_fr[i] for i in idx]
    res["confiance"] = probas.max(axis=1).round(3)
    res["prediction_peu_sure"] = res["confiance"] < SEUIL_INCERTITUDE
    df_probas = pd.DataFrame(probas, columns=[f"P({n})" for n in noms_fr], index=X.index).round(3)
    return pd.concat([res, df_probas], axis=1)


def contexte():
    """Modèles chargés (en cache) : fonctions de probabilités, variables et classes."""
    pipeline, variables, classes = charger_modele()
    modeles = {"Random Forest": lambda X: probas_rf(pipeline, X)}
    ann = charger_ann()
    if ann is not None:
        modeles["ANN"] = lambda X: probas_ann(ann, X)
    return modeles, variables, classes


# =============================================================================
# 6. CARTE
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
# 7. PAGES (chacune se lit de haut en bas, sans onglets)
# =============================================================================
PAGES = {}  # rempli dans main() : sert aux liens entre pages


def page_accueil():
    html(f'<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;'
         f'margin-bottom:14px"><span style="font-size:13px;color:{TEXTE_3}">Projet de Deep Learning · '
         f'M2 Data Science · UFHB</span><span class="badge badge-alerte">{svg("alerte", 14, "currentColor", 2)} '
         f'Démonstration pédagogique</span></div>')

    with st.container(border=True, key="hero"):
        texte, image = st.columns(2, gap="large", vertical_alignment="center")
        with texte:
            html('<h1 style="font-size:32px;line-height:1.2;font-weight:700;margin:0 0 14px;padding:0">'
                 "Classer l'occupation du sol autour de Yamoussoukro</h1>"
                 f'<p style="font-size:16px;line-height:1.6;color:{TEXTE_2};margin:0 0 18px">'
                 "À partir de 10 bandes Sentinel-2 et de 9 indices spectraux, deux modèles prédisent la classe "
                 "de chaque point : forêt, savane, cultures, bâti ou eau.</p>")
            b1, b2 = st.columns([1, 1.2])
            with b1:
                with st.container(key="action_principale"):
                    st.page_link(PAGES["point"], label="Prédire un point", icon=":material/my_location:")
            with b2:
                with st.container(key="action_secondaire"):
                    st.page_link(PAGES["performances"], label="Voir les performances", icon=":material/bar_chart:")
        with image:
            if CHEMIN_BANNIERE.exists():
                st.image(str(CHEMIN_BANNIERE), width="stretch")

    st.write("")
    c = st.columns(4)
    with c[0]:
        carte_chiffre("base", "1 000", "points au sol étiquetés")
    with c[1]:
        carte_chiffre("couches", "19", "variables : 10 bandes + 9 indices")
    with c[2]:
        carte_chiffre("grille", "8", "classes d'occupation du sol")
    with c[3]:
        carte_chiffre("balance", "2", "modèles comparés : RF et ANN")

    titre_section(None, "Les 8 classes")
    cols = st.columns(4)
    for i, (nom, couleur) in enumerate(COULEURS_HEX.items()):
        cols[i % 4].markdown(
            f'<div class="carte" style="padding:14px 16px;margin-bottom:12px"><div class="classe-ligne">'
            f'{pastille(couleur, 14)}<div><div style="font-size:14px;font-weight:500">{nom}</div>'
            f'<div style="font-size:12px;color:{TEXTE_3}">{PART_CLASSES[nom]} % des données</div></div></div></div>',
            unsafe_allow_html=True)

    note("<b>Démonstration, pas un outil opérationnel.</b> Les modèles classent correctement un peu plus "
         "d'un point sur deux et restent faibles sur les classes rares. Ils ne sont valables que pour la zone "
         "de Yamoussoukro et pour des données préparées comme celles du projet.")


def page_performances():
    entete("Performances du Random Forest",
           "Mesurées sur des données jamais vues pendant l'entraînement : 150 points de test, moyenne sur 5 graines.")

    titre_section(1, "Scores globaux")
    c = st.columns(4)
    with c[0]:
        carte_stat("Accuracy", fmt(SCORES_TEST["Accuracy"]), "≈ 56 points sur 100 bien classés")
    with c[1]:
        carte_stat("Macro-F1", fmt(SCORES_TEST["Macro-F1"]), "Moyenne des 8 classes")
    with c[2]:
        carte_stat("MCC", fmt(SCORES_TEST["MCC"]), "0 = hasard, 1 = parfait")
    with c[3]:
        carte_stat("Macro-F1 spatial", fmt(SCORES_TEST["Macro-F1 VC spatiale"]), "Validation par blocs géographiques")

    titre_section(2, "Classes bien et mal reconnues",
                  "F1 sur le test (0 = jamais reconnue, 1 = toujours reconnue) et nombre de points par classe.")
    g, d = st.columns(2, gap="large")
    with g:
        with st.container(border=True):
            st.altair_chart(style_graphique(barres_classes(
                PERF_PAR_CLASSE[["Classe", "F1", "Précision", "Rappel", "n test"]], "F1", "F1 (test)",
                domaine=[0, 1], titre="F1 par classe")), width="stretch")
    with d:
        with st.container(border=True):
            st.altair_chart(style_graphique(barres_classes(
                PERF_PAR_CLASSE[["Classe", "Effectif total"]], "Effectif total", "Nombre de points",
                titre="Nombre de points (1 000 au total)")), width="stretch")
    note("<b>Hévéa</b> et <b>Plans d'eau</b> sont bien reconnues. <b>Zones humides</b> et <b>Sol nu</b> ne le "
         "sont jamais : ce sont les classes les plus rares (27 et 8 points), malgré le suréchantillonnage.")

    titre_section(3, "Où se trompe le modèle ?",
                  "Pour chaque vraie classe (ligne) : part des points attribués à chaque classe prédite (colonne).")
    with st.container(border=True):
        mc = pd.DataFrame(MATRICE_CONFUSION, index=ORDRE_MC, columns=ORDRE_MC).reset_index(names="Vraie classe")
        mc = mc.melt(id_vars="Vraie classe", var_name="Classe prédite", value_name="Part")
        base = alt.Chart(mc).encode(
            x=alt.X("Classe prédite:N", sort=ORDRE_MC, axis=alt.Axis(labelAngle=-30, labelLimit=160)),
            y=alt.Y("Vraie classe:N", sort=ORDRE_MC, axis=alt.Axis(labelLimit=200)))
        cases = base.mark_rect(cornerRadius=2).encode(
            color=alt.Color("Part:Q", scale=alt.Scale(range=["#FFFFFF", VERT], domain=[0, 1], interpolate="rgb"), legend=None),
            tooltip=["Vraie classe", "Classe prédite", alt.Tooltip("Part:Q", format=".0%")])
        txt = base.transform_filter("datum.Part > 0").mark_text(fontSize=12).encode(
            text=alt.Text("Part:Q", format=".0%"),
            color=alt.condition("datum.Part > 0.5", alt.value("white"), alt.value(TEXTE)))
        st.altair_chart(style_graphique((cases + txt).properties(height=420)), width="stretch")
    note("La colonne <b>Hévéa et autres cultures</b> concentre la plupart des erreurs : c'est la classe "
         "majoritaire (60 % des données), le modèle s'y replie quand il hésite.")

    titre_section(4, "Variables les plus utiles", "Importance moyenne (valeurs SHAP) des 19 variables.")
    with st.container(border=True):
        imp = pd.DataFrame({"Variable": list(IMPORTANCE_SHAP), "Importance": list(IMPORTANCE_SHAP.values())})
        imp["Type"] = np.where(imp["Variable"].str.match(r"^B\d"), "Bande", "Indice")
        graph = alt.Chart(imp).mark_bar(cornerRadiusEnd=3, height=12).encode(
            x=alt.X("Importance:Q", title="Importance SHAP moyenne"),
            y=alt.Y("Variable:N", sort="-x", title=None),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["Indice", "Bande"], range=[VERT, "#A7B8AD"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Variable", "Type", "Importance"]).properties(height=420)
        st.altair_chart(style_graphique(graph), width="stretch")
    note("Les indices de végétation (<b>NDVI</b>, <b>NDRE</b>, <b>EVI</b>) sont les variables les plus utiles.")

    titre_section(5, "Formules des indices")
    with st.expander("Afficher les 9 formules", icon=":material/functions:"):
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
        st.caption("Colonnes : B2 bleu, B3 vert, B4 rouge, B5 red-edge, B8 proche infrarouge, B9 SWIR 1, "
                   "B10 SWIR 2. * Formule telle que présente dans les données du projet (la version standard "
                   "utilise − 7,5·B2).")
        fiche = charger_fiche()
        if fiche:
            st.json(fiche, expanded=False)


def page_comparaison():
    g, d = st.columns([3, 2], vertical_alignment="bottom")
    with g:
        entete("ANN contre Random Forest", "Mêmes données, mêmes partitions, 5 graines pour chaque modèle.")
    with d:
        html(f'<div style="display:flex;justify-content:flex-end;margin-bottom:16px">{legende_modeles()}</div>')

    titre_section(1, "Scores à chaque étape d'évaluation")
    for col, (etape, info) in zip(st.columns(3), ETAPES.items()):
        badge_txt, badge_cls = info["badge"]
        barres = ""
        for metrique, rf, ann in info["scores"]:
            gras_rf, gras_ann = ("font-weight:600", "") if rf >= ann else ("", "font-weight:600")
            barres += (f'<div style="margin-top:14px"><div style="font-size:13px;color:#374151;margin-bottom:6px">'
                       f'{metrique}</div>'
                       f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px"><div class="barre-fond">'
                       f'<div class="barre" style="width:{rf * 100:.1f}%;background:{VERT}"></div></div>'
                       f'<span style="font-size:13px;width:42px;text-align:right;{gras_rf}">{fmt(rf)}</span></div>'
                       f'<div style="display:flex;align-items:center;gap:8px"><div class="barre-fond">'
                       f'<div class="barre" style="width:{ann * 100:.1f}%;background:{OCRE}"></div></div>'
                       f'<span style="font-size:13px;width:42px;text-align:right;{gras_ann}">{fmt(ann)}</span></div></div>')
        if etape == "Validation spatiale":
            barres += (f'<p style="font-size:13px;line-height:1.55;color:{TEXTE_2};margin:14px 0 0">Les intervalles '
                       "des deux modèles se recouvrent : la différence n'est pas nette.</p>")
        col.markdown(
            f'<div class="carte"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
            f'<div><div style="font-weight:600;font-size:15px">{etape}</div>'
            f'<div style="font-size:12px;color:{TEXTE_3}">{info["sous_titre"]}</div></div>'
            f'<span class="badge badge-{badge_cls}">{badge_txt}</span></div>{barres}</div>',
            unsafe_allow_html=True)

    titre_section(2, "F1 par classe sur le test", "0 = classe jamais reconnue, 1 = toujours reconnue.")
    html(f'<div class="carte">{barres_doubles([(c, rf, ann) for c, (rf, ann) in F1_TEST_CLASSES.items()])}</div>')
    note("L'ANN fait mieux sur <b>Forêt</b>, <b>Plans d'eau</b> et <b>Zones humides</b> ; le Random Forest un peu "
         "mieux sur <b>Café et cacao</b>. Aucun des deux ne reconnaît <b>Sol nu</b> (2 points de test).")

    titre_section(3, "Surapprentissage et coût",
                  "Écart train/val = Macro-F1 d'entraînement moins Macro-F1 de validation.")
    c = st.columns(4)
    with c[0]:
        carte_stat("Écart train/val · RF", fmt(ECART_TRAIN_VAL["Random Forest"], 2), "Mémorise l'entraînement")
    with c[1]:
        carte_stat("Écart train/val · ANN", fmt(ECART_TRAIN_VAL["ANN"], 2), "Généralise bien")
    with c[2]:
        carte_stat("Entraînement · RF", f"{TEMPS_ENTRAINEMENT['Random Forest']:.0f} s", "Par graine")
    with c[3]:
        carte_stat("Entraînement · ANN", f"{TEMPS_ENTRAINEMENT['ANN']:.0f} s", "Par graine")

    titre_section(4, "Conclusion")
    html(f'<div class="carte"><ol style="margin:0;padding-left:20px;font-size:14px;line-height:1.7;color:#374151">'
         f'<li><b style="color:{TEXTE}">Choix selon le protocole :</b> le Random Forest est retenu, meilleur en '
         "validation (0,450 contre 0,389) et en validation spatiale (0,333 contre 0,288), choisi avant de "
         "regarder le test.</li>"
         f'<li><b style="color:{TEXTE}">Ce que montre le test :</b> l\'ANN généralise mieux (Macro-F1 0,412 contre '
         "0,309 ; accuracy 0,629 contre 0,563).</li>"
         f'<li><b style="color:{TEXTE}">Interprétation :</b> cohérent avec le fort surapprentissage du Random '
         "Forest (écart 0,55 contre 0,05) et le chevauchement des deux modèles en validation spatiale.</li>"
         f'<li><b style="color:{TEXTE}">Limite commune :</b> les classes rares (Sol nu, Zones humides, Savane) '
         "restent mal reconnues, faute de données.</li></ol></div>")
    st.caption("Scores de test de l'ANN : ANN reconstruit à l'identique (scripts/sauvegarder_ann.py).")


def page_fichier():
    modeles, variables, classes = contexte()
    entete("Prédire un fichier CSV", "Chaque ligne est un point au sol ; les deux modèles prédisent sa classe.")

    titre_section(1, "Données")
    source = st.segmented_control("Source des données", ["Fichier d'exemple (48 points)", "Mon propre CSV"],
                                  default="Fichier d'exemple (48 points)", label_visibility="collapsed")
    df = None
    if source == "Mon propre CSV":
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        with c1:
            fichier = st.file_uploader("Importer un CSV", type="csv", label_visibility="collapsed")
        with c2:
            modele_csv = pd.DataFrame(columns=variables + ["lon", "lat"]).to_csv(index=False)
            st.download_button("Modèle de CSV vide", modele_csv, "modele_colonnes.csv", "text/csv",
                               icon=":material/table_view:", width="stretch")
        st.caption(f"Colonnes obligatoires ({len(variables)}) : {', '.join(variables)}. "
                   "Facultatif : lon/lat ou x/y (EPSG:32630) pour la carte. Les autres colonnes sont ignorées.")
        if fichier is not None:
            try:
                df = pd.read_csv(fichier, sep=None, engine="python")  # virgule ou point-virgule
            except Exception as e:
                st.error(f"Impossible de lire le fichier : {e}", icon=":material/error:")
    elif CHEMIN_EXEMPLE.exists():
        df = pd.read_csv(CHEMIN_EXEMPLE)
    else:
        st.error("Fichier d'exemple introuvable (exemple_points.csv).", icon=":material/error:")

    if df is None:
        return

    # Vérification des colonnes
    manquantes = [v for v in variables if v not in df.columns]
    if manquantes:
        st.error(f"**{len(manquantes)} colonne(s) manquante(s)** : {', '.join(manquantes)}", icon=":material/error:")
        st.info("Téléchargez le « Modèle de CSV vide » pour voir les noms de colonnes attendus.",
                icon=":material/info:")
        return

    X = df[variables].apply(pd.to_numeric, errors="coerce")
    invalides = X.isna().any(axis=1)
    if invalides.any():
        st.warning(f"{int(invalides.sum())} ligne(s) avec valeurs manquantes ou non numériques ignorée(s).",
                   icon=":material/warning:")
    X = X[~invalides]
    if X.empty:
        st.error("Aucune ligne exploitable.", icon=":material/error:")
        return

    tous = {nom: predire(fn, classes, X) for nom, fn in modeles.items()}

    titre_section(2, "Résumé")
    nom_modele = st.segmented_control("Modèle affiché", list(tous), default="Random Forest",
                                      format_func=lambda m: f"{m} (retenu)" if m == "Random Forest" else m)
    nom_modele = nom_modele or "Random Forest"  # si l'utilisateur désélectionne
    resultats = tous[nom_modele]
    n, n_doute = len(resultats), int(resultats["prediction_peu_sure"].sum())
    c = st.columns(4)
    with c[0]:
        carte_stat("Points analysés", str(n), "Lignes valides du fichier")
    with c[1]:
        carte_stat("Confiance moyenne", f"{resultats['confiance'].mean():.0%}", "Probabilité de la classe retenue")
    with c[2]:
        carte_stat("Prédictions peu sûres", str(n_doute), f"Confiance inférieure à {SEUIL_INCERTITUDE:.0%}")
    with c[3]:
        if len(tous) == 2:
            accord = (tous["Random Forest"]["classe_predite"] == tous["ANN"]["classe_predite"]).mean()
            carte_stat("Accord RF / ANN", f"{accord:.0%}", "Même classe prédite par les deux")
        else:
            carte_stat("Classe la plus fréquente", resultats["classe_predite"].mode()[0], "")

    numero = 3
    coords = extraire_coordonnees(df.loc[resultats.index])
    if coords is not None:
        titre_section(numero, "Carte des prédictions", "Survolez un point pour voir sa classe et la confiance.")
        afficher_carte(coords, resultats)
        numero += 1

    titre_section(numero, "Répartition")
    g1, g2 = st.columns(2, gap="large")
    with g1:
        with st.container(border=True):
            rep = resultats["classe_predite"].value_counts().rename_axis("Classe").reset_index(name="Points")
            st.altair_chart(style_graphique(barres_classes(rep, "Points", "Nombre de points", hauteur=280,
                                                           titre="Classes prédites")), width="stretch")
    with g2:
        with st.container(border=True):
            graph = alt.Chart(resultats).mark_bar(color=COULEURS_MODELES[nom_modele],
                                                  cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("confiance:Q", bin=alt.Bin(step=0.1), title="Confiance", axis=alt.Axis(format="%")),
                y=alt.Y("count():Q", title="Nombre de points"),
            ).properties(height=280, title="Niveau de confiance")
            st.altair_chart(style_graphique(graph), width="stretch")
    numero += 1

    titre_section(numero, "Détail des prédictions")
    filtre = st.toggle("Afficher seulement les prédictions peu sûres", value=False)
    vue = resultats[resultats["prediction_peu_sure"]] if filtre else resultats
    cols_probas = [c for c in vue.columns if c.startswith("P(")]
    st.dataframe(
        vue, width="stretch", height=380,
        column_config={
            "classe_predite": st.column_config.TextColumn("Classe prédite"),
            "confiance": st.column_config.ProgressColumn("Confiance", min_value=0, max_value=1, format="percent"),
            "prediction_peu_sure": st.column_config.CheckboxColumn("Peu sûre ?"),
            **{c: st.column_config.NumberColumn(c[2:-1], format="percent") for c in cols_probas},
        },
    )

    export = df.loc[resultats.index]
    for nom, res in tous.items():  # résultats des deux modèles, colonnes préfixées rf_ / ann_
        export = export.join(res.add_prefix("rf_" if nom == "Random Forest" else "ann_"))
    st.download_button("Télécharger les résultats (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                       "predictions_occupation_sols.csv", "text/csv", type="primary", icon=":material/download:")


def page_point():
    modeles, variables, classes = contexte()
    entete("Prédire un point", "Saisissez les 10 réflectances (0 à 1). Les indices sont calculés "
                               "automatiquement et les deux modèles prédisent la classe.")

    gauche, droite = st.columns(2, gap="large")

    with gauche:  # --- Saisie
        with st.container(border=True):
            choix = st.selectbox("Partir d'un exemple réel", list(EXEMPLES_BANDES),
                                 help="Points du jeu de données, déjà vus par les modèles pendant l'entraînement.")
            if st.session_state.get("exemple_charge") != choix:  # nouvel exemple : on remplit les champs
                for i, v in enumerate(EXEMPLES_BANDES[choix]):
                    st.session_state[f"B{i + 1}"] = v
                st.session_state["exemple_charge"] = choix

            bandes = {}
            for groupe in ["Visible", "Red-edge et proche infrarouge", "Infrarouge moyen (SWIR)"]:
                html(f'<div class="petit-titre" style="margin:10px 0 2px">{groupe}</div>')
                noms = [b for b, (_, g) in BANDES.items() if g == groupe]
                for col, b in zip(st.columns(4), noms):
                    bandes[b] = col.number_input(BANDES[b][0], min_value=0.0, max_value=1.0,
                                                 step=0.001, format="%.4f", key=b)
            indices = calculer_indices(bandes)
            with st.expander("Voir les 9 indices calculés"):
                st.dataframe(pd.DataFrame([indices]).round(4), hide_index=True, width="stretch")

    with droite:  # --- Résultats des deux modèles
        if any(pd.isna(v) for v in indices.values()):
            st.error("Un indice n'est pas calculable (division par zéro) : vérifiez les valeurs saisies.",
                     icon=":material/error:")
            return

        X = pd.DataFrame([{**bandes, **indices}])[variables]  # ordre exact attendu par les modèles
        res = {nom: predire(fn, classes, X).iloc[0] for nom, fn in modeles.items()}

        for col, (nom, r) in zip(st.columns(len(res)), res.items()):
            classe, conf = r["classe_predite"], float(r["confiance"])
            badge = ('<span class="badge badge-rf" style="font-size:11px;padding:3px 8px">Retenu</span>'
                     if nom == "Random Forest" else "")
            col.markdown(
                f'<div class="carte"><div style="display:flex;justify-content:space-between;align-items:center;'
                f'min-height:22px"><span class="petit-titre">{nom}</span>{badge}</div>'
                f'<div class="classe-ligne" style="margin:12px 0">{pastille(COULEURS_HEX[classe], 14)}'
                f'<span style="font-size:18px;font-weight:600;line-height:1.3">{classe}</span></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:13px;color:{TEXTE_2};'
                f'margin-bottom:6px"><span>Confiance</span><span style="font-weight:600;color:{TEXTE}">{conf:.0%}'
                f'</span></div><div class="barre-fond" style="height:6px"><div class="barre" style="height:6px;'
                f'width:{conf * 100:.0f}%;background:{COULEURS_MODELES[nom]}"></div></div></div>',
                unsafe_allow_html=True)

        st.write("")
        classes_pred = [r["classe_predite"] for r in res.values()]
        if len(res) == 2 and len(set(classes_pred)) == 1:
            statut(f"Les deux modèles sont d'accord : <b>{classes_pred[0]}</b>", ok=True)
        elif len(res) == 2:
            statut("<b>Les deux modèles ne sont pas d'accord</b> : ce point est difficile à classer.", ok=False)
        peu_surs = [nom for nom, r in res.items() if r["prediction_peu_sure"]]
        if peu_surs:
            statut(f"Prédiction peu sûre (confiance inférieure à {SEUIL_INCERTITUDE:.0%}) : {', '.join(peu_surs)}.",
                   ok=False)

        # Probabilités : les 4 classes les plus probables, puis le reste regroupé
        p_rf = res["Random Forest"][[c for c in res["Random Forest"].index if c.startswith("P(")]].astype(float)
        p_ann = res["ANN"][p_rf.index].astype(float) if "ANN" in res else p_rf * 0
        ordre = pd.concat([p_rf, p_ann], axis=1).max(axis=1).sort_values(ascending=False).index
        lignes = [(c[2:-1], p_rf[c], p_ann[c]) for c in ordre[:4]]
        reste = ordre[4:]
        html(f'<div class="carte"><div style="display:flex;justify-content:space-between;align-items:center;'
             f'gap:12px;flex-wrap:wrap;margin-bottom:14px"><span style="font-size:15px;font-weight:600">'
             f'Probabilités par classe</span><div class="legende" style="font-size:12px">'
             f'<span>{pastille(VERT, 10)} RF</span><span>{pastille(OCRE, 10)} ANN</span></div></div>'
             f'{barres_doubles(lignes, 170)}'
             f'<div style="display:grid;grid-template-columns:170px minmax(0,1fr);gap:18px;margin-top:12px;'
             f'font-size:12px;color:{TEXTE_3}"><span>{len(reste)} autres classes</span>'
             f'<span>au plus {max(p_rf[reste].max(), p_ann[reste].max()):.0%} chacune</span></div></div>')


# =============================================================================
# 8. POINT D'ENTRÉE : navigation par sections (Explorer / Prédire)
# =============================================================================
def main():
    st.set_page_config(page_title="Occupation des sols · Yamoussoukro", page_icon=":material/layers:",
                       layout="wide")
    injecter_style()

    if not CHEMIN_MODELE.exists():
        st.error(f"Modèle introuvable : `{CHEMIN_MODELE.relative_to(RACINE)}`")
        st.stop()

    if CHEMIN_LOGO.exists():
        st.logo(str(CHEMIN_LOGO), size="large")

    PAGES.update({
        "accueil": st.Page(page_accueil, title="Accueil", icon=":material/home:", url_path="accueil", default=True),
        "performances": st.Page(page_performances, title="Performances", icon=":material/bar_chart:",
                                url_path="performances"),
        "comparaison": st.Page(page_comparaison, title="ANN contre Random Forest", icon=":material/balance:",
                               url_path="comparaison"),
        "fichier": st.Page(page_fichier, title="Fichier CSV", icon=":material/description:", url_path="fichier"),
        "point": st.Page(page_point, title="Point unique", icon=":material/my_location:", url_path="point"),
    })
    navigation = st.navigation({
        "Explorer": [PAGES["accueil"], PAGES["performances"], PAGES["comparaison"]],
        "Prédire": [PAGES["fichier"], PAGES["point"]],
    })

    with st.sidebar:  # carte « modèle retenu », sous le menu
        html(f'<div class="carte" style="padding:14px;margin-top:12px">'
             f'<div class="petit-titre" style="margin-bottom:8px">Modèle retenu</div>'
             f'<div style="font-weight:600;font-size:14px;margin-bottom:8px">Random Forest · 530 arbres</div>'
             f'<div style="display:flex;justify-content:space-between;font-size:13px;color:{TEXTE_2};margin-bottom:4px">'
             f'<span>Accuracy test</span><b style="color:{TEXTE}">{fmt(SCORES_TEST["Accuracy"])}</b></div>'
             f'<div style="display:flex;justify-content:space-between;font-size:13px;color:{TEXTE_2}">'
             f'<span>Macro-F1 test</span><b style="color:{TEXTE}">{fmt(SCORES_TEST["Macro-F1"])}</b></div></div>'
             f'<p style="font-size:12px;color:{TEXTE_3};margin-top:14px;line-height:1.5">Démonstration pédagogique, '
             "M2 Data Science, UFHB. Valable uniquement pour la zone de Yamoussoukro.</p>")

    navigation.run()


if __name__ == "__main__":
    main()
