# 🛰️ Occupation des sols — Yamoussoukro (Sentinel-2)

Application de démonstration du projet de Deep Learning (Master 2 Data Science, UFHB) :
classification de l'occupation des sols en 8 classes à partir de données Sentinel-2.
Modèle final : Random Forest (530 arbres, 19 variables).

## ▶️ Version en ligne
**Lien :** _à compléter après le déploiement sur Streamlit Community Cloud_

## 💻 Lancer l'application en local
Prérequis : Python 3.11 ou plus récent.

```bash
git clone <URL_DU_DEPOT>
cd school-project
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux : source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
L'application s'ouvre sur http://localhost:8501.

## 🧭 Contenu
| Page | Rôle |
|---|---|
| Accueil | Contexte, classes, limites |
| Comprendre le modèle | Scores, F1 par classe, matrice de confusion, importance des variables, formules des indices |
| Prédire un fichier | CSV d'exemple fourni ou CSV personnel → carte, graphiques, export |
| Prédire un point | Saisie de 10 bandes → classe prédite et probabilités |

## ⚠️ Limites
Démonstration pédagogique : Macro-F1 test = 0,309 ; accuracy = 0,563.
Valable uniquement pour la zone de Yamoussoukro et des données préparées comme celles du projet.
