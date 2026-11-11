# 🛰️ Occupation des sols — Yamoussoukro (Sentinel-2)

Application de démonstration du projet de Deep Learning (Master 2 Data Science, UFHB) :
classification de l'occupation des sols en 8 classes à partir de données Sentinel-2.
Modèle retenu : Random Forest (530 arbres, 19 variables), comparé à un réseau de neurones (ANN 128-64-32).

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
| Section | Page | Rôle |
|---|---|---|
| Explorer | Accueil | Contexte, chiffres clés, les 8 classes, limites |
| Explorer | Performances | Scores du Random Forest, F1 par classe, matrice de confusion, variables utiles (SHAP), formules des indices |
| Explorer | ANN contre Random Forest | Scores en validation, validation spatiale et test ; F1 par classe ; surapprentissage ; conclusion |
| Prédire | Fichier CSV | CSV d'exemple ou personnel → résumé, carte, répartition, détail, export (deux modèles) |
| Prédire | Point unique | Saisie de 10 bandes → classe prédite par les deux modèles et probabilités comparées |

## 🧠 Note sur l'ANN
L'ANN n'avait pas été sauvegardé dans les notebooks. `scripts/sauvegarder_ann.py` le ré-entraîne avec la
recette exacte de `experience A.ipynb` (résultats de validation identiques) et enregistre ses poids dans
`resultats/modeles_finaux/ann_final_poids.npz`. L'application les utilise en numpy, sans TensorFlow.

## ⚠️ Limites
Démonstration pédagogique : Macro-F1 test = 0,309 ; accuracy = 0,563.
Valable uniquement pour la zone de Yamoussoukro et des données préparées comme celles du projet.
