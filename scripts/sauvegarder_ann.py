# -*- coding: utf-8 -*-
"""
Reconstruction et sauvegarde de l'ANN final (pour l'application Streamlit).

L'ANN n'avait pas été sauvegardé dans les notebooks : ce script le ré-entraîne avec EXACTEMENT
la recette de « experience A.ipynb » (expérience F, configuration F3) et config_ann.json :
  - mêmes données (donnees_propres.csv) et même partition 70/15/15 (seed 42, vérifiée avec partitions.csv)
  - StandardScaler appris sur l'entraînement
  - ANN-3 : 19 -> 128 -> 64 -> 32 -> 8, GELU, softmax
  - SGD + Momentum (lr = 0,01 ; momentum = 0,5), lots de 32, 200 epochs max
  - arrêt anticipé sur val_loss (patience 20, meilleurs poids restaurés)
Les 5 graines redonnent les mêmes epochs et Macro-F1 de validation que resultats_detailles.csv.
On sauvegarde la graine 0 (comme le Random Forest) au format .npz (poids numpy) :
l'application n'a donc pas besoin de TensorFlow.

Lancement (depuis la racine du projet) : python scripts/sauvegarder_ann.py
"""
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DOSSIER = os.path.join("resultats", "modeles_finaux")
data = pd.read_csv("donnees_propres.csv")
cfg = json.load(open(os.path.join(DOSSIER, "config_ann.json"), encoding="utf-8"))
VARIABLES = cfg["variables"]

# Encodage des classes et partition identique aux notebooks
encodeur = LabelEncoder()
y = encodeur.fit_transform(data["class_name"])
idx = np.arange(len(y))
idx_train, idx_temp = train_test_split(idx, test_size=0.30, stratify=y, random_state=cfg["seed_partition"])
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, stratify=y[idx_temp], random_state=cfg["seed_partition"])
partitions = pd.read_csv(os.path.join("resultats", "experience_F", "partitions.csv"))
ensemble = np.empty(len(y), dtype=object)
ensemble[idx_train], ensemble[idx_val], ensemble[idx_test] = "train", "validation", "test"
assert (partitions["ensemble"].to_numpy() == ensemble).all(), "Partition différente !"

X = data[VARIABLES].to_numpy(dtype="float32")
normaliseur = StandardScaler().fit(X[idx_train])
X_train, X_val, X_test = (normaliseur.transform(X[i]) for i in (idx_train, idx_val, idx_test))


def construire():
    modele = keras.Sequential([keras.Input(shape=(len(VARIABLES),))])
    for n in cfg["couches_cachees"]:
        modele.add(layers.Dense(n))
        modele.add(layers.Activation(cfg["activation"]))
    modele.add(layers.Dense(len(encodeur.classes_), activation="softmax"))
    modele.compile(optimizer=keras.optimizers.SGD(learning_rate=cfg["taux_apprentissage"], momentum=cfg["momentum"]),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modele


resultats = []
for graine in cfg["graines"]:
    keras.utils.set_random_seed(graine)
    modele = construire()
    arret = keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["arret_anticipe"]["patience"],
                                          restore_best_weights=True)
    hist = modele.fit(X_train, y[idx_train], validation_data=(X_val, y[idx_val]), epochs=cfg["epochs_max"],
                      batch_size=cfg["taille_lot"], callbacks=[arret], verbose=0)
    p_val = modele.predict(X_val, verbose=0).argmax(1)
    p_test = modele.predict(X_test, verbose=0).argmax(1)
    resultats.append({"graine": graine, "epochs": len(hist.history["loss"]),
                      "val_f1_macro": f1_score(y[idx_val], p_val, average="macro"),
                      "test_f1_macro": f1_score(y[idx_test], p_test, average="macro"),
                      "test_accuracy": accuracy_score(y[idx_test], p_test),
                      "test_mcc": matthews_corrcoef(y[idx_test], p_test)})
    print(resultats[-1])

    if graine == 0:  # sauvegarde des poids de la graine 0 + paramètres du StandardScaler
        denses = [c for c in modele.layers if isinstance(c, layers.Dense)]
        poids = {}
        for i, couche in enumerate(denses):
            W, b = couche.get_weights()
            poids[f"W{i}"], poids[f"b{i}"] = W, b
        np.savez_compressed(os.path.join(DOSSIER, "ann_final_poids.npz"),
                            moyenne=normaliseur.mean_, ecart=normaliseur.scale_, **poids)

r = pd.DataFrame(resultats)
print(r.round(3))
print("Moyenne :\n", r.mean().round(3))
