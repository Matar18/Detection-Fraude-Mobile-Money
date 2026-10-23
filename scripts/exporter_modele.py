"""Entraîne le modèle final et exporte les fichiers légers utilisés par l'application.

A lancer depuis la racine du repo, une fois data/paysim.csv en place :
    python scripts/exporter_modele.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from fraude.features import VARIABLES, TYPES_A_RISQUE, preparer  # noqa: E402

SORTIE = RACINE / "modele"
SORTIE.mkdir(exist_ok=True)

types = {"step": "int16", "type": "category", "amount": "float64", "oldbalanceOrg": "float64",
         "isFraud": "int8", "isFlaggedFraud": "int8"}
df = pd.read_csv(RACINE / "data" / "paysim.csv", usecols=list(types), dtype=types)
df["heure"] = df["step"] % 24

# statistiques descriptives pour la page d'accueil
par_type = df.groupby("type", observed=True)["isFraud"].agg(transactions="size", fraudes="sum").reset_index()
par_type.to_csv(SORTIE / "stats_type.csv", index=False)
par_heure = df.groupby("heure")["isFraud"].agg(transactions="size", fraudes="sum").reset_index()
par_heure.to_csv(SORTIE / "stats_heure.csv", index=False)

# même découpage temporel que dans le notebook
d = df[df["type"].isin(TYPES_A_RISQUE) & (df["step"] <= 400)].copy()
train = d[d.step <= 250]
valid = d[(d.step > 250) & (d.step <= 300)]
test = d[d.step > 300]

modele = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                            min_child_samples=200, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=42)
modele.fit(preparer(train), train.isFraud,
           eval_X=(preparer(valid),), eval_y=(valid.isFraud,),
           eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
modele.booster_.save_model(str(SORTIE / "modele_fraude.txt"), num_iteration=modele.best_iteration_)

proba = modele.predict_proba(preparer(test))[:, 1]
y = test.isFraud.values
montant = test.amount.values
nb_jours = test.step.nunique() / 24
perte_totale = (montant * y).sum()

lignes = []
for nom, score in [("probabilite", proba), ("perte_attendue", proba * montant)]:
    ordre = np.argsort(-score)
    y_trie, m_trie = y[ordre], montant[ordre]
    for k in range(5, 401, 5):
        n = int(k * nb_jours)
        lignes.append({"tri": nom, "controles_par_jour": k,
                       "fraudes_detectees": y_trie[:n].sum() / y.sum(),
                       "montant_intercepte": (m_trie[:n] * y_trie[:n]).sum() / perte_totale,
                       "precision": y_trie[:n].mean(),
                       "seuil": float(score[ordre][n - 1])})
pd.DataFrame(lignes).to_csv(SORTIE / "capacite.csv", index=False)

regle = ((test.type == "TRANSFER") & (test.amount > 200_000)).values
resume = {
    "transactions": int(len(df)),
    "fraudes": int(df.isFraud.sum()),
    "bloquees_regle_actuelle": int((df.isFraud & df.isFlaggedFraud).sum()),
    "ap_test": float(average_precision_score(y, proba)),
    "transactions_par_jour_test": float(len(test) / nb_jours),
    "fraudes_par_jour_test": float(y.sum() / nb_jours),
    "regle_large_alertes_par_jour": float(regle.sum() / nb_jours),
    "regle_large_montant_intercepte": float((montant * y)[regle].sum() / perte_totale),
    "regle_large_fraudes_detectees": float(y[regle].sum() / y.sum()),
    "iterations": int(modele.best_iteration_),
}
(SORTIE / "resume.json").write_text(json.dumps(resume, indent=2), encoding="utf-8")

# petit échantillon de la période de test pour la démo de l'application
rng = np.random.default_rng(42)
idx_f = test.index[test.isFraud == 1]
idx_n = test.index[test.isFraud == 0]
echantillon = pd.concat([test.loc[rng.choice(idx_f, 40, replace=False)],
                         test.loc[rng.choice(idx_n, 960, replace=False)]]).sort_values("step")
echantillon[["step", "type", "amount", "oldbalanceOrg", "isFraud"]].to_csv(
    RACINE / "exemples" / "transactions_exemple.csv", index=False)

print(json.dumps(resume, indent=2))
