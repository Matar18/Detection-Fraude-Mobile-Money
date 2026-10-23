"""Calcul des variables utilisées par le modèle.

Ce sont uniquement des informations connues au moment où le client lance
l'opération : montant, type, heure et solde avant l'opération.
"""
import numpy as np
import pandas as pd

VARIABLES = ["log_montant", "transfert", "heure", "nuit", "log_solde_avant",
             "vide_le_compte", "solde_nul", "part_du_solde"]

TYPES_A_RISQUE = ["TRANSFER", "CASH_OUT"]


def preparer(df: pd.DataFrame) -> pd.DataFrame:
    """Attend les colonnes type, amount, oldbalanceOrg et step (ou heure)."""
    x = pd.DataFrame(index=df.index)
    heure = df["heure"] if "heure" in df else df["step"] % 24
    x["log_montant"] = np.log1p(df["amount"])
    x["transfert"] = (df["type"] == "TRANSFER").astype(int)
    x["heure"] = heure.astype(int)
    x["nuit"] = heure.between(0, 7).astype(int)
    x["log_solde_avant"] = np.log1p(df["oldbalanceOrg"])
    x["vide_le_compte"] = ((df["oldbalanceOrg"] > 0) & (df["amount"] >= df["oldbalanceOrg"])).astype(int)
    x["solde_nul"] = (df["oldbalanceOrg"] == 0).astype(int)
    x["part_du_solde"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    return x[VARIABLES]
