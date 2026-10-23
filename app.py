import json
from pathlib import Path

import altair as alt
import lightgbm as lgb
import pandas as pd
import streamlit as st

from fraude.features import TYPES_A_RISQUE, preparer

RACINE = Path(__file__).parent
MODELE = RACINE / "modele"
BLEU, ROUGE, GRIS = "#2b5d8a", "#c0392b", "#9aa5b1"

st.set_page_config(page_title="Détection de fraude mobile money", layout="wide")


@st.cache_resource
def charger_modele():
    return lgb.Booster(model_file=str(MODELE / "modele_fraude.txt"))


@st.cache_data
def charger_donnees():
    resume = json.loads((MODELE / "resume.json").read_text(encoding="utf-8"))
    return (resume,
            pd.read_csv(MODELE / "capacite.csv"),
            pd.read_csv(MODELE / "stats_type.csv"),
            pd.read_csv(MODELE / "stats_heure.csv"),
            pd.read_csv(RACINE / "exemples" / "transactions_exemple.csv"))


def nombre(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", " ")


def pourcent(x, dec=0):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def scorer(df):
    out = df.copy()
    out["probabilite"] = 0.0
    a_risque = out["type"].isin(TYPES_A_RISQUE)
    if a_risque.any():
        out.loc[a_risque, "probabilite"] = modele.predict(preparer(out[a_risque]))
    out["perte_attendue"] = out["probabilite"] * out["amount"]
    return out


modele = charger_modele()
resume, capacite, stats_type, stats_heure, exemple = charger_donnees()

st.title("Détection de fraude sur des transactions mobile money")
st.write("Projet réalisé par Matar Ndoye Keita, MBA Data & IA (ISM Dakar). "
         "Modèle entraîné sur PaySim, un jeu de données simulé à partir d'un mois de transactions "
         "réelles d'un service de mobile money en Afrique.")

onglets = st.tabs(["Résumé", "Tester une transaction", "Capacité de contrôle", "Analyser un fichier"])

# Résumé
with onglets[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analysées", nombre(resume["transactions"]))
    c2.metric("Fraudes", nombre(resume["fraudes"]))
    c3.metric("Bloquées par la règle actuelle", nombre(resume["bloquees_regle_actuelle"]))
    c4.metric("Précision moyenne du modèle (AP)", f"{resume['ap_test']:.2f}".replace(".", ","))

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Fraudes par type d'opération")
        t = stats_type.assign(a_fraude=stats_type.fraudes > 0)
        st.altair_chart(
            alt.Chart(t).mark_bar().encode(
                x=alt.X("transactions:Q", title="Transactions"),
                y=alt.Y("type:N", sort="-x", title=None),
                color=alt.Color("a_fraude:N", scale=alt.Scale(domain=[True, False], range=[ROUGE, GRIS]),
                                legend=None),
                tooltip=["type", alt.Tooltip("transactions:Q", format=","), alt.Tooltip("fraudes:Q", format=",")],
            ).properties(height=260),
            width="stretch")
        st.caption("Seuls les transferts et les retraits contiennent des fraudes.")
    with g2:
        st.subheader("Part de fraude selon l'heure")
        h = stats_heure.assign(part=stats_heure.fraudes / stats_heure.transactions * 100)
        st.altair_chart(
            alt.Chart(h).mark_line(point=True, color=ROUGE).encode(
                x=alt.X("heure:O", title="Heure"),
                y=alt.Y("part:Q", title="Part de fraude (%)"),
                tooltip=["heure", alt.Tooltip("part:Q", format=".2f"), "fraudes", "transactions"],
            ).properties(height=260),
            width="stretch")
        st.caption("La nuit, l'activité normale s'arrête mais pas la fraude.")

    st.subheader("Ce qu'il faut retenir")
    st.write(
        "Le fraudeur transfère presque toujours tout le solde du compte, puis le retire. "
        f"La règle en place (blocage des gros transferts) n'a arrêté que {resume['bloquees_regle_actuelle']} fraudes "
        f"sur {nombre(resume['fraudes'])}. Une version plus large de cette règle produirait environ "
        f"{nombre(resume['regle_large_alertes_par_jour'])} alertes par jour. Le modèle permet au contraire de "
        "concentrer un petit nombre de contrôles sur les opérations les plus risquées.")
    st.caption("Limite : ce sont des données simulées, où les fraudeurs suivent tous le même scénario. "
               "Sur des données réelles, les scores seraient nettement plus bas.")

# Tester une transaction
with onglets[1]:
    st.write("Renseigne une opération pour obtenir son score de risque.")
    f1, f2 = st.columns(2)
    with f1:
        type_op = st.selectbox("Type d'opération", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                               format_func=lambda t: {"TRANSFER": "Transfert", "CASH_OUT": "Retrait",
                                                      "PAYMENT": "Paiement", "CASH_IN": "Dépôt",
                                                      "DEBIT": "Débit"}[t])
        heure = st.slider("Heure de l'opération", 0, 23, 3)
    with f2:
        solde = st.number_input("Solde du compte avant l'opération", min_value=0.0, value=850_000.0, step=10_000.0)
        montant = st.number_input("Montant", min_value=0.0, value=850_000.0, step=10_000.0)

    if type_op not in TYPES_A_RISQUE:
        st.info("Aucune fraude n'a été observée sur ce type d'opération dans les données. "
                "Le modèle ne s'applique qu'aux transferts et aux retraits.")
    else:
        res = scorer(pd.DataFrame([{"type": type_op, "amount": montant, "oldbalanceOrg": solde, "heure": heure}]))
        proba, perte = res.probabilite.iloc[0], res.perte_attendue.iloc[0]
        seuil_100 = capacite.query("tri == 'perte_attendue' and controles_par_jour == 100").seuil.iloc[0]
        r1, r2 = st.columns(2)
        r1.metric("Probabilité de fraude", pourcent(proba, 1))
        r2.metric("Perte attendue", nombre(perte))
        if perte >= seuil_100:
            st.warning("Avec une équipe qui contrôle 100 opérations par jour, celle-ci ferait partie des contrôles.")
        else:
            st.write("Avec une équipe qui contrôle 100 opérations par jour, celle-ci ne serait pas contrôlée.")
        if solde > 0 and montant >= solde:
            st.caption("L'opération vide le compte, c'est le signal le plus fort dans ces données.")

# Capacité de contrôle
with onglets[2]:
    st.write(f"Sur la période de test, il y a environ {nombre(resume['transactions_par_jour_test'])} transferts "
             f"et retraits par jour, dont {nombre(resume['fraudes_par_jour_test'])} fraudes. "
             "Une équipe ne peut en contrôler qu'une petite partie. "
             "On contrôle d'abord les opérations avec le score le plus élevé.")
    k = st.slider("Nombre de contrôles par jour", 5, 400, 100, step=5)
    lp = capacite.query("tri == 'perte_attendue' and controles_par_jour == @k").iloc[0]
    lq = capacite.query("tri == 'probabilite' and controles_par_jour == @k").iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Montant fraudé intercepté", pourcent(lp.montant_intercepte),
              delta=f"{(lp.montant_intercepte - lq.montant_intercepte) * 100:+.0f} pts vs tri par probabilité")
    m2.metric("Fraudes détectées", pourcent(lp.fraudes_detectees))
    m3.metric("Alertes qui sont de vraies fraudes", pourcent(lp.precision))

    courbe = capacite.assign(
        montant=capacite.montant_intercepte * 100,
        methode=capacite.tri.map({"perte_attendue": "tri par perte attendue", "probabilite": "tri par probabilité"}))
    base = alt.Chart(courbe).mark_line(strokeWidth=2.5).encode(
        x=alt.X("controles_par_jour:Q", title="Contrôles par jour"),
        y=alt.Y("montant:Q", title="Montant fraudé intercepté (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("methode:N", scale=alt.Scale(range=[BLEU, GRIS]), title=None,
                        legend=alt.Legend(orient="bottom")),
        tooltip=["controles_par_jour", alt.Tooltip("montant:Q", format=".1f"), "methode"])
    regle = alt.Chart(pd.DataFrame({"y": [resume["regle_large_montant_intercepte"] * 100]})).mark_rule(
        strokeDash=[5, 4], strokeWidth=1.5).encode(y="y:Q", color=alt.value(ROUGE))
    repere = alt.Chart(pd.DataFrame({"x": [k]})).mark_rule(color="#555").encode(x="x:Q")
    st.altair_chart((base + regle + repere).properties(height=340), width="stretch")
    st.caption(f"Ligne rouge : règle « transfert de plus de 200 000 », qui demande environ "
               f"{nombre(resume['regle_large_alertes_par_jour'])} contrôles par jour.")

# Analyser un fichier
with onglets[3]:
    st.write("Charge un fichier CSV de transactions pour obtenir la liste des opérations à contrôler en priorité. "
             "Colonnes attendues : type, amount, oldbalanceOrg, et step ou heure.")
    fichier = st.file_uploader("Fichier CSV", type="csv")
    if fichier is not None:
        data = pd.read_csv(fichier)
    else:
        st.caption("Sans fichier, l'application utilise un exemple de 1 000 transactions de la période de test.")
        data = exemple

    if True:
        manquantes = {"type", "amount", "oldbalanceOrg"} - set(data.columns)
        if manquantes or not ({"step", "heure"} & set(data.columns)):
            st.error("Il manque des colonnes : " + ", ".join(sorted(manquantes | ({"step ou heure"}
                     if not ({"step", "heure"} & set(data.columns)) else set()))))
        else:
            res = scorer(data).sort_values("perte_attendue", ascending=False)
            n = st.number_input("Nombre d'opérations à contrôler", 1, len(res), min(40, len(res)))
            top = res.head(int(n))
            st.write(f"{len(res)} opérations analysées. Les {int(n)} premières représentent "
                     f"{pourcent(top.perte_attendue.sum() / max(res.perte_attendue.sum(), 1))} de la perte attendue totale.")
            if "isFraud" in res:
                st.write(f"Fraudes réelles dans cette sélection : {int(top.isFraud.sum())} "
                         f"sur {int(res.isFraud.sum())} présentes dans le fichier.")
            affichage = top.assign(probabilite=(top.probabilite * 100).round(1),
                                   perte_attendue=top.perte_attendue.round(0))
            st.dataframe(affichage, width="stretch", hide_index=True)
            st.download_button("Télécharger les résultats", res.to_csv(index=False).encode("utf-8"),
                               "transactions_scorees.csv", "text/csv")
