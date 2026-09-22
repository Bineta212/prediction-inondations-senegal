"""
Etape 6 : Ingenierie des variables
"""
import pandas as pd
import numpy as np

zones = pd.read_csv("data/zones_clean.csv")
obs = pd.read_csv("data/observations_clean.csv")

df = obs.merge(zones, on="zone_id", how="left")

# --- log(distance au plan d'eau) ---
df["log_distance_plan_eau"] = np.log1p(df["distance_plan_eau_m"])

# --- indicateur "zone basse" : altitude relative a la mediane de sa localite ---
med_alt_localite = df.groupby("localite")["altitude_m"].transform("median")
df["zone_basse"] = (df["altitude_m"] < med_alt_localite).astype(int)

# --- indice pluie x impermeabilisation ---
df["indice_pluie_impermeabilisation"] = df["pluie_max_24h_mm"] * df["taux_impermeabilisation_pct"] / 100

# --- indice de vulnerabilite du sol (assainissement) ---
assain_map = {"Aucun": 0, "Partiel": 1, "Complet": 2, "Inconnu": 1}
df["assainissement_score"] = df["reseau_assainissement"].map(assain_map)

# --- historique normalise ---
df["historique_frequence"] = df["nb_inondations_2005_2014"] / 10.0

# --- encodage categoriel (one-hot) pour type_sol / occupation_sol ---
df = pd.get_dummies(df, columns=["type_sol", "occupation_sol", "reseau_assainissement"],
                     prefix=["sol", "occ", "assain"], drop_first=False)

# On EXCLUT duree_submersion_jours (fuite de donnees) et hauteur_crue_fleuve_m brute (trop de NaN, gardee via is_riverain)
df["hauteur_crue_fleuve_m"] = df["hauteur_crue_fleuve_m"].fillna(0)

leak_cols = ["duree_submersion_jours"]
df = df.drop(columns=leak_cols)

df.to_csv("data/dataset_model.csv", index=False)
print("Dataset final:", df.shape)
print(df.columns.tolist())
