"""
Etape 5 : Nettoyage complet -> sauvegarde zones_clean.csv et observations_clean.csv
"""
import pandas as pd
import numpy as np

zones = pd.read_csv("data/zones.csv")
obs = pd.read_csv("data/observations.csv")

# --- 1. Harmoniser les noms de localites ---
harmon = {
    "Guediawaye": "Guédiawaye",
    "Keur-Massar": "Keur Massar",
}
zones["localite"] = zones["localite"].replace(harmon)

# --- 2. Coordonnees inversees (lat/lon swap) ---
mask_inv = (zones.latitude < 5) | (zones.latitude > 20) | (zones.longitude > 0)
print("Zones a corriger (inversion lat/lon):", mask_inv.sum())
lat_ok = zones.loc[mask_inv, "longitude"].copy()
lon_ok = zones.loc[mask_inv, "latitude"].copy()
zones.loc[mask_inv, "latitude"] = lat_ok
zones.loc[mask_inv, "longitude"] = lon_ok

# verification post-correction
bad = zones[(zones.latitude < 12) | (zones.latitude > 17) | (zones.longitude < -18) | (zones.longitude > -11)]
print("Zones encore hors Senegal apres correction:", len(bad))

# --- 3. Altitudes -9999 : absence de donnee satellite -> imputer par mediane de la localite (sinon region) ---
zones["altitude_m"] = zones["altitude_m"].replace(-9999, np.nan)
zones["altitude_m"] = zones.groupby("localite")["altitude_m"].transform(lambda s: s.fillna(s.median()))
zones["altitude_m"] = zones.groupby("region")["altitude_m"].transform(lambda s: s.fillna(s.median()))
print("Altitudes manquantes restantes:", zones.altitude_m.isna().sum())

# --- 4. Valeurs manquantes zones : reseau_assainissement (categoriel), profondeur_nappe (numerique) ---
zones["reseau_assainissement"] = zones["reseau_assainissement"].fillna("Inconnu")
zones["profondeur_nappe_m"] = zones.groupby("localite")["profondeur_nappe_m"].transform(lambda s: s.fillna(s.median()))
zones["profondeur_nappe_m"] = zones["profondeur_nappe_m"].fillna(zones["profondeur_nappe_m"].median())

# --- 5. Doublons zones ---
n_dup_zones = zones.duplicated(subset=["zone_id"]).sum()
zones = zones.drop_duplicates(subset=["zone_id"]).reset_index(drop=True)
print("Doublons zone_id supprimes:", n_dup_zones)

# --- 6. Observations : doublons ---
n_dup_obs = obs.duplicated(subset=["zone_id", "annee"]).sum()
obs = obs.drop_duplicates(subset=["zone_id", "annee"], keep="first").reset_index(drop=True)
print("Doublons observations supprimes:", n_dup_obs)

# --- 7. Valeurs manquantes observations : imputation par mediane de la zone puis de l'annee ---
for col in ["cumul_pluie_hivernage_mm", "pluie_max_24h_mm"]:
    obs[col] = obs.groupby("zone_id")[col].transform(lambda s: s.fillna(s.median()))
    obs[col] = obs.groupby("annee")[col].transform(lambda s: s.fillna(s.median()))
print("NA restants obs:\n", obs[["cumul_pluie_hivernage_mm", "pluie_max_24h_mm"]].isna().sum())

# hauteur_crue_fleuve_m : NaN = non riveraine -> on garde NaN mais on cree un flag
obs["is_riverain"] = obs["hauteur_crue_fleuve_m"].notna().astype(int)

# --- 8. Coherence duree_submersion / inondation (fuite de donnees signalee) ---
incoherent = ((obs.inondation == 0) & (obs.duree_submersion_jours > 0)) | ((obs.inondation == 1) & (obs.duree_submersion_jours == 0))
print("Lignes incoherentes inondation/duree:", incoherent.sum())

zones.to_csv("data/zones_clean.csv", index=False)
obs.to_csv("data/observations_clean.csv", index=False)
print("\nSauvegarde OK :", zones.shape, obs.shape)
