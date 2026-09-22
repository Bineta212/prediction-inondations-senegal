"""
Etape 1-2 : Chargement, jointure, exploration des defauts, nettoyage
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

zones = pd.read_csv("data/zones.csv")
obs = pd.read_csv("data/observations.csv")

print("=== ZONES ===")
print(zones.shape)
print(zones.isna().sum())
print()
print("=== OBSERVATIONS ===")
print(obs.shape)
print(obs.isna().sum())

print("\n--- Cles ---")
print("zone_id zones uniques:", zones.zone_id.nunique(), "/ lignes:", len(zones))
print("zone_id obs presents dans zones ?", obs.zone_id.isin(zones.zone_id).all())
print("zones sans observation:", set(zones.zone_id) - set(obs.zone_id))

print("\n--- Localites brutes ---")
print(sorted(zones.localite.unique()))

print("\n--- Altitudes suspectes ---")
print((zones.altitude_m == -9999).sum(), "lignes a -9999")

print("\n--- Coordonnees (Senegal approx lat 12-17, lon -18 a -11) ---")
bad_coords = zones[(zones.latitude < 12) | (zones.latitude > 17) | (zones.longitude < -18) | (zones.longitude > -11)]
print(len(bad_coords), "zones hors Senegal (avant correction)")
print(bad_coords[["zone_id","localite","latitude","longitude"]])

print("\n--- Doublons dans observations ---")
print("lignes dupliquees (toutes colonnes):", obs.duplicated().sum())
print("doublons zone_id+annee:", obs.duplicated(subset=["zone_id","annee"]).sum())

print("\n--- type_sol / occupation_sol / reseau_assainissement ---")
print(zones.type_sol.value_counts())
print(zones.occupation_sol.value_counts())
print(zones.reseau_assainissement.value_counts(dropna=False))

print("\n--- Regions ---")
print(zones.region.value_counts())

print("\n--- Taux d'inondation global ---")
print(obs.inondation.mean())
print(obs.groupby("annee").inondation.mean())
