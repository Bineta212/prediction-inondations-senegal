"""
Etape 11 : Cartographie des risques sous scenario de pluie extreme (120 mm/24h)
"""
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import Fullscreen, MiniMap

bundle = joblib.load("outputs/best_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]
threshold = bundle["threshold"]
model_name = bundle["model_name"]

zones = pd.read_csv("data/zones_clean.csv")
df_full = pd.read_csv("data/dataset_model.csv")

# --- Construire le scenario : pluie extreme 120mm/24h, coherente avec 72h et cumul ---
SCENARIO_PLUIE_24H = 120.0
scenario = zones.copy()
scenario["annee"] = 2025
scenario["pluie_max_24h_mm"] = SCENARIO_PLUIE_24H
scenario["pluie_max_72h_mm"] = SCENARIO_PLUIE_24H * 1.6   # hypothese : evenement concentre sur 3 jours
scenario["nb_jours_pluie"] = df_full.groupby("zone_id")["nb_jours_pluie"].mean().reindex(scenario.zone_id).values
scenario["cumul_pluie_hivernage_mm"] = df_full.groupby("zone_id")["cumul_pluie_hivernage_mm"].mean().reindex(scenario.zone_id).values
# hauteur de crue : on majore de 40% pour les zones riveraines historiquement renseignees
hist_crue = df_full.groupby("zone_id")["hauteur_crue_fleuve_m"].mean()
scenario["hauteur_crue_fleuve_m"] = (hist_crue.reindex(scenario.zone_id).fillna(0).values) * 1.4
scenario["is_riverain"] = (hist_crue.reindex(scenario.zone_id).fillna(0).values > 0).astype(int)

# --- Recalcul des variables dependantes de la pluie ---
scenario["indice_pluie_impermeabilisation"] = scenario["pluie_max_24h_mm"] * scenario["taux_impermeabilisation_pct"] / 100
scenario["log_distance_plan_eau"] = np.log1p(scenario["distance_plan_eau_m"])
med_alt_localite = scenario.groupby("localite")["altitude_m"].transform("median")
scenario["zone_basse"] = (scenario["altitude_m"] < med_alt_localite).astype(int)
assain_map = {"Aucun": 0, "Partiel": 1, "Complet": 2, "Inconnu": 1}
scenario["assainissement_score"] = scenario["reseau_assainissement"].map(assain_map)
scenario["historique_frequence"] = scenario["nb_inondations_2005_2014"] / 10.0

scenario_enc = pd.get_dummies(scenario, columns=["type_sol", "occupation_sol", "reseau_assainissement"],
                               prefix=["sol", "occ", "assain"], drop_first=False)

# aligner les colonnes sur celles utilisees a l'entrainement
for c in feature_cols:
    if c not in scenario_enc.columns:
        scenario_enc[c] = 0
X_scenario = scaler.transform(scenario_enc[feature_cols])

proba_scenario = model.predict_proba(X_scenario)[:, 1]
scenario["probabilite_inondation"] = proba_scenario

# --- Classification en 4 niveaux de risque ---
def niveau_risque(p):
    if p < 0.25:
        return "Faible"
    elif p < 0.50:
        return "Modéré"
    elif p < 0.75:
        return "Élevé"
    else:
        return "Très élevé"

scenario["niveau_risque"] = scenario["probabilite_inondation"].apply(niveau_risque)
print(scenario["niveau_risque"].value_counts())
print("\nProbabilite moyenne par region:")
print(scenario.groupby("region")["probabilite_inondation"].mean().sort_values(ascending=False))

# --- Facteur dominant par zone (pour l'info-bulle) : plus grande contribution parmi les facteurs cles ---
if hasattr(model, "coef_"):
    coefs = pd.Series(model.coef_[0], index=feature_cols)
    key_factors = ["pluie_max_24h_mm", "taux_impermeabilisation_pct", "altitude_m", "profondeur_nappe_m",
                   "assainissement_score", "historique_frequence", "hauteur_crue_fleuve_m", "zone_basse"]
    contrib = X_scenario_df = pd.DataFrame(X_scenario, columns=feature_cols)[key_factors] * coefs[key_factors].values
    scenario["facteur_dominant"] = contrib.abs().idxmax(axis=1)

facteur_labels = {
    "pluie_max_24h_mm": "Pluie extrême (24h)",
    "taux_impermeabilisation_pct": "Imperméabilisation",
    "altitude_m": "Faible altitude",
    "profondeur_nappe_m": "Nappe phréatique proche",
    "assainissement_score": "Assainissement",
    "historique_frequence": "Historique d'inondations",
    "hauteur_crue_fleuve_m": "Crue du fleuve",
    "zone_basse": "Zone basse (relative)",
}
scenario["facteur_dominant_label"] = scenario["facteur_dominant"].map(facteur_labels)

scenario.to_csv("outputs/scenario_risque_120mm.csv", index=False)

# ================= CARTE NATIONALE =================
risk_colors = {"Faible": "#4a9e4a", "Modéré": "#d9c94a", "Élevé": "#e08a2b", "Très élevé": "#c0392b"}

def build_map(data, center, zoom):
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
    Fullscreen().add_to(m)
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=5 + row.probabilite_inondation * 9,
            color=risk_colors[row.niveau_risque],
            fill=True, fill_color=risk_colors[row.niveau_risque], fill_opacity=0.85, weight=1,
            popup=folium.Popup(
                f"<b>{row.localite}</b> ({row.region})<br>"
                f"Zone : {row.zone_id}<br>"
                f"Probabilité d'inondation (scénario 120mm/24h) : <b>{row.probabilite_inondation*100:.0f}%</b><br>"
                f"Niveau de risque : <b>{row.niveau_risque}</b><br>"
                f"Facteur dominant : {row.facteur_dominant_label}",
                max_width=280,
            ),
        ).add_to(m)
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background:white;
    padding:10px 14px; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,0.3); font-size:13px;">
    <b>Niveau de risque</b><br>
    <span style="color:{risk_colors['Faible']}">&#9679;</span> Faible (&lt;25%)<br>
    <span style="color:{risk_colors['Modéré']}">&#9679;</span> Modéré (25-49%)<br>
    <span style="color:{risk_colors['Élevé']}">&#9679;</span> Élevé (50-74%)<br>
    <span style="color:{risk_colors['Très élevé']}">&#9679;</span> Très élevé (&ge;75%)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

m_national = build_map(scenario, center=[14.55, -14.6], zoom=7)
m_national.save("outputs/carte_risque_national_scenario_extreme.html")
print("\nCarte nationale sauvegardee.")

# ================= ZOOM BANLIEUE DAKAR =================
dakar_localites = ["Pikine", "Guédiawaye", "Keur Massar", "Yeumbeul", "Thiaroye", "Malika", "Jaxaay", "Diamaguène", "Mbao"]
dakar_zones = scenario[scenario.localite.isin(dakar_localites)]
print("\nZones banlieue Dakar:", len(dakar_zones))
m_dakar = build_map(dakar_zones, center=[dakar_zones.latitude.mean(), dakar_zones.longitude.mean()], zoom=12)
m_dakar.save("outputs/carte_risque_banlieue_dakar.html")
print("Carte banlieue Dakar sauvegardee.")

# --- Classement des localites prioritaires ---
priorite = scenario.groupby("localite").agg(
    region=("region", "first"),
    n_zones=("zone_id", "count"),
    proba_moyenne=("probabilite_inondation", "mean"),
    pct_tres_eleve=("niveau_risque", lambda s: (s == "Très élevé").mean() * 100),
    pop_totale_proxy=("densite_pop_hab_km2", "sum"),
).sort_values("proba_moyenne", ascending=False)
priorite.to_csv("outputs/localites_prioritaires.csv")
print("\n=== TOP 15 localites prioritaires (scenario extreme) ===")
print(priorite.head(15).round(2))
