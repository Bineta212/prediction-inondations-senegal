"""
Etape 3-4 : EDA + carte exploratoire Folium
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import folium
from folium.plugins import MarkerCluster

sns.set_theme(style="whitegrid", font_scale=1.0)

# Palette sobre et coherente (bleu sequentiel pour magnitude, gris neutre)
BLUE = "#2563a8"
BLUE_DARK = "#123a63"
GRAY = "#8a94a3"
ACCENT = "#c85a3a"

zones = pd.read_csv("data/zones_clean.csv")
obs = pd.read_csv("data/observations_clean.csv")
df = obs.merge(zones, on="zone_id", how="left")
print("Jointure OK:", df.shape)

# --- 1. Taux d'inondation par annee ---
taux_annee = df.groupby("annee").inondation.mean().reset_index()
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(taux_annee.annee.astype(str), taux_annee.inondation * 100, color=BLUE, width=0.6)
for i, v in enumerate(taux_annee.inondation * 100):
    if taux_annee.annee.iloc[i] == 2020:
        bars[i].set_color(ACCENT)
ax.set_ylabel("Taux d'inondation (%)")
ax.set_title("Taux d'inondation par hivernage (2015-2024)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/01_taux_par_annee.png", dpi=150)
plt.close()
print(taux_annee)

# --- 2. Taux d'inondation par region ---
taux_region = df.groupby("region").inondation.mean().sort_values(ascending=False).reset_index()
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(taux_region.region, taux_region.inondation * 100, color=BLUE)
ax.invert_yaxis()
ax.set_xlabel("Taux d'inondation (%)")
ax.set_title("Taux d'inondation par région (2015-2024)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/02_taux_par_region.png", dpi=150)
plt.close()
print(taux_region)

# --- 3. Taux d'inondation par occupation du sol ---
taux_occ = df.groupby("occupation_sol").inondation.mean().sort_values(ascending=False).reset_index()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(taux_occ.occupation_sol, taux_occ.inondation * 100, color=BLUE)
ax.invert_yaxis()
ax.set_xlabel("Taux d'inondation (%)")
ax.set_title("Taux d'inondation par occupation du sol")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/03_taux_par_occupation.png", dpi=150)
plt.close()
print(taux_occ)

# --- 4. Pluie max 24h vs inondation ---
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df, x="inondation", y="pluie_max_24h_mm", hue="inondation",
            palette={0: GRAY, 1: BLUE}, legend=False, ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Pas d'inondation", "Inondation"])
ax.set_xlabel("")
ax.set_ylabel("Pluie max 24h (mm)")
ax.set_title("Pluie maximale sur 24h selon l'inondation")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/04_pluie24h_vs_inondation.png", dpi=150)
plt.close()

# --- 5. Boxplots altitude / nappe selon la cible ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=df, x="inondation", y="altitude_m", hue="inondation",
            palette={0: GRAY, 1: BLUE}, legend=False, ax=axes[0])
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Non", "Oui"])
axes[0].set_xlabel("Inondation"); axes[0].set_ylabel("Altitude (m)")
axes[0].set_title("Altitude selon l'inondation")
axes[0].spines[["top", "right"]].set_visible(False)

sns.boxplot(data=df, x="inondation", y="profondeur_nappe_m", hue="inondation",
            palette={0: GRAY, 1: BLUE}, legend=False, ax=axes[1])
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Non", "Oui"])
axes[1].set_xlabel("Inondation"); axes[1].set_ylabel("Profondeur nappe (m)")
axes[1].set_title("Profondeur de la nappe selon l'inondation")
axes[1].spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/05_altitude_nappe_vs_inondation.png", dpi=150)
plt.close()

# --- Correlation numerique rapide ---
num_cols = ["altitude_m", "pente_pct", "distance_plan_eau_m", "taux_impermeabilisation_pct",
            "densite_pop_hab_km2", "profondeur_nappe_m", "nb_inondations_2005_2014",
            "cumul_pluie_hivernage_mm", "pluie_max_24h_mm", "pluie_max_72h_mm"]
corr_target = df[num_cols + ["inondation"]].corr(numeric_only=True)["inondation"].drop("inondation").sort_values()
print("\nCorrelation (point-bisériale) avec la cible:\n", corr_target)

# ============ CARTE EXPLORATOIRE FOLIUM (taux historique par zone) ============
hist = df.groupby("zone_id").agg(
    taux_hist=("inondation", "mean"),
    localite=("localite", "first"),
    region=("region", "first"),
    latitude=("latitude", "first"),
    longitude=("longitude", "first"),
).reset_index()

center = [14.55, -14.6]
m = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap")

def color_for_rate(r):
    if r == 0:
        return "#4a9e4a"
    elif r < 0.2:
        return "#d9c94a"
    elif r < 0.4:
        return "#e08a2b"
    else:
        return "#c0392b"

for _, row in hist.iterrows():
    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=4 + row.taux_hist * 10,
        color=color_for_rate(row.taux_hist),
        fill=True,
        fill_color=color_for_rate(row.taux_hist),
        fill_opacity=0.8,
        weight=1,
        popup=folium.Popup(
            f"<b>{row.localite}</b> ({row.region})<br>Zone: {row.zone_id}<br>"
            f"Taux d'inondation historique (2015-2024): {row.taux_hist*100:.0f}%",
            max_width=250,
        ),
    ).add_to(m)

legend_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background:white;
padding:10px 14px; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,0.3); font-size:13px;">
<b>Taux d'inondation historique (2015-2024)</b><br>
<span style="color:#4a9e4a">&#9679;</span> 0 %<br>
<span style="color:#d9c94a">&#9679;</span> 1-19 %<br>
<span style="color:#e08a2b">&#9679;</span> 20-39 %<br>
<span style="color:#c0392b">&#9679;</span> &ge; 40 %
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
m.save("outputs/carte_exploratoire_taux_historique.html")
print("\nCarte exploratoire sauvegardee.")
