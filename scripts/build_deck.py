import json, os

ROOT = "/home/claude/flood_project/deck/project"
os.makedirs(f"{ROOT}/slides", exist_ok=True)

# ---------- Palette ----------
BG_LIGHT = "#fbfbf8"
BG_DARK = "#0f1e33"
TEXT_DARK = "#16213d"
TEXT_MUTED = "#525d6b"
TEXT_LIGHT = "#f2f4f7"
TEXT_LIGHT_MUTED = "#aab6c6"
ACCENT = "#2563a8"
ACCENT_DARK = "#123a63"
ACCENT2 = "#c85a3a"
CARD_BG = "#ffffff"
BORDER = "#e3e6e4"
GREEN = "#4a9e4a"
YELLOW = "#c9a227"
ORANGE = "#e08a2b"
RED = "#c0392b"

FONT_LINK = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap"
FONT = "'IBM Plex Sans', Arial, sans-serif"

def eyebrow(text, color=ACCENT):
    return (f'<p style="font-family:{FONT}; font-size:24px; font-weight:600; '
            f'letter-spacing:2px; text-transform:uppercase; color:{color}; margin:0">{text}</p>')

def pagefoot(n, dark=False):
    color = TEXT_LIGHT_MUTED if dark else TEXT_MUTED
    return (f'<p style="position:absolute; right:64px; bottom:56px; font-size:22px; '
            f'color:{color}">{n} / 13</p>')

slides = {}
order = []

def add(sid, html):
    slides[sid] = html
    order.append(sid)

# ============================================================ 1. COVER
add("cover", f'''<section id="cover" data-transition="fade" style="background:linear-gradient(160deg, {BG_DARK} 0%, #182c4d 100%); color:{TEXT_LIGHT}; font-family:{FONT}; padding:128px; display:flex; flex-direction:column; justify-content:space-between">
  <div style="display:flex; flex-direction:column; gap:24px">
    {eyebrow("Projet Machine Learning — Secteur Environnement", "#7fb2e8")}
    <h1 style="font-size:96px; font-weight:700; line-height:1.08; max-width:1500px; margin:0">Prédiction et cartographie des risques d'inondation au Sénégal</h1>
    <p style="font-size:36px; font-weight:400; color:{TEXT_LIGHT_MUTED}; max-width:1300px; margin:0">Un modèle de classification pour anticiper les zones inondables et cibler les actions de l'ONAS, de la Protection civile et des collectivités.</p>
  </div>
  <div style="display:flex; justify-content:space-between; align-items:flex-end">
    <p style="font-size:26px; color:{TEXT_LIGHT_MUTED}">425 zones · 38 localités · hivernages 2015-2024</p>
    <p style="font-size:26px; color:{TEXT_LIGHT_MUTED}">Septembre 2026</p>
  </div>
</section>''')

# ============================================================ 2. CONTEXTE
add("contexte", f'''<section id="contexte" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:40px">
  {eyebrow("Contexte")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">Un enjeu qui revient chaque hivernage</h2>
  <div style="display:flex; gap:56px; align-items:flex-start">
    <div style="flex:1.3; display:flex; flex-direction:column; gap:20px">
      <p style="font-size:30px; line-height:1.5; color:{TEXT_DARK}; margin:0">Pikine, Guédiawaye, Keur Massar, Yeumbeul, mais aussi Touba, Kaolack, Saint-Louis ou Ziguinchor : l'urbanisation des zones basses et des Niayes, la remontée de la nappe phréatique, l'assainissement insuffisant et les pluies extrêmes exposent le pays chaque année.</p>
      <p style="font-size:30px; line-height:1.5; color:{TEXT_MUTED}; margin:0">Anticiper ces zones permet de cibler les pompages, le curage des canaux et l'alerte des populations avant la saison des pluies.</p>
    </div>
    <div style="flex:1; background:{CARD_BG}; border:1px solid {BORDER}; border-radius:20px; padding:48px; display:flex; flex-direction:column; gap:12px; align-items:center; justify-content:center">
      <p style="font-size:88px; font-weight:700; color:{ACCENT}; margin:0">21,6 %</p>
      <p style="font-size:26px; color:{TEXT_MUTED}; text-align:center; margin:0">des zones inondées en moyenne chaque hivernage (2015-2024)</p>
    </div>
  </div>
  {pagefoot(2)}
</section>''')

# ============================================================ 3. OBJECTIF / CADRAGE
add("objectif", f'''<section id="objectif" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:40px">
  {eyebrow("Objectif et cadrage")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">Prédire, puis cartographier le risque</h2>
  <p style="font-size:30px; line-height:1.5; color:{TEXT_DARK}; margin:0; max-width:1600px">Estimer la probabilité qu'une zone soit inondée pendant l'hivernage — puis classer les zones en 4 niveaux de risque sur une carte, à l'échelle nationale et pour la banlieue de Dakar.</p>
  <div style="display:flex; gap:32px; margin-top:16px">
    <div style="flex:1; background:{CARD_BG}; border:1px solid {BORDER}; border-radius:20px; padding:40px; display:flex; flex-direction:column; gap:16px">
      <h3 style="font-size:32px; font-weight:600; color:{ACCENT}; margin:0">Type de problème</h3>
      <p style="font-size:27px; line-height:1.5; color:{TEXT_MUTED}; margin:0">Classification binaire supervisée (inondation oui/non) + analyse spatiale</p>
    </div>
    <div style="flex:1; background:{CARD_BG}; border:1px solid {BORDER}; border-radius:20px; padding:40px; display:flex; flex-direction:column; gap:16px">
      <h3 style="font-size:32px; font-weight:600; color:{ACCENT2}; margin:0">Coût des erreurs</h3>
      <p style="font-size:27px; line-height:1.5; color:{TEXT_MUTED}; margin:0">Un faux négatif (zone inondée non anticipée) coûte bien plus qu'une fausse alerte → <b>le rappel est la métrique prioritaire</b> (objectif ≥ 80 %)</p>
    </div>
  </div>
  {pagefoot(3)}
</section>''')

# ============================================================ 4. DONNEES
add("donnees", f'''<section id="donnees" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:40px">
  {eyebrow("Données")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">Deux fichiers, une clé commune : zone_id</h2>
  <table style="font-family:{FONT}; font-size:28px; color:{TEXT_DARK}; border-radius:16px; overflow:hidden">
    <tr style="background:{ACCENT_DARK}">
      <th style="width:34%; color:#ffffff; text-align:left; padding:20px 24px">Fichier</th>
      <th style="width:22%; color:#ffffff; text-align:left; padding:20px 24px">Granularité</th>
      <th style="width:44%; color:#ffffff; text-align:left; padding:20px 24px">Contenu</th>
    </tr>
    <tr style="background:{CARD_BG}">
      <td style="padding:20px 24px">zones.csv (425 lignes)</td>
      <td style="padding:20px 24px">1 ligne / zone</td>
      <td style="padding:20px 24px; color:{TEXT_MUTED}">Altitude, pente, sol, occupation du sol, imperméabilisation, densité, assainissement, nappe, historique 2005-2014</td>
    </tr>
    <tr style="background:#f1f3f5">
      <td style="padding:20px 24px">observations.csv (4 275 lignes)</td>
      <td style="padding:20px 24px">1 ligne / zone × année</td>
      <td style="padding:20px 24px; color:{TEXT_MUTED}">Pluviométrie (cumul, 24h, 72h), crue du fleuve, cible <i>inondation</i>, durée de submersion</td>
    </tr>
  </table>
  <p style="font-size:27px; color:{TEXT_MUTED}; margin:0">38 localités · 14 régions · hivernages 2015 à 2024 · données synthétiques à visée pédagogique</p>
  {pagefoot(4)}
</section>''')

# ============================================================ 5. NETTOYAGE
def defect_row(label, ampleur, fix, bg):
    return (f'<tr style="background:{bg}"><td style="padding:16px 24px">{label}</td>'
            f'<td style="padding:16px 24px; text-align:center">{ampleur}</td>'
            f'<td style="padding:16px 24px; color:{TEXT_MUTED}">{fix}</td></tr>')

add("nettoyage", f'''<section id="nettoyage" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:36px">
  {eyebrow("Nettoyage")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">Six défauts détectés et corrigés</h2>
  <table style="font-family:{FONT}; font-size:25px; color:{TEXT_DARK}; border-radius:16px; overflow:hidden">
    <tr style="background:{ACCENT_DARK}">
      <th style="width:34%; color:#fff; text-align:left; padding:16px 24px">Défaut</th>
      <th style="width:14%; color:#fff; text-align:center; padding:16px 24px">Ampleur</th>
      <th style="width:52%; color:#fff; text-align:left; padding:16px 24px">Correction</th>
    </tr>
    {defect_row("Altitudes à -9999", "7 zones", "Imputation par médiane locale/régionale", CARD_BG)}
    {defect_row("Coordonnées inversées", "4 zones", "Détection hors Sénégal, permutation lat/lon", "#f1f3f5")}
    {defect_row("Localités non harmonisées", "2 variantes", "Fusion vers le nom officiel", CARD_BG)}
    {defect_row("Valeurs manquantes", "3-16 %", "Imputation médiane/mode (zone, année)", "#f1f3f5")}
    {defect_row("Observations dupliquées", "25 lignes", "Suppression (zone_id + année)", CARD_BG)}
    {defect_row("Fuite de données (durée submersion)", "1 variable", "Exclue des variables explicatives", "#f1f3f5")}
  </table>
  <p style="font-size:26px; color:{TEXT_MUTED}; margin:0">Après nettoyage : <b style="color:{TEXT_DARK}">425 zones</b> · <b style="color:{TEXT_DARK}">4 250 observations</b> valides</p>
  {pagefoot(5)}
</section>''')

# ============================================================ 6. EDA (bar chart)
annees = [("2015",0.200000),("2016",0.185882),("2017",0.195294),("2018",0.192941),
          ("2019",0.190588),("2020",0.289412),("2021",0.232941),("2022",0.247059),
          ("2023",0.209412),("2024",0.221176)]
MAXH = 230
bars_html = []
for yr, v in annees:
    h = round(v/0.30*MAXH)
    color = ACCENT2 if yr == "2020" else ACCENT
    bars_html.append(f'''<div style="display:flex; flex-direction:column; align-items:center; gap:10px; width:120px">
      <p style="font-size:22px; font-weight:600; color:{color}; margin:0">{v*100:.0f}%</p>
      <div style="width:64px; height:{h}px; background:{color}; border-radius:8px 8px 0 0"></div>
      <p style="font-size:22px; color:{TEXT_MUTED}; margin:0">{yr}</p>
    </div>''')
bars_row = "\n".join(bars_html)

add("eda", f'''<section id="eda" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:32px">
  {eyebrow("Exploration des données")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">2020, l'hivernage le plus touché</h2>
  <p style="font-size:28px; color:{TEXT_MUTED}; margin:0">Taux d'inondation par hivernage, après nettoyage (2015-2024)</p>
  <div style="display:flex; align-items:flex-end; gap:24px; justify-content:center; background:{CARD_BG}; border:1px solid {BORDER}; border-radius:20px; padding:40px 32px 24px; margin-top:8px">
    {bars_row}
  </div>
  <p style="font-size:26px; color:{TEXT_MUTED}; margin:0">Zones humides (Niayes/bas-fonds) et urbain dense atteignent 27 % ; agricole et végétation naturelle restent sous 11 %.</p>
  {pagefoot(6)}
</section>''')

# ============================================================ 7. METHODOLOGIE
add("methodologie", f'''<section id="methodologie" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:40px">
  {eyebrow("Méthodologie")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">Une validation qui respecte le temps et l'espace</h2>
  <div style="display:flex; gap:32px">
    <div style="flex:1; background:{CARD_BG}; border:1px solid {BORDER}; border-radius:20px; padding:44px; display:flex; flex-direction:column; gap:20px">
      <h3 style="font-size:34px; font-weight:600; color:{ACCENT}; margin:0">Découpage temporel</h3>
      <div style="display:flex; align-items:center; gap:16px">
        <p style="font-size:28px; font-weight:600; background:#eaf1f8; color:{ACCENT}; padding:12px 20px; border-radius:10px; margin:0">2015 → 2023</p>
        <x-icon name="Chart" style="color:{TEXT_MUTED}; width:28px; height:28px"></x-icon>
        <p style="font-size:28px; font-weight:600; background:#fbeee8; color:{ACCENT2}; padding:12px 20px; border-radius:10px; margin:0">2024 = test</p>
      </div>
      <p style="font-size:26px; color:{TEXT_MUTED}; line-height:1.5; margin:0">Le modèle est testé sur une année entièrement future, jamais vue à l'entraînement.</p>
    </div>
    <div style="flex:1; background:{CARD_BG}; border:1px solid {BORDER}; border-radius:20px; padding:44px; display:flex; flex-direction:column; gap:20px">
      <h3 style="font-size:34px; font-weight:600; color:{ACCENT}; margin:0">GroupKFold par région</h3>
      <p style="font-size:26px; color:{TEXT_MUTED}; line-height:1.5; margin:0">5 blocs de validation croisée groupés par région : des zones voisines ne se retrouvent jamais à la fois en apprentissage et en validation.</p>
      <p style="font-size:26px; color:{TEXT_DARK}; line-height:1.5; margin:0"><b>Évite</b> l'autocorrélation spatiale qui gonflerait artificiellement la performance.</p>
    </div>
  </div>
  {pagefoot(7)}
</section>''')

# ============================================================ 8. MODELES / RESULTATS
def model_row(name, r05, seuil, rad, prec, f1, auc, bg, bold=False):
    w = "font-weight:700" if bold else ""
    return (f'<tr style="background:{bg}"><td style="padding:18px 24px; {w}">{name}</td>'
            f'<td style="padding:18px 24px; text-align:center">{r05}</td>'
            f'<td style="padding:18px 24px; text-align:center">{seuil}</td>'
            f'<td style="padding:18px 24px; text-align:center; color:{ACCENT}; font-weight:700">{rad}</td>'
            f'<td style="padding:18px 24px; text-align:center">{prec}</td>'
            f'<td style="padding:18px 24px; text-align:center">{f1}</td>'
            f'<td style="padding:18px 24px; text-align:center">{auc}</td></tr>')

add("modeles", f'''<section id="modeles" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:36px">
  {eyebrow("Modélisation")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">Trois modèles, un choix interprétable</h2>
  <table style="font-family:{FONT}; font-size:24px; color:{TEXT_DARK}; border-radius:16px; overflow:hidden">
    <tr style="background:{ACCENT_DARK}">
      <th style="width:24%; color:#fff; text-align:left; padding:16px 24px">Modèle</th>
      <th style="width:13%; color:#fff; padding:16px 24px">Rappel (0,5)</th>
      <th style="width:13%; color:#fff; padding:16px 24px">Seuil ajusté</th>
      <th style="width:14%; color:#fff; padding:16px 24px">Rappel ajusté</th>
      <th style="width:12%; color:#fff; padding:16px 24px">Précision</th>
      <th style="width:12%; color:#fff; padding:16px 24px">F1</th>
      <th style="width:12%; color:#fff; padding:16px 24px">AUC-ROC</th>
    </tr>
    {model_row("Régression logistique", "85,1 %", "0,40", "90,4 %", "40,5 %", "0,56", "0,858", CARD_BG, bold=True)}
    {model_row("Random Forest", "77,7 %", "0,40", "84,0 %", "44,4 %", "0,58", "0,856", "#f1f3f5")}
    {model_row("Gradient Boosting", "83,0 %", "0,30", "89,4 %", "37,2 %", "0,53", "0,845", CARD_BG)}
  </table>
  <p style="font-size:28px; color:{TEXT_MUTED}; margin:0"><b style="color:{TEXT_DARK}">Modèle retenu : régression logistique</b> — meilleure AUC, directement interprétable, rappel de 90 % sur 2024 (non vue à l'entraînement).</p>
  {pagefoot(8)}
</section>''')

# ============================================================ 9. FACTEURS
add("facteurs", f'''<section id="facteurs" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:36px">
  {eyebrow("Interprétation")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">Ce qui pèse sur le risque</h2>
  <div style="display:flex; gap:32px">
    <div style="flex:1; background:#fbeee8; border-radius:20px; padding:44px; display:flex; flex-direction:column; gap:18px">
      <h3 style="font-size:32px; font-weight:600; color:{ACCENT2}; margin:0">Facteurs aggravants</h3>
      <ul style="font-size:27px; line-height:1.6; color:{TEXT_DARK}; margin:0; padding-left:32px">
        <li>Historique d'inondations 2005-2014</li>
        <li>Pluie extrême (24h / 72h)</li>
        <li>Indice pluie × imperméabilisation</li>
        <li>Nappe phréatique proche de la surface</li>
        <li>Zone basse (relative à la localité)</li>
      </ul>
    </div>
    <div style="flex:1; background:#eaf5ea; border-radius:20px; padding:44px; display:flex; flex-direction:column; gap:18px">
      <h3 style="font-size:32px; font-weight:600; color:{GREEN}; margin:0">Facteurs protecteurs</h3>
      <ul style="font-size:27px; line-height:1.6; color:{TEXT_DARK}; margin:0; padding-left:32px">
        <li>Altitude élevée</li>
        <li>Assainissement complet</li>
        <li>Pente plus marquée</li>
        <li>Éloignement du plan d'eau</li>
      </ul>
    </div>
  </div>
  {pagefoot(9)}
</section>''')

# ============================================================ 10. CARTOGRAPHIE
legend_items = [("Faible","<25%",GREEN), ("Modéré","25-49%",YELLOW), ("Élevé","50-74%",ORANGE), ("Très élevé","≥75%",RED)]
legend_html = "".join([f'''<div style="display:flex; align-items:center; gap:14px">
  <div style="width:28px; height:28px; border-radius:50%; background:{c}"></div>
  <p style="font-size:26px; color:{TEXT_DARK}; margin:0">{lab} <span style="color:{TEXT_MUTED}">({rng})</span></p>
</div>''' for lab,rng,c in legend_items])

add("cartographie", f'''<section id="cartographie" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:36px">
  {eyebrow("Cartographie du risque")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">Scénario de pluie extrême : 120 mm/24h</h2>
  <div style="display:flex; gap:32px">
    <div style="flex:1; background:{CARD_BG}; border:1px solid {BORDER}; border-radius:20px; padding:44px; display:flex; flex-direction:column; gap:16px; align-items:center; justify-content:center">
      <div style="display:flex; align-items:baseline; gap:20px">
        <p style="font-size:56px; font-weight:700; color:{TEXT_MUTED}; margin:0">21,6 %</p>
        <p style="font-size:40px; color:{TEXT_MUTED}; margin:0">→</p>
        <p style="font-size:72px; font-weight:700; color:{ACCENT2}; margin:0">84,0 %</p>
      </div>
      <p style="font-size:25px; color:{TEXT_MUTED}; text-align:center; margin:0">Probabilité moyenne d'inondation : climat normal vs scénario extrême (99ᵉ centile historique)</p>
    </div>
    <div style="flex:1; background:{CARD_BG}; border:1px solid {BORDER}; border-radius:20px; padding:44px; display:flex; flex-direction:column; gap:20px; justify-content:center">
      <h3 style="font-size:30px; font-weight:600; margin:0">4 niveaux de risque</h3>
      {legend_html}
    </div>
  </div>
  <p style="font-size:26px; color:{TEXT_MUTED}; margin:0">Cartes interactives Folium fournies séparément : vue nationale + zoom banlieue de Dakar, avec info-bulles (localité, probabilité, facteur dominant).</p>
  {pagefoot(10)}
</section>''')

# ============================================================ 11. PRIORITES
top_localities = [
    ("Saint-Louis","Saint-Louis","98,8%"), ("Malika","Dakar","98,0%"), ("Matam","Matam","96,8%"),
    ("Podor","Saint-Louis","96,6%"), ("Keur Massar","Dakar","96,0%"), ("Yeumbeul","Dakar","95,4%"),
    ("Kaolack","Kaolack","94,6%"), ("Guédiawaye","Dakar","93,8%"),
]
rows = []
for i,(loc,reg,p) in enumerate(top_localities):
    bg = CARD_BG if i % 2 == 0 else "#f1f3f5"
    rows.append(f'<tr style="background:{bg}"><td style="padding:16px 24px">{i+1}</td><td style="padding:16px 24px">{loc}</td><td style="padding:16px 24px; color:{TEXT_MUTED}">{reg}</td><td style="padding:16px 24px; text-align:right; font-weight:700; color:{ACCENT2}">{p}</td></tr>')
rows_html = "\n".join(rows)

add("priorites", f'''<section id="priorites" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:32px">
  {eyebrow("Restitution")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">Localités prioritaires</h2>
  <p style="font-size:27px; color:{TEXT_MUTED}; margin:0">Classées par probabilité moyenne d'inondation sous le scénario extrême (top 8 sur 38 localités)</p>
  <table style="font-family:{FONT}; font-size:26px; color:{TEXT_DARK}; border-radius:16px; overflow:hidden">
    <tr style="background:{ACCENT_DARK}">
      <th style="width:8%; color:#fff; text-align:left; padding:16px 24px">#</th>
      <th style="width:36%; color:#fff; text-align:left; padding:16px 24px">Localité</th>
      <th style="width:36%; color:#fff; text-align:left; padding:16px 24px">Région</th>
      <th style="width:20%; color:#fff; text-align:right; padding:16px 24px">Probabilité</th>
    </tr>
    {rows_html}
  </table>
  {pagefoot(11)}
</section>''')

# ============================================================ 12. RECOMMANDATIONS
recos = [
    ("Curage &amp; bassins de rétention", "Localités en tête de classement, avant chaque hivernage"),
    ("Pompage renforcé", "Zones humides (Niayes/bas-fonds) et zones basses relatives"),
    ("Encadrement de la construction", "Zones basses, nappe peu profonde, fort historique"),
    ("Alerte précoce par SMS", "ANACIM + opérateurs, seuil &gt; 80-100 mm/24h"),
    ("Extension de l'assainissement", "Aucun → Partiel → Complet : le levier le plus direct"),
    ("Données réelles", "CHIRPS, SRTM, OpenStreetMap, Copernicus EMS, ANACIM"),
]
cards = []
for title, sub in recos:
    cards.append(f'''<div style="flex:1; min-width:440px; background:{CARD_BG}; border:1px solid {BORDER}; border-radius:18px; padding:32px; display:flex; flex-direction:column; gap:10px">
      <h3 style="font-size:28px; font-weight:600; color:{ACCENT}; margin:0">{title}</h3>
      <p style="font-size:23px; color:{TEXT_MUTED}; line-height:1.4; margin:0">{sub}</p>
    </div>''')
cards_html = "\n".join(cards)

add("recommandations", f'''<section id="recommandations" data-transition="fade" style="background:{BG_LIGHT}; color:{TEXT_DARK}; font-family:{FONT}; padding:128px 128px 160px; display:flex; flex-direction:column; gap:32px">
  {eyebrow("Recommandations")}
  <h2 style="font-size:64px; font-weight:700; margin:0; line-height:1.15">Six actions opérationnelles</h2>
  <div style="display:flex; flex-wrap:wrap; gap:24px">
    {cards_html}
  </div>
  {pagefoot(12)}
</section>''')

# ============================================================ 13. CLOTURE
add("cloture", f'''<section id="cloture" data-transition="fade" style="background:linear-gradient(160deg, {BG_DARK} 0%, #182c4d 100%); color:{TEXT_LIGHT}; font-family:{FONT}; padding:128px; display:flex; flex-direction:column; justify-content:space-between">
  <div style="display:flex; flex-direction:column; gap:24px; max-width:1500px">
    {eyebrow("Limites et perspectives", "#7fb2e8")}
    <p style="font-size:30px; line-height:1.6; color:{TEXT_LIGHT_MUTED}; margin:0">Données 100 % synthétiques · scénario extrême = extrapolation à lire comme un classement relatif · prochaine étape : rebrancher des données réelles (CHIRPS, SRTM, Copernicus EMS) et un tableau de bord d'alerte.</p>
  </div>
  <div>
    <h1 style="font-size:104px; font-weight:700; margin:0">Merci. Questions ?</h1>
  </div>
  <p style="font-size:26px; color:{TEXT_LIGHT_MUTED}">Notebook · cartes interactives · rapport · livrables joints</p>
</section>''')

for sid in order:
    with open(f"{ROOT}/slides/{sid}.html", "w", encoding="utf-8") as f:
        f.write(slides[sid])

deck = {
    "v": 4,
    "createdOnFiles": {"v": 1, "at": "2026-09-20T13:40:00Z"},
    "title": "Prédiction et cartographie des risques d'inondation au Sénégal",
    "order": order,
    "sections": {
        "intro": {"description": "Contexte, enjeu et objectif", "start": "cover"},
        "donnees_meth": {"description": "Données, nettoyage et méthodologie", "start": "donnees"},
        "resultats": {"description": "Modélisation, interprétation et cartographie du risque", "start": "modeles"},
        "restitution": {"description": "Localités prioritaires, recommandations, limites", "start": "priorites"},
    },
    "faces": {
        "ibm-plex-sans": {"family": "IBM Plex Sans", "href": FONT_LINK}
    }
}
with open(f"{ROOT}/deck.json", "w", encoding="utf-8") as f:
    json.dump(deck, f, ensure_ascii=False, indent=2)

print("Slides generated:", order)
