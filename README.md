# Prédiction et cartographie des risques d'inondation au Sénégal

Projet de machine learning (classification binaire) prédisant la probabilité qu'une zone du Sénégal soit inondée pendant l'hivernage, à partir de ses caractéristiques physiques, urbaines et pluviométriques, puis cartographiant le risque sous un scénario de pluie extrême.

## Contenu du dépôt

- **`notebook_inondation_senegal.ipynb`** — notebook Jupyter commenté et exécuté : jointure, exploration, nettoyage, feature engineering, modélisation (régression logistique / Random Forest / Gradient Boosting), validation temporelle + spatiale (GroupKFold par région), interprétation, scénario de pluie extrême et cartographie.
- **`notebook_pret_a_l_emploi.zip`** — le notebook accompagné des données, prêt à être importé tel quel dans Google Colab.
- **`rapport_inondation_senegal.docx`** — rapport complet (contexte, méthodologie, résultats, facteurs déterminants, cartographie, localités prioritaires, recommandations, limites).
- **`scripts/`** — scripts Python source, étape par étape (exploration, nettoyage, feature engineering, modélisation, interprétation, cartographie des risques), plus les scripts de génération du notebook et de la présentation.
- **`data/`** — données brutes (`zones.csv`, `observations.csv`) et données nettoyées (`*_clean.csv`, `dataset_model.csv`).
- **`figures/`** — graphiques générés (exploration, courbes ROC, matrice de confusion, importance des variables).
- **`outputs/`** — résultats de modélisation (métriques, seuils, modèle entraîné `best_model.joblib`) et cartes interactives Folium (`.html`) : carte exploratoire, carte de risque nationale (scénario extrême), zoom banlieue de Dakar.

## Résultats clés

- Modèle retenu : régression logistique (`class_weight="balanced"`), seuil ajusté = 0,40.
- Test temporel 2024 (année non vue à l'entraînement) : rappel = 90,4 %, précision = 40,5 %, AUC-ROC = 0,858.
- Facteur le plus stable : historique d'inondations 2005-2014.
- Scénario de pluie extrême (120 mm/24h) : probabilité moyenne d'inondation 21,6 % → 84,0 % ; localités prioritaires en tête : Saint-Louis, Malika, Matam, Podor, Keur Massar, Yeumbeul, Kaolack, Guédiawaye.

## Limites

Données 100 % synthétiques, à visée pédagogique. Le scénario de pluie extrême est une extrapolation à lire comme un classement relatif plutôt qu'une fréquence exacte. Toute utilisation opérationnelle nécessite une revalidation avec des données réelles (CHIRPS, SRTM, OpenStreetMap, Copernicus EMS, ANACIM).
