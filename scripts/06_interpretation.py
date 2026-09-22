"""
Etape 10 : Interpretation (importance des variables, matrice de confusion visuelle)
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

bundle = joblib.load("outputs/best_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]
threshold = bundle["threshold"]
model_name = bundle["model_name"]

df = pd.read_csv("data/dataset_model.csv")
test_df = df[df.annee == 2024].copy()
X_test = scaler.transform(test_df[feature_cols])
y_test = test_df["inondation"]
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)

# --- Matrice de confusion (modele retenu, seuil ajuste) ---
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pas d'inondation", "Inondation"])
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title(f"Matrice de confusion — {model_name}\n(test 2024, seuil={threshold:.2f})")
plt.tight_layout()
plt.savefig("figures/07_confusion_matrix.png", dpi=150)
plt.close()

# --- Importance des variables ---
# Pour la regression logistique : coefficients standardises = importance
if hasattr(model, "coef_"):
    importance = pd.Series(model.coef_[0], index=feature_cols).sort_values()
    title = f"Coefficients standardisés — {model_name}"
    xlabel = "Coefficient (effet sur le log-odds d'inondation)"
else:
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    title = f"Importance des variables — {model_name}"
    xlabel = "Importance"

top = pd.concat([importance.head(8), importance.tail(8)]).drop_duplicates()
top = top.sort_values()
colors = ["#c85a3a" if v < 0 else "#2563a8" for v in top.values]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top.index, top.values, color=colors)
ax.set_xlabel(xlabel)
ax.set_title(title)
ax.spines[["top", "right"]].set_visible(False)
ax.axvline(0, color="black", lw=0.8)
plt.tight_layout()
plt.savefig("figures/08_importance_variables.png", dpi=150)
plt.close()

print("=== Top facteurs (modele retenu) ===")
print(importance.sort_values(ascending=False).head(10))
print("\n=== Facteurs protecteurs ===")
print(importance.sort_values().head(10))

# --- Aussi entrainer un RF pour feature_importances_ (complementaire, plus robuste/non-lineaire) ---
rf = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                             class_weight="balanced", random_state=42, n_jobs=-1)
train_df = df[df.annee < 2024]
X_train = scaler.fit_transform(train_df[feature_cols])
rf.fit(X_train, train_df["inondation"])
rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Importance Random Forest (top 10) ===")
print(rf_imp.head(10))

fig, ax = plt.subplots(figsize=(8, 6))
top_rf = rf_imp.head(12).sort_values()
ax.barh(top_rf.index, top_rf.values, color="#2563a8")
ax.set_xlabel("Importance (Random Forest)")
ax.set_title("Importance des variables — Random Forest (validation croisée)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/09_importance_rf.png", dpi=150)
plt.close()

importance.to_csv("outputs/importance_modele_retenu.csv")
rf_imp.to_csv("outputs/importance_rf.csv")
