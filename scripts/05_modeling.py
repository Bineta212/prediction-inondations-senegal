"""
Etape 7-9 : Decoupage, modelisation, evaluation
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, precision_recall_curve, classification_report)

RNG = 42
df = pd.read_csv("data/dataset_model.csv")

# --- Decoupage temporel : 2024 = test ---
train_df = df[df.annee < 2024].copy()
test_df = df[df.annee == 2024].copy()
print("Train:", train_df.shape, "Test (2024):", test_df.shape)
print("Taux inondation train:", train_df.inondation.mean().round(3), " test:", test_df.inondation.mean().round(3))

id_cols = ["zone_id", "annee", "localite", "region", "latitude", "longitude", "inondation"]
feature_cols = [c for c in df.columns if c not in id_cols]
print("\nNb features:", len(feature_cols))

X_train, y_train = train_df[feature_cols], train_df["inondation"]
X_test, y_test = test_df[feature_cols], test_df["inondation"]
groups_train = train_df["region"]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Validation croisee groupee par region (verification robustesse, pas de fuite spatiale) ---
gkf = GroupKFold(n_splits=5)
models_cfg = {
    "Regression logistique": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG),
    "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                             class_weight="balanced", random_state=RNG, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                     random_state=RNG),
}

cv_results = {}
for name, model in models_cfg.items():
    recalls, aucs = [], []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_s, y_train, groups=groups_train)):
        m = model.__class__(**model.get_params())
        # Gradient boosting has no class_weight -> use sample_weight
        if name == "Gradient Boosting":
            sw = np.where(y_train.iloc[tr_idx] == 1, (y_train.iloc[tr_idx] == 0).sum() / (y_train.iloc[tr_idx] == 1).sum(), 1.0)
            m.fit(X_train_s[tr_idx], y_train.iloc[tr_idx], sample_weight=sw)
        else:
            m.fit(X_train_s[tr_idx], y_train.iloc[tr_idx])
        proba = m.predict_proba(X_train_s[val_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        recalls.append(recall_score(y_train.iloc[val_idx], pred))
        aucs.append(roc_auc_score(y_train.iloc[val_idx], proba))
    cv_results[name] = {"recall_mean": np.mean(recalls), "recall_std": np.std(recalls),
                         "auc_mean": np.mean(aucs), "auc_std": np.std(aucs)}
    print(f"{name:25s} CV(GroupKFold/region) recall={np.mean(recalls):.3f}±{np.std(recalls):.3f}  "
          f"AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")

# --- Entrainement final sur tout le train, evaluation sur test 2024 ---
final_models = {}
proba_test = {}
for name, model in models_cfg.items():
    if name == "Gradient Boosting":
        sw = np.where(y_train == 1, (y_train == 0).sum() / (y_train == 1).sum(), 1.0)
        model.fit(X_train_s, y_train, sample_weight=sw)
    else:
        model.fit(X_train_s, y_train)
    final_models[name] = model
    proba_test[name] = model.predict_proba(X_test_s)[:, 1]

# --- Seuil ajuste pour rappel >= 80% (sur le TRAIN, via validation, puis applique au test) ---
def best_threshold_for_recall(y_true, proba, target_recall=0.80):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_prec = 0.5, -1
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rec = recall_score(y_true, pred)
        if rec >= target_recall:
            prec = precision_score(y_true, pred, zero_division=0)
            if prec > best_prec:
                best_prec, best_t = prec, t
    return best_t

# seuil calcule par validation croisee (out-of-fold) pour eviter le surapprentissage du seuil
oof_proba = {name: np.zeros(len(X_train_s)) for name in models_cfg}
for name, model in models_cfg.items():
    for tr_idx, val_idx in gkf.split(X_train_s, y_train, groups=groups_train):
        m = model.__class__(**model.get_params())
        if name == "Gradient Boosting":
            sw = np.where(y_train.iloc[tr_idx] == 1, (y_train.iloc[tr_idx] == 0).sum() / (y_train.iloc[tr_idx] == 1).sum(), 1.0)
            m.fit(X_train_s[tr_idx], y_train.iloc[tr_idx], sample_weight=sw)
        else:
            m.fit(X_train_s[tr_idx], y_train.iloc[tr_idx])
        oof_proba[name][val_idx] = m.predict_proba(X_train_s[val_idx])[:, 1]

thresholds = {}
results = []
for name in models_cfg:
    t = best_threshold_for_recall(y_train, oof_proba[name], target_recall=0.80)
    thresholds[name] = t
    pred_default = (proba_test[name] >= 0.5).astype(int)
    pred_adj = (proba_test[name] >= t).astype(int)

    for label, pred in [("seuil 0.5", pred_default), (f"seuil ajusté ({t:.2f})", pred_adj)]:
        cm = confusion_matrix(y_test, pred)
        results.append({
            "modele": name, "seuil": label,
            "rappel": recall_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba_test[name]),
            "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1],
        })

res_df = pd.DataFrame(results)
pd.set_option("display.width", 160)
print("\n=== RESULTATS SUR LE TEST 2024 (validation temporelle) ===")
print(res_df.to_string(index=False))
res_df.to_csv("outputs/resultats_test_2024.csv", index=False)

with open("outputs/thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

# --- ROC curves ---
fig, ax = plt.subplots(figsize=(6, 6))
colors = {"Regression logistique": "#8a94a3", "Random Forest": "#2563a8", "Gradient Boosting": "#c85a3a"}
for name in models_cfg:
    fpr, tpr, _ = roc_curve(y_test, proba_test[name])
    auc = roc_auc_score(y_test, proba_test[name])
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", color=colors[name], lw=2)
ax.plot([0, 1], [0, 1], "--", color="lightgray")
ax.set_xlabel("Taux de faux positifs")
ax.set_ylabel("Taux de vrais positifs (rappel)")
ax.set_title("Courbes ROC — validation temporelle (test = 2024)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/06_roc_curves.png", dpi=150)
plt.close()

# --- Choix du modele retenu : meilleur AUC + meilleur rappel au seuil ajuste ---
best_model_name = res_df[res_df.seuil.str.contains("ajust")].sort_values("roc_auc", ascending=False).iloc[0]["modele"]
print(f"\nModele retenu : {best_model_name}  (seuil = {thresholds[best_model_name]:.2f})")

import joblib
joblib.dump({"model": final_models[best_model_name], "scaler": scaler, "feature_cols": feature_cols,
             "threshold": thresholds[best_model_name], "model_name": best_model_name},
            "outputs/best_model.joblib")

with open("outputs/best_model_name.txt", "w") as f:
    f.write(best_model_name)
