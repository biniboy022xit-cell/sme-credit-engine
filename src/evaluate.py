"""Proper evaluation: cross-validation + probability calibration check."""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, TARGET, add_features

df = add_features(pd.read_csv("data/smes.csv"))
X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]

prep = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
base = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])

# 1. Cross-validation: how stable is AUC across different data splits?
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(base, X, y, cv=cv, scoring="roc_auc")
print("5-fold cross-validated AUC:")
print(f"  scores: {np.round(scores, 3)}")
print(f"  mean: {scores.mean():.3f}  std: {scores.std():.3f}")

# 2. Calibration: do predicted probabilities match observed frequencies?
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
uncalibrated = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
uncalibrated.fit(X_train, y_train)
p_uncal = uncalibrated.predict_proba(X_test)[:, 1]

calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)
p_cal = calibrated.predict_proba(X_test)[:, 1]

print("\nBrier score (lower is better, perfect = 0):")
print(f"  uncalibrated: {brier_score_loss(y_test, p_uncal):.4f}")
print(f"  calibrated:   {brier_score_loss(y_test, p_cal):.4f}")
print(f"  AUC unchanged by calibration: {roc_auc_score(y_test, p_uncal):.3f} -> "
      f"{roc_auc_score(y_test, p_cal):.3f}")

# Save a calibration plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
CalibrationDisplay.from_predictions(y_test, p_uncal, n_bins=8, ax=ax, name="Uncalibrated")
CalibrationDisplay.from_predictions(y_test, p_cal, n_bins=8, ax=ax, name="Calibrated")
ax.set_title("Reliability diagram: predicted vs. actual default rate")
fig.tight_layout()
fig.savefig("docs/calibration.png", dpi=120)
print("\nSaved docs/calibration.png")
