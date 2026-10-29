"""Show how different risk cut-offs trade off caught defaulters vs. false alarms."""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from features import CATEGORICAL, NUMERIC, TARGET, add_features

df = add_features(pd.read_csv("data/smes.csv"))
X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = joblib.load("models/credit_model.joblib")
proba = model.predict_proba(X_test)[:, 1]

print(f"{'cut-off':>8} {'flagged':>8} {'defaulters caught':>18} {'flags that were real':>21}")
for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    flagged = proba >= t
    caught = (flagged & (y_test == 1)).sum() / (y_test == 1).sum()
    precision = (flagged & (y_test == 1)).sum() / max(flagged.sum(), 1)
    print(f"{t:>8.2f} {flagged.mean():>8.0%} {caught:>18.0%} {precision:>21.0%}")
