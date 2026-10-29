"""Train and evaluate a first credit-risk model on the synthetic SME data."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, TARGET, add_features

df = add_features(pd.read_csv("data/smes.csv"))
X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

prep = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])

models = {
    "logistic_regression": Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
    "gradient_boosting": Pipeline([
        ("prep", prep),
        ("clf", HistGradientBoostingClassifier(random_state=42)),
    ]),
}

print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"Default rate (test): {y_test.mean():.1%}\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"== {name} ==")
    print(f"AUC:   {roc_auc_score(y_test, proba):.3f}")
    print(f"Brier: {brier_score_loss(y_test, proba):.3f}  (lower is better)")
    print(classification_report(y_test, proba > 0.5, digits=3, zero_division=0))

# Keep the logistic regression: simpler and explainable
final = models["logistic_regression"]
joblib.dump(final, "models/credit_model.joblib")

names = final.named_steps["prep"].get_feature_names_out()
coefs = final.named_steps["clf"].coef_[0]
ranked = sorted(zip(names, coefs), key=lambda t: abs(t[1]), reverse=True)
print("Top drivers (logistic regression, standardised):")
for n, c in ranked[:8]:
    arrow = "raises risk" if c > 0 else "lowers risk"
    print(f"  {n:28s} {c:+.3f}  {arrow}")
print("\nSaved models/credit_model.joblib")
