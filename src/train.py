"""
Train two models:
1. XGBoost regressor — predict total cases in next 4 weeks
2. Random Forest classifier — predict outbreak risk (low/medium/high)

Models saved to data/model_reg.pkl and data/model_clf.pkl
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, r2_score,
                             classification_report, accuracy_score)
from xgboost import XGBRegressor

from features import build_features, FEATURE_COLS, TARGET_REG, TARGET_CLF


def load_data():
    df = pd.read_csv("data/measles_weekly.csv", parse_dates=["date"])
    return build_features(df)


def train_regressor(df):
    clean = df.dropna(subset=FEATURE_COLS + [TARGET_REG])
    X = clean[FEATURE_COLS]
    y = clean[TARGET_REG]

    # time series split — never train on future data
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores, r2_scores = [], []

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        preds = model.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, preds))
        r2_scores.append(r2_score(y_val, preds))

    print(f"Regressor CV — MAE: {np.mean(mae_scores):.1f} cases | R²: {np.mean(r2_scores):.3f}")

    # final fit on all data
    model.fit(X, y)
    return model


def train_classifier(df):
    clean = df.dropna(subset=FEATURE_COLS + [TARGET_CLF])
    X = clean[FEATURE_COLS]
    y = clean[TARGET_CLF].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    acc_scores = []

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc_scores.append(accuracy_score(y_val, preds))

    print(f"Classifier CV — Accuracy: {np.mean(acc_scores):.3f}")

    model.fit(X, y)

    # feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("\nTop 5 features (classifier):")
    print(importances.sort_values(ascending=False).head())

    return model


def save_models(reg_model, clf_model):
    os.makedirs("data", exist_ok=True)
    with open("data/model_reg.pkl", "wb") as f:
        pickle.dump(reg_model, f)
    with open("data/model_clf.pkl", "wb") as f:
        pickle.dump(clf_model, f)
    print("\nModels saved to data/model_reg.pkl and data/model_clf.pkl")


def main():
    print("Loading and engineering features...")
    df = load_data()
    print(f"Dataset: {len(df)} weeks, {df['year'].min()}–{df['year'].max()}\n")

    print("Training XGBoost regressor...")
    reg_model = train_regressor(df)

    print("\nTraining Random Forest classifier...")
    clf_model = train_classifier(df)

    save_models(reg_model, clf_model)


if __name__ == "__main__":
    main()
