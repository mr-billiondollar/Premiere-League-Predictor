"""
train_model.py
Trains a Random Forest and an XGBoost classifier to predict match outcome
(H/D/A) from pre-match rolling team form, compares them against a naive
baseline, and saves the trained models for later use on upcoming fixtures.

IMPORTANT: the train/test split is CHRONOLOGICAL, not random. Shuffling
would let the model train on data from "the future" relative to some test
matches, which would quietly inflate accuracy and not reflect how you'd
actually use this model (predicting matches that haven't happened yet).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "model_ready.csv"
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLS = [
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_avg_shots", "home_avg_shots_on_target", "home_form_points_last5",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_avg_shots", "away_avg_shots_on_target", "away_form_points_last5",
]
TARGET_COL = "FTR"
TEST_SEASONS = ["2024-25", "2025-26"]


def chronological_split(df: pd.DataFrame):
    train = df[~df["Season"].isin(TEST_SEASONS)].copy()
    test = df[df["Season"].isin(TEST_SEASONS)].copy()
    print(f"Train: {len(train)} fixtures ({train['Season'].min()} to {train['Season'].max()})")
    print(f"Test:  {len(test)} fixtures ({', '.join(TEST_SEASONS)}) -- model has never seen these")
    return train, test


def baseline_accuracy(y_train, y_test):
    """What accuracy would we get by always guessing the majority class?"""
    majority_class = y_train.mode()[0]
    preds = [majority_class] * len(y_test)
    return accuracy_score(y_test, preds), majority_class


def evaluate(name, y_test, preds, label_names):
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=label_names, digits=3)
    cm = confusion_matrix(y_test, preds)
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Accuracy: {acc:.3f}")
    print(f"\nClassification report:\n{report}")
    print(f"Confusion matrix (rows=actual, cols=predicted) {list(label_names)}:")
    print(cm)
    return acc


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    train_df, test_df = chronological_split(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    # Baseline: what if we just always predicted the most common result?
    base_acc, majority_class = baseline_accuracy(y_train, y_test)
    print(f"\nBaseline (always predict '{majority_class}'): {base_acc:.3f} accuracy")
    print("Any model we ship needs to clear this bar to be worth anything.")

    label_encoder = LabelEncoder().fit(y_train)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    sample_weights = compute_sample_weight("balanced", y_train_enc)
    results = {}
    MODELS_DIR.mkdir(exist_ok=True)

    configs = {
        # name -> (use_class_balancing, description)
        "random_forest_balanced": ("rf", True),
        "random_forest_unweighted": ("rf", False),
        "xgboost_balanced": ("xgb", True),
        "xgboost_unweighted": ("xgb", False),
    }

    for name, (model_type, balanced) in configs.items():
        if model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=10,
                class_weight="balanced" if balanced else None,
                random_state=42, n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", num_class=3,
                random_state=42, eval_metric="mlogloss",
            )
            fit_kwargs = {"sample_weight": sample_weights} if balanced else {}
            model.fit(X_train, y_train_enc, **fit_kwargs)
            preds = label_encoder.inverse_transform(model.predict(X_test))

        acc = evaluate(name, y_test, preds, label_encoder.classes_)
        draws_predicted = int((preds == "D").sum())
        results[name] = {"model": model, "accuracy": acc, "draws_predicted": draws_predicted}
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.pkl")

    # --- Summary: the accuracy vs draw-coverage tradeoff ---
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'Model':<28}{'Accuracy':<12}{'Draws predicted (of ' + str((y_test=='D').sum()) + ' actual)'}")
    print(f"{'Baseline (majority class)':<28}{base_acc:<12.3f}0")
    for name, r in results.items():
        print(f"{name:<28}{r['accuracy']:<12.3f}{r['draws_predicted']}")
    print(f"\nSaved all 4 model variants + label encoder + feature list to {MODELS_DIR}/")
    print("Recommended default for Phase 5: the 'unweighted' variants (highest raw accuracy).")
    print("Keep the 'balanced' variants around too -- worth mentioning both in your writeup.")

    # --- Feature importance (for the writeup / interview talking points) ---
    rf_importance = pd.Series(
        results["random_forest_unweighted"]["model"].feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    xgb_importance = pd.Series(
        results["xgboost_unweighted"]["model"].feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nRandom Forest top features:")
    print(rf_importance.round(3))
    print("\nXGBoost top features:")
    print(xgb_importance.round(3))

    return {"baseline_acc": base_acc, "results": results,
            "rf_importance": rf_importance, "xgb_importance": xgb_importance}


if __name__ == "__main__":
    main()
