"""Train BINARY classification models on ml_dataset.csv
Target: impact_high (0 = not high impact, 1 = high impact)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# ---- Setup paths ----
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_BINARY = BASE_DIR / "results" / "03_binary_classification"

# Create results folder if it doesn't exist
RESULTS_BINARY.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}")
print(f"RESULTS_BINARY: {RESULTS_BINARY}\n")

# Global list to collect all plots
all_figures = []

# =====================================================================
# 1. DATA LOADING & PREPARATION
# =====================================================================

def load_data():
    """Load ml_dataset.csv"""
    df = pd.read_csv(DATA_PROCESSED / "ml_dataset.csv", sep=";")
    print(f"✅ Loaded ml_dataset.csv: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}\n")
    return df


def prepare_features_binary(df: pd.DataFrame):
    """
    Prepare features X and binary target y = impact_high.
    Use the SAME feature set as the multiclass models / GTA6 scenarios:
      - is_rockstar
      - sentiment_negative
      - market_return
      - franchise_gta
      - vix_level
      - vix_30d_mean
      - event_type_marketing
      - event_type_negative
      - event_type_release
      - vix_regime_medium
      - vix_regime_high
    """
    if "impact_high" not in df.columns:
        raise KeyError(
            "Column 'impact_high' not found in ml_dataset.csv. "
            "Check build_ml_dataset.py."
        )

    target = "impact_high"

    feature_cols = [
        "is_rockstar",
        "sentiment_negative",
        "market_return",
        "franchise_gta",
        "vix_level",
        "vix_30d_mean",
        "event_type_marketing",
        "event_type_negative",
        "event_type_release",
        "vix_regime_medium",
        "vix_regime_high",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns in ml_dataset: {missing}")

    X = df[feature_cols].copy()
    y = df[target].astype(int)

    print(f"Feature matrix X shape (before NaN drop): {X.shape}")
    print(f"Target y shape (before NaN drop): {y.shape}")

    # still drop rows with NaNs (you already saw 152 → 133)
    mask = X.isna().any(axis=1) | y.isna()
    X = X[~mask].reset_index(drop=True)
    y = y[~mask].reset_index(drop=True)

    print(f"Feature matrix X shape (after NaN drop): {X.shape}")
    print(f"Target y shape (after NaN drop): {y.shape}")
    print("\nBinary target distribution (impact_high):")
    print(y.value_counts().rename({0: "Not high", 1: "High"}), "\n")

    return X, y

def split_data(X, y):
    """Split data into train/test with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # ✅ Reset indices after split to avoid misalignment
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"Train size: {X_train.shape}")
    print(f"Test size:  {X_test.shape}\n")

    return X_train, X_test, y_train, y_test


# =====================================================================
# 2. MODEL EVALUATION HELPERS
# =====================================================================

def evaluate_model(model, X_test, y_test, name="Model"):
    """Evaluate model on the test set and save confusion matrix as PNG."""
    
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*70}")
    print(f"=== {name} (BINARY: impact_high) ===")
    print(f"{'='*70}")
    print(f"Accuracy:        {acc:.3f}")
    print(f"F1-score (macro): {f1:.3f}")
    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Not high", "High"],
            digits=3,
            zero_division=0
        )
    )

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not high", "High"],
        yticklabels=["Not high", "High"],
        ax=ax
    )
    ax.set_title(f"{name} – Confusion Matrix (Binary)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    # ✅ SAVE PNG TO RESULTS
    png_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    png_path = RESULTS_BINARY / f"{png_name}_confusion_matrix.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"   📊 Saved PNG: {png_path}\n")

    return acc, f1


def check_overfitting(model, X_train, X_test, y_train, y_test, name="Model"):
    """Check for overfitting using train vs test accuracy."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    overfit_gap = train_acc - test_acc

    print(f"\n{'='*70}")
    print(f"🔍 OVERFITTING CHECK (BINARY): {name}")
    print(f"{'='*70}")
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test  Accuracy: {test_acc:.3f}")
    print(f"Gap (train - test): {overfit_gap:.3f}")

    if overfit_gap > 0.10:
        print("⚠️  WARNING: Possible overfitting (gap > 0.10)")
    else:
        print("✅ Good generalization (gap ≤ 0.10)")

    return train_acc, test_acc, overfit_gap


def cross_validate_model(model, X, y, name="Model"):
    """5-fold stratified cross-validation on FULL dataset (binary)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")

    print(f"\n📊 {name} – 5-fold CV (BINARY):")
    print(f"  Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"  F1-macro: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    return acc_scores.mean(), f1_scores.mean()


# =====================================================================
# 3. INDIVIDUAL MODEL TRAINING
# =====================================================================

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression with imputation for NaN values"""
    
    print("🤖 Training Logistic Regression (binary)...")
    
    # ✅ CREATE PIPELINE WITH IMPUTATION
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    acc, f1 = evaluate_model(pipeline, X_test, y_test, name="Logistic Regression (Binary)")
    
    return pipeline, acc, f1


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with imputation for NaN values"""
    print("\n🤖 Training Random Forest (binary)...")
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', RandomForestClassifier(
            n_estimators=300,
            max_depth=6,          # 🔧 limit depth
            min_samples_leaf=2,   # 🔧 regularization
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    acc, f1 = evaluate_model(pipeline, X_test, y_test, name="Random Forest (Binary)")
    
    return pipeline, acc, f1


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting with imputation for NaN values"""
    print("\n🤖 Training Gradient Boosting (binary)...")
    
    # ✅ CREATE PIPELINE WITH IMPUTATION
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    acc, f1 = evaluate_model(pipeline, X_test, y_test, name="Gradient Boosting (Binary)")
    
    return pipeline, acc, f1


def train_neural_network(X_train, X_test, y_train, y_test):
    """Train Neural Network with imputation for NaN values"""
    print("\n🤖 Training Neural Network (MLP, binary)...")
    
    # ✅ CREATE PIPELINE WITH IMPUTATION
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            verbose=0
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    acc, f1 = evaluate_model(pipeline, X_test, y_test, name="Neural Network (MLP, Binary)")
    
    return pipeline, acc, f1


def compare_models(models, X_test, y_test):
    """Compare models on test set and save comparison PNG."""
    
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        results.append({"Model": name, "Accuracy": acc, "F1_macro": f1})

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON – BINARY (impact_high)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width / 2, results_df["Accuracy"], width, label="Accuracy", alpha=0.8)
    ax.bar(x + width / 2, results_df["F1_macro"], width, label="F1-macro", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Binary Model Comparison: Accuracy vs F1-macro")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # ✅ SAVE PNG TO RESULTS
    png_path = RESULTS_BINARY / "binary_model_comparison.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"   📊 Saved PNG: {png_path}\n")

    return results_df


# =====================================================================
# 4. MAIN
# =====================================================================

def main():
    # 1) Load & prepare
    df = load_data()
    X, y = prepare_features_binary(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2) Train models
    log_reg, lr_acc, lr_f1 = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf, rf_acc, rf_f1 = train_random_forest(X_train, X_test, y_train, y_test)
    gb, gb_acc, gb_f1 = train_gradient_boosting(X_train, X_test, y_train, y_test)
    mlp, mlp_acc, mlp_f1 = train_neural_network(X_train, X_test, y_train, y_test)

    # 3) Compare models on test set
    models = {
        "Logistic Regression (Binary)": log_reg,
        "Random Forest (Binary)": rf,
        "Gradient Boosting (Binary)": gb,
        "Neural Network (MLP, Binary)": mlp,
    }

    results_test = compare_models(models, X_test, y_test)

    # 4) Overfitting checks
    for name, model in models.items():
        check_overfitting(model, X_train, X_test, y_train, y_test, name=name)

    # 5) 5-fold CV on full dataset
    print("\n" + "=" * 70)
    print("🔁 5-FOLD STRATIFIED CROSS-VALIDATION – BINARY")
    print("=" * 70)

    cv_results = []
    for name, model in models.items():
        cv_acc, cv_f1 = cross_validate_model(model, X, y, name=name)
        cv_results.append(
            {
                "Model": name,
                "CV_Accuracy_mean": cv_acc,
                "CV_F1_macro_mean": cv_f1,
            }
        )

    cv_df = pd.DataFrame(cv_results)

    print("\n📄 Binary cross-validation summary:")
    print(cv_df.to_string(index=False))

    # 6) ✅ SAVE CSVs TO RESULTS FOLDER
    
    # Save test results
    test_csv = RESULTS_BINARY / "binary_test_results.csv"
    results_test.to_csv(test_csv, index=False)
    print(f"✅ Test results saved to: {test_csv}")
    
    # Save CV results
    cv_csv = RESULTS_BINARY / "binary_cv_results.csv"
    cv_df.to_csv(cv_csv, index=False)
    print(f"✅ CV results saved to: {cv_csv}")

    print("\n" + "="*70)
    print("✅ BINARY CLASSIFICATION COMPLETE")
    print("="*70)
    print(f"📁 All results saved to: {RESULTS_BINARY}")
    print("="*70 + "\n")

    return {
        "models": models,
        "results_test": results_test,
        "results_cv": cv_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    results = main()

