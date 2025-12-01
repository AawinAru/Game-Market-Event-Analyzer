#train_binary.py
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
    We drop the multiclass labels to avoid accidental leakage.
    """
    if "impact_high" not in df.columns:
        raise KeyError(
            "Column 'impact_high' not found in ml_dataset.csv. "
            "Check build_ml_dataset.py."
        )

    target = "impact_high"

    # Drop all label-related columns from features
    drop_cols = [c for c in ["impact_label", "impact_label_num", "impact_high"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target].astype(int)

    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target y shape: {y.shape}")
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
    print("\n🤖 Training Logistic Regression (binary)...")
    log_reg = LogisticRegression(max_iter=5000)
    log_reg.fit(X_train, y_train)
    acc, f1 = evaluate_model(log_reg, X_test, y_test, name="Logistic Regression (Binary)")
    return log_reg, acc, f1


def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n🤖 Training Random Forest (binary)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    acc, f1 = evaluate_model(rf, X_test, y_test, name="Random Forest (Binary)")
    return rf, acc, f1


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    print("\n🤖 Training Gradient Boosting (binary)...")
    gb = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    acc, f1 = evaluate_model(gb, X_test, y_test, name="Gradient Boosting (Binary)")
    return gb, acc, f1


def train_neural_network(X_train, X_test, y_train, y_test):
    print("\n🤖 Training Neural Network (MLP, binary)...")
    # ✅ Increased max_iter + early stopping
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=2000,  # ✅ Changed from 500
        random_state=42,
        early_stopping=True,  # ✅ New: stop if validation doesn't improve
        validation_fraction=0.1,  # ✅ New: use 10% for validation
        n_iter_no_change=50,  # ✅ New: stop after 50 iters with no improvement
        verbose=0  # Set to 1 to see training progress
    )
    mlp.fit(X_train, y_train)
    acc, f1 = evaluate_model(mlp, X_test, y_test, name="Neural Network (MLP, Binary)")
    return mlp, acc, f1


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
        if "Logistic" in name:
            m = LogisticRegression(max_iter=5000)
        elif "Random Forest" in name:
            m = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
        elif "Gradient Boosting" in name:
            m = GradientBoostingClassifier(
                learning_rate=0.05,
                n_estimators=300,
                random_state=42,
            )
        else:  # MLP
            m = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=2000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                verbose=0
            )

        cv_acc, cv_f1 = cross_validate_model(m, X, y, name=name)
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

    # 6) ✅ SAVE CSVs TO RESULTS FOLDER (PNGs already saved above)
    
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

#train_classification.py
"""Train classification models on ml_dataset.csv (Multi-class: Low/Medium/High)"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Setup paths ----
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_MULTICLASS = BASE_DIR / "results" / "04_multiclass_classification"

# Create results folder if it doesn't exist
RESULTS_MULTICLASS.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}")
print(f"RESULTS_MULTICLASS: {RESULTS_MULTICLASS}\n")


def load_data():
    """Load ml_dataset.csv"""
    df = pd.read_csv(DATA_PROCESSED / "ml_dataset.csv", sep=";")
    print(f"✅ Loaded ml_dataset.csv: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}\n")
    return df


def prepare_features(df):
    """Prepare features and target"""
    target = "impact_label_num"
    
    X = df.drop(columns=["impact_label", "impact_label_num"])
    y = df[target]
    
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target y shape: {y.shape}\n")
    
    return X, y


def split_data(X, y):
    """Split data into train/test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
    
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}\n")
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, name="Model"):
    """Evaluate model and display results"""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print(f"=== {name} ===")
    print(f"{'='*70}")
    print(f"Accuracy: {round(acc, 3)}")
    print(f"F1-score (macro): {round(f1, 3)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low", "Med", "High"],
        yticklabels=["Low", "Med", "High"]
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    # ✅ SAVE PNG TO RESULTS
    png_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    png_path = RESULTS_MULTICLASS / f"{png_name}_confusion_matrix.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Plot saved to: {png_path}")
    
    plt.close()
    
    return acc, f1


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression"""
    print("\n🤖 Training Logistic Regression...")
    # ✅ Removed multi_class="multinomial"
    log_reg = LogisticRegression(max_iter=5000)
    log_reg.fit(X_train, y_train)
    acc, f1 = evaluate_model(log_reg, X_test, y_test, name="Logistic Regression")
    return log_reg, acc, f1


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest"""
    print("\n🤖 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    acc, f1 = evaluate_model(rf, X_test, y_test, name="Random Forest")
    return rf, acc, f1


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting"""
    print("\n🤖 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=300,
        random_state=42
    )
    gb.fit(X_train, y_train)
    acc, f1 = evaluate_model(gb, X_test, y_test, name="Gradient Boosting")
    return gb, acc, f1


def train_neural_network(X_train, X_test, y_train, y_test):
    """Train Neural Network (MLP)"""
    print("\n🤖 Training Neural Network (MLP)...")
    # ✅ Increased max_iter + early stopping
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=2000,  # ✅ Changed from 500
        random_state=42,
        early_stopping=True,  # ✅ Stop early if no improvement
        validation_fraction=0.1,  # ✅ Use 10% for validation
        n_iter_no_change=50,  # ✅ Stop after 50 iterations with no improvement
        verbose=0  # Set to 1 to see training progress
    )
    mlp.fit(X_train, y_train)
    acc, f1 = evaluate_model(mlp, X_test, y_test, name="Neural Network (MLP)")
    return mlp, acc, f1


def compare_models(models, X_test, y_test):
    """Compare all models"""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        results.append({"Model": name, "Accuracy": acc, "F1-score (macro)": f1})
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x - width/2, results_df["Accuracy"], width, label="Accuracy", alpha=0.8)
    ax.bar(x + width/2, results_df["F1-score (macro)"], width, label="F1-score", alpha=0.8)
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Multiclass Model Comparison: Accuracy vs F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ✅ SAVE PNG TO RESULTS
    png_path = RESULTS_MULTICLASS / "multiclass_model_comparison.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Plot saved to: {png_path}")
    
    plt.close()
    
    return results_df


def check_overfitting(model, X_train, X_test, y_train, y_test, name="Model"):
    """Check for overfitting"""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    overfitting_risk = train_acc - test_acc
    
    print(f"\n{'='*70}")
    print(f"🔍 OVERFITTING CHECK: {name}")
    print(f"{'='*70}")
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:  {test_acc:.3f}")
    print(f"Overfitting risk: {overfitting_risk:.3f}")
    
    if overfitting_risk > 0.10:
        print("⚠️  WARNING: Possible overfitting (gap > 0.10)")
    else:
        print("✅ Good generalization (gap <= 0.10)")
    
    return train_acc, test_acc, overfitting_risk

def cross_validate_model(model, X, y, name="Model"):
    """5-fold stratified cross-validation on the FULL dataset."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")

    print(f"\n📊 {name} – 5-fold CV:")
    print(f"  Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"  F1-macro: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    return acc_scores.mean(), f1_scores.mean()

def main():
    """Main execution"""
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # ---- Train models ----
    log_reg, lr_acc, lr_f1 = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf, rf_acc, rf_f1 = train_random_forest(X_train, X_test, y_train, y_test)
    gb, gb_acc, gb_f1 = train_gradient_boosting(X_train, X_test, y_train, y_test)
    mlp, mlp_acc, mlp_f1 = train_neural_network(X_train, X_test, y_train, y_test)

    # ---- Compare models on TEST set ----
    models = {
        "Logistic Regression": log_reg,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Neural Network (MLP)": mlp
    }

    results_df = compare_models(models, X_test, y_test)

    # ---- Overfitting checks for ALL models ----
    for name, model in models.items():
        check_overfitting(model, X_train, X_test, y_train, y_test, name=name)

    # ---- 5-fold cross-validation on FULL dataset ----
    print("\n" + "="*70)
    print("🔁 5-FOLD STRATIFIED CROSS-VALIDATION (on full dataset)")
    print("="*70)

    cv_results = []
    for name, model in models.items():
        # Create a fresh clone (to avoid leakage)
        if name == "Logistic Regression":
            m = LogisticRegression(max_iter=5000)
        elif name == "Random Forest":
            m = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
        elif name == "Gradient Boosting":
            m = GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, random_state=42)
        else:  # MLP
            m = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=2000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50
            )

        cv_acc, cv_f1 = cross_validate_model(m, X, y, name=name)
        cv_results.append({
            "Model": name,
            "CV_Accuracy_mean": cv_acc,
            "CV_F1_macro_mean": cv_f1
        })

    cv_df = pd.DataFrame(cv_results)
    print("\n📄 Cross-validation summary:")
    print(cv_df.to_string(index=False))

    # ✅ SAVE CSVs TO RESULTS FOLDER (PNGs already saved above)
    
    # Save test results
    test_csv = RESULTS_MULTICLASS / "multiclass_test_results.csv"
    results_df.to_csv(test_csv, index=False)
    print(f"\n✅ Test results saved to: {test_csv}")
    
    # Save CV results
    cv_csv = RESULTS_MULTICLASS / "multiclass_cv_results.csv"
    cv_df.to_csv(cv_csv, index=False)
    print(f"✅ CV results saved to: {cv_csv}")

    print("\n" + "="*70)
    print("✅ MULTICLASS CLASSIFICATION COMPLETE")
    print("="*70)
    print(f"📁 All results saved to: {RESULTS_MULTICLASS}")
    print("="*70 + "\n")

    return {
        "models": models,
        "results_test": results_df,
        "results_cv": cv_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


if __name__ == "__main__":
    results = main()
#gta6_predictions.py
"""
Scenario prediction for a GTA 6–style announcement using ML.

- Loads ml_dataset.csv
- Trains Gradient Boosting on impact_label_num (0=Low, 1=Medium, 2=High)
- Builds bear / base / bull scenarios for a Rockstar GTA 6 "major announcement"
- Saves scenario predictions to data/processed/gta6_scenario_predictions_ml.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_MULTICLASS = BASE_DIR / "results" / "04_multiclass_classification"

print(f"BASE_DIR:       {BASE_DIR}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}\n")


# ---------------------------------------------------------------------
# Load ML dataset
# ---------------------------------------------------------------------
def load_ml_dataset():
    path = DATA_PROCESSED / "ml_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"ml_dataset.csv not found at: {path}")

    print(f"📥 Loading ML dataset from: {path}")
    # ✅ ADD sep=";" to read semicolon-delimited CSV
    df = pd.read_csv(path, sep=";")

    print(f"✅ Loaded ml_dataset: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}\n")
    return df


# ---------------------------------------------------------------------
# Build X, y for multiclass model (impact_label_num)
# ---------------------------------------------------------------------
def build_xy_for_multiclass(df):
    """
    Build feature matrix X and target y for multiclass classification.

    We mirror the feature set from train_classification.py:
      - is_rockstar
      - sentiment_negative
      - market_return
      - AR_event
      - franchise_gta
      - vix_level
      - vix_30d_mean
      - event_type_marketing
      - event_type_negative
      - event_type_release
      - vix_regime_medium
      - vix_regime_high
    """
    required_cols = [
        "is_rockstar",
        "sentiment_negative",
        "market_return",
        "AR_event",
        "franchise_gta",
        "vix_level",
        "vix_30d_mean",
        "event_type_marketing",
        "event_type_negative",
        "event_type_release",
        "vix_regime_medium",
        "vix_regime_high",
        "impact_label_num",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in ml_dataset: {missing}")

    # Drop rows with missing label
    df_clean = df.dropna(subset=["impact_label_num"]).copy()

    X = df_clean[
        [
            "is_rockstar",
            "sentiment_negative",
            "market_return",
            "AR_event",
            "franchise_gta",
            "vix_level",
            "vix_30d_mean",
            "event_type_marketing",
            "event_type_negative",
            "event_type_release",
            "vix_regime_medium",
            "vix_regime_high",
        ]
    ].copy()

    y = df_clean["impact_label_num"].astype(int)

    # Ensure numeric dtype
    X = X.apply(pd.to_numeric, errors="coerce")
    mask = X.isna().any(axis=1)
    if mask.sum() > 0:
        print(f"⚠️ Dropping {mask.sum()} rows with NaNs in features")
        X = X[~mask]
        y = y[~mask]

    print(f"📐 Final X shape: {X.shape}, y shape: {y.shape}\n")
    return X, y


# ---------------------------------------------------------------------
# Train Gradient Boosting on full dataset (after quick sanity check)
# ---------------------------------------------------------------------
def train_best_model(X, y, random_state=42):
    """
    Use GradientBoostingClassifier as the main scenario model.
    First do a quick train/test split to print a sanity report,
    then refit on the full dataset for scenario prediction.
    """
    print("🤖 Training Gradient Boosting (sanity check with train/test split)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    gb = GradientBoostingClassifier(random_state=random_state)
    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)
    print("📊 Sanity check – Gradient Boosting classification report (hold-out split):")
    print(classification_report(y_test, y_pred))
    print()

    # Refit on full data for final scenario predictions
    print("🔁 Re-training Gradient Boosting on FULL dataset for scenario predictions...")
    gb_full = GradientBoostingClassifier(random_state=random_state)
    gb_full.fit(X, y)

    return gb_full


# ---------------------------------------------------------------------
# Build scenario feature rows for a GTA 6–style announcement event
# ---------------------------------------------------------------------
def build_gta6_scenarios(model_features):
    """
    Build three scenario rows (bear/base/bull) compatible with the ML features.

    Assumptions:
      - Event: GTA 6 major announcement (trailer / key reveal)
      - Rockstar event: is_rockstar = 1
      - GTA franchise: franchise_gta = 1
      - Sentiment: positive → sentiment_negative = 0
      - Event type: "marketing" → event_type_marketing = 1, others 0
      - AR_event is set to 0 for ex-ante prediction (we don't know realized AR yet)
      - VIX regime + market_return vary by scenario
    """

    # Start with a zero vector
    base = {col: 0.0 for col in model_features}

    # Non-scenario-specific features
    base["is_rockstar"] = 1.0
    base["sentiment_negative"] = 0.0      # positive news
    base["AR_event"] = 0.0                # ex-ante prediction
    base["franchise_gta"] = 1.0
    base["event_type_marketing"] = 1.0    # announcement/trailer
    base["event_type_negative"] = 0.0
    base["event_type_release"] = 0.0

    scenarios = []

    # Bear market: negative index, high volatility
    s_bear = base.copy()
    s_bear["market_return"] = -0.02       # -2%
    s_bear["vix_level"] = 35.0
    s_bear["vix_30d_mean"] = 30.0
    s_bear["vix_regime_medium"] = 0.0
    s_bear["vix_regime_high"] = 1.0
    scenarios.append(("bear", s_bear))

    # Base market: flat index, medium volatility
    s_base = base.copy()
    s_base["market_return"] = 0.0
    s_base["vix_level"] = 20.0
    s_base["vix_30d_mean"] = 18.0
    s_base["vix_regime_medium"] = 1.0
    s_base["vix_regime_high"] = 0.0
    scenarios.append(("base", s_base))

    # Bull market: positive index, low volatility
    s_bull = base.copy()
    s_bull["market_return"] = 0.02        # +2%
    s_bull["vix_level"] = 15.0
    s_bull["vix_30d_mean"] = 15.0
    s_bull["vix_regime_medium"] = 0.0
    s_bull["vix_regime_high"] = 0.0
    scenarios.append(("bull", s_bull))

    rows = []
    for name, d in scenarios:
        row = {col: d.get(col, 0.0) for col in model_features}
        row["scenario"] = name
        rows.append(row)

    df_scenarios = pd.DataFrame(rows)
    df_scenarios = df_scenarios[model_features + ["scenario"]]

    print("✅ GTA 6 scenario feature rows (input to model):")
    print(df_scenarios, "\n")

    return df_scenarios


# ---------------------------------------------------------------------
# Map class index to label
# ---------------------------------------------------------------------
IMPACT_MAP = {
    0: "Low impact",
    1: "Medium impact",
    2: "High impact",
}


def run_gta6_scenario_predictions():
    # 1. Load dataset & build X, y
    df_ml = load_ml_dataset()
    X, y = build_xy_for_multiclass(df_ml)

    # 2. Train Gradient Boosting on full dataset
    model = train_best_model(X, y, random_state=42)

    # 3. Build scenarios
    feature_cols = list(X.columns)  # all input features
    df_scen = build_gta6_scenarios(feature_cols)

    X_scen = df_scen[feature_cols].copy()
    scenario_names = df_scen["scenario"].values

    # 4. Predict class + probabilities
    y_pred = model.predict(X_scen)
    y_proba = model.predict_proba(X_scen)

    results = []
    for i, sc in enumerate(scenario_names):
        cls = int(y_pred[i])
        label = IMPACT_MAP.get(cls, f"class_{cls}")
        proba_vec = y_proba[i]

        results.append(
            {
                "scenario": sc,
                "predicted_class": cls,
                "predicted_label": label,
                "proba_low": float(proba_vec[0]),
                "proba_medium": float(proba_vec[1]),
                "proba_high": float(proba_vec[2]),
            }
        )

    df_results = pd.DataFrame(results)

    # 5. Save results
    out_path = RESULTS_MULTICLASS / "gta6_scenario_predictions_ml.csv"
    df_results.to_csv(out_path, index=False)

    print("📊 GTA 6 scenario ML predictions:")
    print(df_results, "\n")
    print(f"✅ Saved GTA 6 scenario predictions to: {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_gta6_scenario_predictions()

#ols.py
"""
OLS regression of abnormal returns on event features and GTA Google Trends.

- Input:  data/processed/regression_dataset_with_gta_trends.csv
- Output:
    - results/02_ols_regression/ols_regression_results.txt  (all 3 models)
    - results/02_ols_regression/ols_regression_coeffs.csv   (all params)
    - results/02_ols_regression/gta_scenario_predictions.csv (bull/base/bear AR_event)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

# ✅ SETUP PATHS
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_OLS = BASE_DIR / "results" / "02_ols_regression"

# Create results folder if it doesn't exist
RESULTS_OLS.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR:       {BASE_DIR}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}")
print(f"RESULTS_OLS:    {RESULTS_OLS}\n")

# Which return we explain
TARGET_VAR = "AR_event"


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
def load_regression_data():
    """Load regression dataset with GTA trends."""
    path = DATA_PROCESSED / "regression_dataset_with_gta_trends.csv"

    if not path.exists():
        raise FileNotFoundError(f"❌ File not found: {path}")

    print(f"📥 Loading regression dataset from: {path}")
    df = pd.read_csv(path, sep=";")

    # Parse dates if present (not used in OLS, just clean)
    for col in ["event_date", "trading_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"✅ Loaded regression dataset: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}\n")
    return df


# ---------------------------------------------------------------------
# Feature preparation (cleaner, avoids dummy trap)
# ---------------------------------------------------------------------
def prepare_features(df, target_col=TARGET_VAR, use_trends=True, ttwo_only=False):
    """
    Build X, y for OLS with proper index tracking.
    """

    data = df.copy()

    # Restrict to TTWO if needed
    if ttwo_only:
        before = len(data)
        data = data[data["ticker"] == "TTWO"].copy()
        data = data.reset_index(drop=True)  # ✅ Reset index after filtering
        print(f"🔎 TTWO-only subset: {len(data)} rows (was {before})")

    # Drop rows with missing target
    data = data.dropna(subset=[target_col])
    data = data.reset_index(drop=True)  # ✅ Reset index after dropping NaN
    print(f"✅ Rows after dropping missing {target_col}: {len(data)}")

    # ----------------- Target -----------------
    y = data[target_col].copy()

    # ----------------- Base numeric features -----------------
    X = pd.DataFrame(index=data.index)

    for col in ["market_return", "is_rockstar"]:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")
        X[col] = data[col]

    # ----------------- Sentiment dummies -----------------
    if "sentiment" in data.columns:
        sent = data["sentiment"].astype(str).str.lower().str.strip()
    else:
        sent = pd.Series("", index=data.index)

    # Baseline = neutral → include only negative & positive
    X["sent_negative"] = (sent == "negative").astype(int)
    X["sent_positive"] = (sent == "positive").astype(int)

    print("   Sentiment dummies added: ['sent_negative', 'sent_positive']")

    # ----------------- Event type dummies -----------------
    if "event_type" in data.columns:
        evt = data["event_type"].astype(str).str.lower().str.strip()
    else:
        evt = pd.Series("", index=data.index)

    # Baseline = "corporate event" (all zeros)
    def evt_flag(label):
        return (evt == label).astype(int)

    X["evt_delay"] = evt_flag("delay")
    X["evt_earnings"] = evt_flag("earnings")
    X["evt_leak"] = evt_flag("leak")
    X["evt_major announcement"] = evt_flag("major announcement")
    X["evt_release"] = evt_flag("release")
    X["evt_trailer/reveal"] = evt_flag("trailer/reveal")

    print(
        "   Event-type dummies added: "
        "['evt_delay','evt_earnings','evt_leak',"
        "'evt_major announcement','evt_release','evt_trailer/reveal']"
    )

    # ----------------- GTA trend features (z-scored) -----------------
    trend_cols = []
    if use_trends:
        candidate_trends = [
            "trend_gta_z",
            "trend_gta6_z",
            "trend_gtavi_z",
            "trend_rockstar_z",
        ]
        for col in candidate_trends:
            if col in data.columns:
                X[col] = data[col].fillna(0.0)
                trend_cols.append(col)
        if trend_cols:
            print(f"   Using GTA trend z-features: {trend_cols}")
        else:
            print("⚠️ use_trends=True but no trend z-columns found → running without trends.")

    # ----------------- Numeric conversion & cleaning -----------------
    print("\n🔧 Converting X and y to numeric...")

    for col in X.columns:
        # convert bool→int and everything to numeric
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = pd.to_numeric(y, errors="coerce")

    mask = X.isna().any(axis=1) | y.isna()
    n_dropped = mask.sum()
    if n_dropped > 0:
        print(f"   Dropped {n_dropped} rows with NaN in predictors/target")
        X = X[~mask].copy()
        y = y[~mask].copy()

    # Reset indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Add constant after numeric conversion
    X = sm.add_constant(X)

    print(f"📐 Final feature matrix: X={X.shape}, y={y.shape}")
    print(f"   Columns: {list(X.columns)}\n")

    # ✅ FIXED: Return data aligned with X, y (all have same RangeIndex)
    return X, y, data  # data already has matching index


# ---------------------------------------------------------------------
# Run one OLS model
# ---------------------------------------------------------------------
def run_model(name, df, target_col=TARGET_VAR, use_trends=True, ttwo_only=False):
    X, y, data_used = prepare_features(
        df, target_col=target_col, use_trends=use_trends, ttwo_only=ttwo_only
    )

    model = sm.OLS(y, X).fit()

    print("=" * 80)
    print(name)
    print("=" * 80)
    print(model.summary())
    print("\n")

    return model, data_used


# ---------------------------------------------------------------------
# Scenario builder: bull / base / bear for GTA VI-like TTWO event
# ---------------------------------------------------------------------
def build_gta_scenarios(model):
    """
    Build bull/base/bear scenarios for a GTA VI-style event:
    - TTWO
    - is_rockstar = 1
    - positive sentiment
    - 'major announcement' event type
    - Different market returns and GTA trend z-levels.
    """

    cols = model.model.exog_names  # includes 'const'
    rows = []

    market_scenarios = {
        "bear": -0.02,   # -2% SP500 day
        "base":  0.00,   # flat
        "bull":  0.02,   # +2% SP500 day
    }

    trend_scenarios = {
        "low":  -1.0,    # 1 std below avg GTA hype
        "mid":   0.0,    # average GTA hype
        "high":  1.0,    # 1 std above avg GTA hype
    }

    for m_name, m_val in market_scenarios.items():
        for t_name, t_val in trend_scenarios.items():
            row = {c: 0.0 for c in cols}

            if "const" in row:
                row["const"] = 1.0

            # Market & Rockstar
            if "market_return" in row:
                row["market_return"] = m_val
            if "is_rockstar" in row:
                row["is_rockstar"] = 1.0  # GTA VI = Rockstar game

            # Sentiment: assume positive hype
            if "sent_negative" in row:
                row["sent_negative"] = 0.0
            if "sent_positive" in row:
                row["sent_positive"] = 1.0

            # Event type: assume "major announcement"
            for col in row.keys():
                if col.startswith("evt_"):
                    row[col] = 0.0
            if "evt_major announcement" in row:
                row["evt_major announcement"] = 1.0

            # GTA trends: set to trend z-level
            for col in ["trend_gta_z", "trend_gta6_z", "trend_gtavi_z", "trend_rockstar_z"]:
                if col in row:
                    row[col] = t_val

            row["scenario"] = f"{m_name}_{t_name}"
            rows.append(row)

    scen_df = pd.DataFrame(rows).set_index("scenario")

    # Ensure columns order matches model exog
    scen_X = scen_df[cols]
    preds = model.predict(scen_X)

    scen_df["pred_AR_event"] = preds
    scen_df["market_scenario"] = [s.split("_")[0] for s in scen_df.index]
    scen_df["trend_scenario"] = [s.split("_")[1] for s in scen_df.index]

    return scen_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    df = load_regression_data()

    # -------- Model 1: baseline, all publishers, no GTA trends --------
    m1, d1 = run_model(
        "MODEL 1 – Baseline (all publishers, no GTA trends)",
        df,
        target_col=TARGET_VAR,
        use_trends=False,
        ttwo_only=False,
    )

    # -------- Model 2: all publishers + GTA trends --------
    m2, d2 = run_model(
        "MODEL 2 – All publishers + GTA trend z-features",
        df,
        target_col=TARGET_VAR,
        use_trends=True,
        ttwo_only=False,
    )

    # -------- Model 3: TTWO only + GTA trends --------
    m3, d3 = run_model(
        "MODEL 3 – TTWO-only + GTA trend z-features",
        df,
        target_col=TARGET_VAR,
        use_trends=True,
        ttwo_only=True,
    )

    # -----------------------------------------------------------------
    # MERGE ALL OLS SUMMARIES INTO ONE FILE
    # ✅ SAVE TO RESULTS FOLDER
    # -----------------------------------------------------------------
    merged_summary_path = RESULTS_OLS / "ols_regression_results.txt"

    print(f"💾 Merging all OLS summaries into: {merged_summary_path}")

    with open(merged_summary_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("OLS REGRESSION RESULTS – ALL MODELS\n")
        f.write("=" * 100 + "\n\n")

        def write_model(name, model):
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"{name}\n")
            f.write("=" * 100 + "\n")
            f.write(model.summary().as_text())
            f.write("\n\n")

        write_model("MODEL 1 – Baseline (All Publishers, No Trends)", m1)
        write_model("MODEL 2 – All Publishers + GTA Trends", m2)
        write_model("MODEL 3 – TTWO Only + GTA Trends", m3)

    print("✅ All summaries saved to:", merged_summary_path)

    # -----------------------------------------------------------------
    # MERGE ALL COEFFICIENTS INTO ONE CSV (UNION OF PARAMETERS)
    # ✅ SAVE TO RESULTS FOLDER
    # -----------------------------------------------------------------
    merged_coeffs_path = RESULTS_OLS / "ols_regression_coeffs.csv"

    all_idx = sorted(set(m1.params.index) | set(m2.params.index) | set(m3.params.index))
    coeffs_df = pd.DataFrame(index=all_idx)
    coeffs_df["model1_baseline"] = m1.params.reindex(all_idx)
    coeffs_df["model2_all_trends"] = m2.params.reindex(all_idx)
    coeffs_df["model3_ttwo_trends"] = m3.params.reindex(all_idx)

    coeffs_df = coeffs_df.reset_index().rename(columns={"index": "parameter"})
    coeffs_df.to_csv(merged_coeffs_path, index=False)
    print("✅ All model coefficients saved to:", merged_coeffs_path)

    # -----------------------------------------------------------------
    # GTA VI BULL / BASE / BEAR SCENARIOS (using Model 2)
    # ✅ SAVE TO RESULTS FOLDER
    # -----------------------------------------------------------------
    scen_df = build_gta_scenarios(m2)
    print("\n📊 Scenario predictions (Model 2 – AR_event):")
    print(scen_df[["market_scenario", "trend_scenario", "pred_AR_event"]])

    scen_out_path = RESULTS_OLS / "gta_scenario_predictions.csv"
    scen_df.to_csv(scen_out_path, index=False)
    print("✅ Scenario predictions saved to:", scen_out_path)

    print("\n🎉 OLS regression + scenarios completed cleanly.")
    print(f"📁 All results saved to: {RESULTS_OLS}\n")


if __name__ == "__main__":
    main()