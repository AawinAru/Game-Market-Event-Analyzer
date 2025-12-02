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

    # ✅ Use the SAME feature list as in gta6_prediction.py
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
    y = df[target]

    print(f"Feature matrix X shape (before NaN drop): {X.shape}")
    print(f"Target y shape (before NaN drop): {y.shape}")

    # DROP NaN ROWS AND RESET INDEX
    mask = X.isna().any(axis=1) | y.isna()
    X = X[~mask].reset_index(drop=True)
    y = y[~mask].reset_index(drop=True)

    print(f"Feature matrix X shape (after NaN drop): {X.shape}")
    print(f"Target y shape (after NaN drop): {y.shape}\n")

    return X, y


def split_data(X, y):
    """Split data into train/test with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
    
    # ✅ Reset indices after split to avoid misalignment
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

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
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42
    )
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