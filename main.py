"""
Main entry point for the Game Market Event Analyzer project.
Orchestrates the entire pipeline:
  1. Data preparation (trends.py)
  2. OLS regression (ols.py)
  3. Binary classification (train_binary.py)
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

print("=" * 80)
print("🎮 GAME MARKET EVENT ANALYZER – MAIN PIPELINE")
print("=" * 80)
print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def run_pipeline():
    """Run the complete analysis pipeline."""
    
    # ✅ STEP 1: Download/Prepare GTA Trends + Build Regression Dataset
    print("\n" + "=" * 80)
    print("📊 STEP 1: PREPARE DATA WITH GTA TRENDS")
    print("=" * 80)
    try:
        from eventstudy.features.trends import main as trends_main
        print("▶️  Running trends.py...")
        trends_main()
        print("✅ Step 1 complete: regression_dataset_with_gta_trends.csv created\n")
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        return False

    # ✅ STEP 2: OLS Regression Analysis
    print("\n" + "=" * 80)
    print("📈 STEP 2: OLS REGRESSION ANALYSIS")
    print("=" * 80)
    try:
        from eventstudy.models.ols import main as ols_main
        print("▶️  Running ols.py...")
        ols_main()
        print("✅ Step 2 complete: OLS regression results saved\n")
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        return False

    # ✅ STEP 3: Binary Classification Models
    print("\n" + "=" * 80)
    print("🤖 STEP 3: BINARY CLASSIFICATION MODELS")
    print("=" * 80)
    try:
        from eventstudy.models.train_binary import main as binary_main
        print("▶️  Running train_binary.py...")
        binary_main()
        print("✅ Step 3 complete: Binary classification results saved\n")
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        return False

    return True


def print_summary():
    """Print final summary of outputs."""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PROCESSED = BASE_DIR / "data" / "processed"
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print("\n📁 Output Files Generated:")
    print("\n🔹 Regression Data:")
    print(f"   • {DATA_PROCESSED / 'regression_dataset_with_gta_trends.csv'}")
    
    print("\n🔹 OLS Results:")
    print(f"   • {DATA_PROCESSED / 'ols_regression_results.txt'}")
    print(f"   • {DATA_PROCESSED / 'ols_regression_coeffs.csv'}")
    print(f"   • {DATA_PROCESSED / 'gta_scenario_predictions.csv'}")
    
    print("\n🔹 Binary Classification Results:")
    print(f"   • {DATA_PROCESSED / 'binary_all_plots.pdf'}")
    print(f"   • {DATA_PROCESSED / 'binary_all_results.xlsx'}")
    
    print("\n" + "=" * 80)
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        success = run_pipeline()
        if success:
            print_summary()
            sys.exit(0)
        else:
            print("\n❌ Pipeline failed. Check errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




        __________________


"""
Main entry point for the Game Market Event Analyzer project.
Orchestrates the entire pipeline:
  1. Data preparation (trends.py)
  2. OLS regression (ols.py)
  3. Binary classification (train_binary.py)
  4. Multiclass classification (train_multiclass.py)
"""

import sys
from pathlib import Path
from datetime import datetime
import shutil

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

# ✅ CREATE RESULTS DIRECTORY STRUCTURE
RESULTS_DIR = BASE_DIR / "results"
RESULTS_TRENDS = RESULTS_DIR / "01_trends_analysis"
RESULTS_OLS = RESULTS_DIR / "02_ols_regression"
RESULTS_BINARY = RESULTS_DIR / "03_binary_classification"
RESULTS_MULTICLASS = RESULTS_DIR / "04_multiclass_classification"
RESULTS_SUMMARY = RESULTS_DIR / "05_summary"

for dir_path in [RESULTS_TRENDS, RESULTS_OLS, RESULTS_BINARY, RESULTS_MULTICLASS, RESULTS_SUMMARY]:
    dir_path.mkdir(parents=True, exist_ok=True)

DATA_PROCESSED = BASE_DIR / "data" / "processed"

print("=" * 80)
print("🎮 GAME MARKET EVENT ANALYZER – MAIN PIPELINE")
print("=" * 80)
print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def copy_file(src, dest_dir, dest_name=None):
    """Copy file to results directory."""
    if dest_name is None:
        dest_name = src.name
    
    if src.exists():
        shutil.copy(src, dest_dir / dest_name)
        print(f"   ✅ Copied: {dest_name}")
    else:
        print(f"   ⚠️  Not found: {src}")


def run_pipeline():
    """Run the complete analysis pipeline."""
    
    # ✅ STEP 1: Download/Prepare GTA Trends
    print("\n" + "=" * 80)
    print("📊 STEP 1: PREPARE DATA WITH GTA TRENDS")
    print("=" * 80)
    try:
        from eventstudy.features.trends import main as trends_main
        print("▶️  Running trends.py...")
        trends_main()
        print("✅ Step 1 complete\n")
        
        # Copy trends results
        print("📁 Organizing trends results...")
        copy_file(
            DATA_PROCESSED / "regression_dataset_with_gta_trends.csv",
            RESULTS_TRENDS,
            "regression_dataset_with_gta_trends.csv"
        )
        copy_file(
            DATA_PROCESSED / "gta_trends.csv",
            RESULTS_TRENDS,
            "gta_trends_raw.csv"
        )
        
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ✅ STEP 2: OLS Regression
    print("\n" + "=" * 80)
    print("📈 STEP 2: OLS REGRESSION ANALYSIS")
    print("=" * 80)
    try:
        from eventstudy.models.ols import main as ols_main
        print("▶️  Running ols.py...")
        ols_main()
        print("✅ Step 2 complete\n")
        
        # Copy OLS results
        print("📁 Organizing OLS results...")
        copy_file(
            DATA_PROCESSED / "ols_regression_results.txt",
            RESULTS_OLS,
            "ols_regression_results.txt"
        )
        copy_file(
            DATA_PROCESSED / "ols_regression_coeffs.csv",
            RESULTS_OLS,
            "ols_regression_coefficients.csv"
        )
        copy_file(
            DATA_PROCESSED / "gta_scenario_predictions.csv",
            RESULTS_OLS,
            "gta_scenario_predictions.csv"
        )
        
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ✅ STEP 3: Binary Classification
    print("\n" + "=" * 80)
    print("🤖 STEP 3: BINARY CLASSIFICATION MODELS")
    print("=" * 80)
    try:
        from eventstudy.models.train_binary import main as binary_main
        print("▶️  Running train_binary.py...")
        binary_main()
        print("✅ Step 3 complete\n")
        
        # Copy binary results
        print("📁 Organizing binary classification results...")
        copy_file(
            DATA_PROCESSED / "binary_all_plots.pdf",
            RESULTS_BINARY,
            "binary_model_plots.pdf"
        )
        copy_file(
            DATA_PROCESSED / "binary_all_results.xlsx",
            RESULTS_BINARY,
            "binary_model_results.xlsx"
        )
        
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ✅ STEP 4: Multiclass Classification
    print("\n" + "=" * 80)
    print("🎯 STEP 4: MULTICLASS CLASSIFICATION MODELS")
    print("=" * 80)
    try:
        from eventstudy.models.train_multiclass import main as multiclass_main
        print("▶️  Running train_multiclass.py...")
        multiclass_main()
        print("✅ Step 4 complete\n")
        
        # Copy multiclass results
        print("📁 Organizing multiclass classification results...")
        copy_file(
            DATA_PROCESSED / "multiclass_all_plots.pdf",
            RESULTS_MULTICLASS,
            "multiclass_model_plots.pdf"
        )
        copy_file(
            DATA_PROCESSED / "multiclass_all_results.xlsx",
            RESULTS_MULTICLASS,
            "multiclass_model_results.xlsx"
        )
        
    except Exception as e:
        print(f"❌ Step 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def create_summary_report():
    """Create a markdown summary of all results."""
    summary_file = RESULTS_SUMMARY / "README.md"
    
    content = f"""# Game Market Event Analyzer - Results Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Project Overview
This analysis examines abnormal returns (AR) in game publisher stocks around significant GTA-related events, incorporating Google Trends data.

---

## 📁 Results Structure

### 1️⃣ Trends Analysis (`01_trends_analysis/`)
- **regression_dataset_with_gta_trends.csv**
  - Complete dataset with GTA Google Trends
  - Columns: Event data, abnormal returns, GTA trend z-scores
  
- **gta_trends_raw.csv**
  - Raw weekly Google Trends data
  - Keywords: GTA, GTA 6, GTA VI, Rockstar Games

---

### 2️⃣ OLS Regression (`02_ols_regression/`)
- **ols_regression_results.txt**
  - 3 statistical models with significance tests
  - Model 1: Baseline (all publishers, no trends)
  - Model 2: All publishers + GTA trends
  - Model 3: TTWO only + GTA trends
  
- **ols_regression_coefficients.csv**
  - Regression parameters & significance
  - p-values, R², adjusted R², confidence intervals
  
- **gta_scenario_predictions.csv**
  - Bull/Base/Bear scenario predictions
  - Impact of different trend levels on abnormal returns

---

### 3️⃣ Binary Classification (`03_binary_classification/`)
**Target:** `impact_high` (0 = not high, 1 = high)

- **binary_model_plots.pdf**
  - Confusion matrices (4 models)
  - Model comparison visualization
  
- **binary_model_results.xlsx**
  - Sheet 1: Test set performance (Accuracy, F1-score)
  - Sheet 2: 5-fold cross-validation results

**Models Trained:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- Neural Network (MLP)

---

### 4️⃣ Multiclass Classification (`04_multiclass_classification/`)
**Target:** `impact_label` (3 classes: low, medium, high)

- **multiclass_model_plots.pdf**
  - Confusion matrices (4 models)
  - Per-class performance metrics
  - Model comparison visualization
  
- **multiclass_model_results.xlsx**
  - Sheet 1: Test set performance (Accuracy, F1-macro, F1-weighted)
  - Sheet 2: 5-fold cross-validation results
  - Sheet 3: Per-class precision/recall/F1

**Models Trained:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- Neural Network (MLP)

---

### 5️⃣ Summary (`05_summary/`)
- **README.md** - This file
- **RESULTS.txt** - Detailed findings

---

## 🎯 Key Outputs

| Analysis | Format | Location |
|----------|--------|----------|
| Regression Data | CSV | `01_trends_analysis/` |
| OLS Models | TXT, CSV | `02_ols_regression/` |
| Binary Classification | PDF, XLSX | `03_binary_classification/` |
| Multiclass Classification | PDF, XLSX | `04_multiclass_classification/` |

---

## 📊 Model Comparison

### Classification Models Performance
All models evaluated using:
- **Accuracy** - Overall correctness
- **F1-score** - Balance between precision & recall
- **Confusion Matrix** - Detailed error analysis
- **5-Fold Cross-Validation** - Generalization capability

---

## 🚀 How to Use Results

1. **Regression Analysis**: 
   - Open `ols_regression_results.txt` for detailed statistics
   - Use coefficients to understand feature importance

2. **Scenario Analysis**: 
   - Use `gta_scenario_predictions.csv` for bull/base/bear scenarios

3. **Binary Classification**: 
   - Check `binary_model_results.xlsx` for which model performs best
   - Use best model for binary predictions (high vs not high impact)

4. **Multiclass Classification**: 
   - Check `multiclass_model_results.xlsx` for multi-level predictions
   - Use for 3-class impact predictions (low, medium, high)

5. **Visualizations**: 
   - View PDFs for confusion matrices and model comparisons

---

## ✅ Pipeline Status

- ✅ Trends data collected & processed
- ✅ OLS regression models fitted & analyzed
- ✅ Binary classification models trained & evaluated
- ✅ Multiclass classification models trained & evaluated
- ✅ Results organized & saved

"""
    
    summary_file.write_text(content)
    print(f"📄 Summary report created: {summary_file}")


def print_final_summary():
    """Print final summary of all results."""
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print("\n📁 RESULTS STRUCTURE:")
    print(f"\n   {RESULTS_DIR}/")
    print(f"   ├── 01_trends_analysis/")
    print(f"   │   ├── regression_dataset_with_gta_trends.csv")
    print(f"   │   └── gta_trends_raw.csv")
    print(f"   ├── 02_ols_regression/")
    print(f"   │   ├── ols_regression_results.txt")
    print(f"   │   ├── ols_regression_coefficients.csv")
    print(f"   │   └── gta_scenario_predictions.csv")
    print(f"   ├── 03_binary_classification/")
    print(f"   │   ├── binary_model_plots.pdf")
    print(f"   │   └── binary_model_results.xlsx")
    print(f"   ├── 04_multiclass_classification/")
    print(f"   │   ├── multiclass_model_plots.pdf")
    print(f"   │   └── multiclass_model_results.xlsx")
    print(f"   └── 05_summary/")
    print(f"       └── README.md")
    
    print("\n📊 What's Included:")
    print("   • Regression analysis with GTA trends")
    print("   • OLS statistical models + scenario predictions")
    print("   • Binary classification (high vs not high impact)")
    print("   • Multiclass classification (low, medium, high impact)")
    print("   • Confusion matrices & model comparisons")
    print("   • 5-fold cross-validation results")
    
    print("\n" + "=" * 80)
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 All results saved to: {RESULTS_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        success = run_pipeline()
        if success:
            create_summary_report()
            print_final_summary()
            sys.exit(0)
        else:
            print("\n❌ Pipeline failed. Check errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)