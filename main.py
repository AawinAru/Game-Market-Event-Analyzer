"""
Main entry point for the Game Market Event Analyzer project.
Complete pipeline orchestration:
  0. Data Loading (download & build prices)
  1. Features Engineering (compute returns, AR, CAR, labels)
  2. Regression Dataset (merge with GTA trends)
  3. ML Dataset (prepare for classification)
  4. OLS Regression (abnormal returns analysis)
  5. Binary Classification (high vs not high impact)
  6. Multiclass Classification (low, medium, high impact)
  7. GTA 6 Scenario Predictions
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

# ✅ CREATE RESULTS DIRECTORY STRUCTURE
RESULTS_DIR = BASE_DIR / "results"
RESULTS_OLS = RESULTS_DIR / "02_ols_regression"
RESULTS_BINARY = RESULTS_DIR / "03_binary_classification"
RESULTS_MULTICLASS = RESULTS_DIR / "04_multiclass_classification"
RESULTS_SUMMARY = RESULTS_DIR / "05_summary"

for dir_path in [RESULTS_OLS, RESULTS_BINARY, RESULTS_MULTICLASS, RESULTS_SUMMARY]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("🎮 GAME MARKET EVENT ANALYZER – COMPLETE PIPELINE")
print("="*80)
print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def run_pipeline():
    """Run the complete analysis pipeline."""
    
    # ✅ STEP 0: Data Loading & Download
    print("\n" + "=" * 80)
    print("📥 STEP 0: DATA LOADING & DOWNLOAD")
    print("=" * 80)
    try:
        from eventstudy.data.data_loader import main as data_loader_main
        print("▶️  Running data_loader.py...")
        prices_long = data_loader_main()  # ✅ Only returns prices_long now
        print("✅ Step 0 complete: Raw data downloaded & prices built\n")
    except Exception as e:
        print(f"❌ Step 0 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ✅ STEP 1: Features Engineering
    print("\n" + "=" * 80)
    print("📊 STEP 1: FEATURES ENGINEERING")
    print("=" * 80)
    try:
        from eventstudy.features import (
            compute_returns,
            merge_event_returns,
            compute_ar_car,
            car_into_labels,
        )
        from eventstudy.features.build_ml_dataset import build_ml_dataset
        
        print("▶️  Step 1a: compute_returns...")
        compute_returns()
        
        print("▶️  Step 1b: merge_event_returns...")
        merge_event_returns()
        
        print("▶️  Step 1c: compute_ar_car...")
        compute_ar_car()
        
        print("▶️  Step 1d: car_into_labels...")
        car_into_labels()
        
        print("✅ Step 1 complete: Features engineered\n")
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ✅ STEP 2: Build ML Dataset
    print("\n" + "=" * 80)
    print("🎯 STEP 2: BUILD ML DATASET")
    print("=" * 80)
    try:
        print("▶️  Running build_ml_dataset...")
        build_ml_dataset()
        print("✅ Step 2 complete: ML dataset built\n")
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ✅ STEP 3: Trends Analysis & Regression Dataset
    print("\n" + "=" * 80)
    print("📈 STEP 3: GTA TRENDS & REGRESSION DATASET")
    print("=" * 80)
    try:
        print("▶️  Running trends.py...")
        from eventstudy.features.trends import main as trends_main
        trends_main()
        print("✅ Step 3 complete: Regression dataset with GTA trends\n")
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


    # ✅ STEP 4: OLS Regression
    print("\n" + "=" * 80)
    print("📉 STEP 4: OLS REGRESSION ANALYSIS")
    print("=" * 80)
    try:
        print("▶️  Running ols.py...")
        from eventstudy.models.ols import main as ols_main
        ols_main()
        print("✅ Step 4 complete: OLS regression results saved\n")
    except Exception as e:
        print(f"❌ Step 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ✅ STEP 5: Multiclass Classification
    print("\n" + "=" * 80)
    print("🎯 STEP 6: MULTICLASS CLASSIFICATION MODELS")
    print("=" * 80)
    try:
        print("▶️  Running train_classification.py...")
        from eventstudy.models.train_classification import main as multiclass_main
        multiclass_main()
        print("✅ Step 6 complete: Multiclass classification results saved\n")
    except Exception as e:
        print(f"❌ Step 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ✅ STEP 6: Binary Classification
    print("\n" + "=" * 80)
    print("🤖 STEP 5: BINARY CLASSIFICATION MODELS")
    print("=" * 80)
    try:
        print("▶️  Running train_binary.py...")
        from eventstudy.models.train_binary import main as binary_main
        binary_main()
        print("✅ Step 5 complete: Binary classification results saved\n")
    except Exception as e:
        print(f"❌ Step 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ✅ STEP 7: GTA 6 Predictions
    print("\n" + "=" * 80)
    print("🎮 STEP 7: GTA 6 SCENARIO PREDICTIONS")
    print("=" * 80)
    try:
        print("▶️  Running gta6_prediction.py...")
        from eventstudy.models.gta6_prediction import main as gta6_main
        gta6_main()
        print("✅ Step 7 complete: GTA 6 predictions saved\n")
    except Exception as e:
        print(f"❌ Step 7 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ✅ STEP 8: Backtests (ML & OLS)
    print("\n" + "=" * 80)
    print("🔍 STEP 8: BACKTESTS ON GTA VI EVENTS")
    print("=" * 80)
    try:
        from eventstudy.models.backtest_ml import main as backtest_ml_main
        from eventstudy.models.backtest_ols import main as backtest_ols_main

        backtest_ml_main()
        backtest_ols_main()
        print("✅ Step 8 complete: Backtest results saved\n")
    except Exception as e:
        print(f"❌ Step 8 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def print_final_summary():
    """Print final summary of all results."""
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print("\n📁 RESULTS DIRECTORY STRUCTURE:")
    print(f"\n   {RESULTS_DIR}/")
    print(f"   ├── 02_ols_regression/")
    print(f"   │   ├── ols_regression_results.txt")
    print(f"   │   ├── ols_regression_coeffs.csv")
    print(f"   │   └── gta_scenario_predictions.csv")
    print(f"   ├── 03_binary_classification/")
    print(f"   │   ├── binary_test_results.csv")
    print(f"   │   ├── binary_cv_results.csv")
    print(f"   │   ├── binary_model_comparison.png")
    print(f"   │   └── *_confusion_matrix.png")
    print(f"   ├── 04_multiclass_classification/")
    print(f"   │   ├── multiclass_test_results.csv")
    print(f"   │   ├── multiclass_cv_results.csv")
    print(f"   │   ├── gta6_scenario_predictions_ml.csv")
    print(f"   │   ├── multiclass_model_comparison.png")
    print(f"   │   └── *_confusion_matrix.png")
    print(f"   └── 05_summary/")
    
    
    print("\n📈 ANALYSIS OUTPUTS:")
    print("   ✅ Step 4: OLS Regression (3 models + scenario predictions)")
    print("   ✅ Step 5: Binary Classification (4 models: LR, RF, GB, MLP)")
    print("   ✅ Step 6: Multiclass Classification (4 models: LR, RF, GB, MLP)")
    print("   ✅ Step 7: GTA 6 Scenario Predictions (ML-based)")
    print("   ✅ Step 8: Backtests on GTA VI Events (ML & OLS)")
    print("   ✅ Confusion Matrices & Model Comparisons (PNG)")
    print("   ✅ Cross-Validation Results (CSV)")
    print("   ✅ Feature Importance & Detailed Reports")
    
    print("\n" + "=" * 80)
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 All results saved to: {RESULTS_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        success = run_pipeline()
        if success:
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
