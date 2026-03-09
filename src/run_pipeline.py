"""
run_pipeline.py
===============
Full pipeline orchestrator.

Run order
---------
1. build_clean_dataset  -- construct leak-free 21d modeling table
2. run_eda              -- 13 EDA visualizations + dataset profile
3. train_models_main    -- 9 sklearn models, GroupCV, calibration, threshold, temporal holdout
4. create_evaluation_report -- calibration curve, subgroup, permutation importance
5. generate_shap_artifacts  -- SHAP global / local / group
6. train_neural_network     -- Keras MLP (5-fold CV + holdout)

Usage
-----
  python -m src.run_pipeline              # full pipeline
  python -m src.run_pipeline --skip-nn   # skip neural network (faster)
"""
from __future__ import annotations

import argparse
import sys
import time


def run_all(skip_nn: bool = False) -> None:
    t0 = time.time()

    print("=" * 60)
    print("OULAD Pipeline — Step 1/6: Build clean dataset")
    print("=" * 60)
    from src.build_clean_dataset import build_clean_dataset
    build_clean_dataset()

    print("\n" + "=" * 60)
    print("OULAD Pipeline — Step 2/6: EDA")
    print("=" * 60)
    from src.eda import run_eda
    run_eda()

    print("\n" + "=" * 60)
    print("OULAD Pipeline — Step 3/6: Model training (9 models, GroupCV)")
    print("=" * 60)
    from src.train_models import main as train_models_main
    train_models_main()

    print("\n" + "=" * 60)
    print("OULAD Pipeline — Step 4/6: Evaluation report")
    print("=" * 60)
    from src.evaluate_models import create_evaluation_report
    create_evaluation_report()

    print("\n" + "=" * 60)
    print("OULAD Pipeline — Step 5/6: SHAP explainability")
    print("=" * 60)
    from src.explainability import generate_shap_artifacts
    generate_shap_artifacts()

    if not skip_nn:
        print("\n" + "=" * 60)
        print("OULAD Pipeline — Step 6/6: Neural network (Keras)")
        print("=" * 60)
        from src.neural_network import train_neural_network
        train_neural_network()
    else:
        print("\nNeural network skipped (--skip-nn).")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Full pipeline complete in {elapsed/60:.1f} minutes.")
    print(f"Artifacts written to: artifacts/")
    print(f"Models saved to:      models/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OULAD full pipeline.")
    parser.add_argument("--skip-nn", action="store_true",
                        help="Skip neural network training (faster iteration).")
    args = parser.parse_args()
    run_all(skip_nn=args.skip_nn)
