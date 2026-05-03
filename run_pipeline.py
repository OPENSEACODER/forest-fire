"""
run_pipeline.py
---------------
One-shot command-line runner for all VanSuraksha pipeline stages.

Usage
-----
Run all stages:
    python run_pipeline.py

Run individual stages:
    python run_pipeline.py --stage 1   # data fetch + merge
    python run_pipeline.py --stage 2   # feature engineering
    python run_pipeline.py --stage 4   # model training

The stages match the original script numbering (1, 2, 4).
Stage 3 (EDA) is run automatically between 2 and 4 when running all stages.
"""

import argparse
import sys
import time

from logger import logger


def run_stage_1():
    from pipeline.data_loader import run
    logger.info("── Stage 1: Data fetch & merge ──────────────────────")
    run()


def run_stage_2():
    from pipeline.preprocessor import run
    logger.info("── Stage 2: Feature engineering ─────────────────────")
    run()


def run_stage_3():
    """Optional EDA — skipped silently if seaborn/matplotlib unavailable."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from pipeline.eda import run_full_eda
        logger.info("── Stage 3: EDA ──────────────────────────────────────")
        run_full_eda()
    except (ImportError, FileNotFoundError) as e:
        logger.warning("Stage 3 (EDA) skipped: %s", e)


def run_stage_4():
    from pipeline.trainer import run
    logger.info("── Stage 4: Model training ───────────────────────────")
    run()


def main():
    parser = argparse.ArgumentParser(description="VanSuraksha pipeline runner")
    parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run a specific stage (1=data, 2=preprocess, 3=eda, 4=train). "
             "Omit to run all stages in sequence.",
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.stage == 1:
        run_stage_1()
    elif args.stage == 2:
        run_stage_2()
    elif args.stage == 3:
        run_stage_3()
    elif args.stage == 4:
        run_stage_4()
    else:
        # Full pipeline
        run_stage_1()
        run_stage_2()
        run_stage_3()
        run_stage_4()

    logger.info("=== Pipeline complete in %.2f min ===", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
