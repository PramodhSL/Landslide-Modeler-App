# =============================================================================
# File: main_engine.py
# Author: Pramodh Gamage (with AI assistance)
# Version: 2.0
# Description: Main engine with core stage functions and imports
# =============================================================================

import os
import time
import traceback
from typing import Callable, Optional, Union

import pandas as pd

from config import Config
from utils import setup_logging, log_memory_usage, identify_coordinate_columns, validate_input_data
from spatial import perform_fr_analysis
from models import perform_model_tuning, generate_model_analysis
from prediction import generate_final_map

# =============================================================================
# COMBINED HTML REPORT
# =============================================================================

def generate_combined_html_report(output_folder: str) -> Optional[str]:
    """Generate a simple combined HTML report comparing models.

    The report embeds the `model_comparison_report.csv` and any ROC/PR images
    found in `output/model_analysis/` for each model.

    Returns the path to the generated HTML file or None on failure.
    """
    try:
        report_csv = os.path.join(output_folder, "model_comparison_report.csv")
        analysis_folder = os.path.join(output_folder, "model_analysis")
        if not os.path.exists(report_csv):
            import logging
            logging.warning("No model comparison CSV found; skipping HTML report generation")
            return None

        df = pd.read_csv(report_csv)

        html_lines = [
            "<html>",
            "<head><meta charset='utf-8'><title>Model Comparison Report</title></head>",
            "<body>",
            f"<h1>Model Comparison Report</h1>",
            f"<p>Generated: {time.ctime()}</p>",
            "<h2>Summary Table</h2>",
            df.to_html(index=False, classes='table table-striped'),
            "<h2>Per-model ROC and PR curves</h2>"
        ]

        for model_name in df['Model'].tolist():
            safe_name = model_name.replace(' ', '_')
            roc_path = os.path.join(analysis_folder, f"roc_curve_{safe_name}.png")
            pr_path = os.path.join(analysis_folder, f"precision_recall_curve_{safe_name}.png")
            html_lines.append(f"<h3>{model_name}</h3>")
            if os.path.exists(roc_path):
                html_lines.append(f"<div><img src=\"{os.path.relpath(roc_path, output_folder)}\" width=600></div>")
            if os.path.exists(pr_path):
                html_lines.append(f"<div><img src=\"{os.path.relpath(pr_path, output_folder)}\" width=600></div>")

        html_lines.append("</body></html>")

        html_path = os.path.join(output_folder, "model_comparison_report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_lines))

        import logging
        logging.getLogger('LandslideAnalysis').info(f"Combined HTML report written to: {html_path}")
        return html_path

    except Exception as e:
        import logging
        logging.warning(f"Failed to generate combined HTML report: {e}")
        return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_complete_analysis(input_csv: str, raster_folder: str, output_folder: str,
                          selected_model: Optional[str] = None,
                          analyze_all_models: bool = False,
                          gui_log_callback: Optional[Callable] = None,
                          progress_callback: Optional[Callable] = None,
                          mode: str = 'full') -> Union[bool, pd.DataFrame]:
    """
    Run complete landslide susceptibility analysis pipeline.

    Args:
        input_csv: Path to input CSV with landslide data
        raster_folder: Path to folder containing raster files
        output_folder: Path to output folder
        selected_model: Optional model name. If None, uses best model from tuning.
        analyze_all_models: Whether to analyze all tuned models
        gui_log_callback: Callback for GUI logging
        progress_callback: Callback for progress updates (0-100)
        mode: 'full' for complete analysis, 'compare' for model comparison only

    Returns:
        True/False for full mode, DataFrame for compare mode
    """
    # Validate configuration
    if not Config.validate():
        print("Configuration validation failed!")
        return False

    # Setup logging
    logger = setup_logging(output_folder)

    def log_callback(message):
        logger.info(message)
        if gui_log_callback:
            gui_log_callback(message)

    try:
        logger.info("="*70)
        logger.info("LANDSLIDE SUSCEPTIBILITY ANALYSIS - ENHANCED VERSION 2.0")
        logger.info("="*70)
        logger.info(f"Input CSV: {input_csv}")
        logger.info(f"Raster folder: {raster_folder}")
        logger.info(f"Output folder: {output_folder}")
        logger.info("Configuration:")
        logger.info(f"  - Balance strategy: {Config.BALANCE_STRATEGY}")
        logger.info(f"  - Spatial CV: {Config.USE_SPATIAL_CV}")
        logger.info(f"  - Stacking ensemble: {Config.USE_STACKING}")
        logger.info(f"  - Parallel workers: {Config.PARALLEL_WORKERS}")
        logger.info("="*70)

        # If coordinates are required, check CSV first and abort if missing
        if Config.REQUIRE_COORDINATES:
            try:
                df_check = pd.read_csv(input_csv, nrows=5)
                # Case-sensitive column check
                coord_pairs = [
                    ['X_coord', 'Y_coord'],
                    ['x', 'y'],
                    ['X', 'Y']
                ]
                has_coords = any(all(col in df_check.columns for col in pair) for pair in coord_pairs)
                if not has_coords:
                    logger.error("Config.REQUIRE_COORDINATES is True but X/Y coordinates are not present in input CSV. Aborting.")
                    logger.error("Expected column pairs: X_coord/Y_coord, x/y, or X/Y")
                    return False
            except Exception as e:
                logger.error(f"Could not read input CSV to validate coordinates: {e}")
                return False

        if mode == 'map_only':
            # Determine selected_model if not provided
            if selected_model is None:
                try:
                    report_df = pd.read_csv(os.path.join(output_folder, "model_comparison_report.csv"))
                    selected_model = report_df.loc[0, "Model"]
                    logger.info(f"Using best model from comparison: {selected_model}")
                except Exception as e:
                    logger.error(f"Could not read model comparison report for map_only mode: {e}")
                    return False
            if progress_callback:
                progress_callback(0)
        else:
            # Stage 1: Frequency Ratio Analysis
            logger.info("\nStarting Stage 1: Frequency Ratio Analysis")
            if progress_callback:
                progress_callback(0)
            fr_success = perform_fr_analysis(input_csv, output_folder, log_callback, progress_callback)
            if not fr_success:
                logger.error("FR Analysis failed!")
                return False
            logger.info("FR Analysis completed successfully")
            if progress_callback:
                progress_callback(25)

            # Stage 2: Model Tuning
            logger.info("\nStarting Stage 2: Model Tuning")
            if progress_callback:
                progress_callback(30) 
            tuning_success = perform_model_tuning(input_csv, output_folder, log_callback, progress_callback)
            if not tuning_success:
                logger.error("Model tuning failed!")
                return False
            logger.info("Model tuning completed successfully")
            if progress_callback:
                progress_callback(50)

            # Determine best model (and optionally analyze all tuned models)
            try:
                report_df = pd.read_csv(os.path.join(output_folder, "model_comparison_report.csv"))
            except Exception as e:
                logger.error(f"Could not read model comparison report: {e}")
                return False

            if mode == 'compare':
                if progress_callback:
                    progress_callback(100)
                return report_df

            if mode != 'map_only':
                if analyze_all_models:
                    logger.info("\nRunning detailed analysis for all tuned models...")
                    for m in report_df['Model'].tolist():
                        logger.info(f"\n--- Detailed analysis for: {m} ---")
                        try:
                            ok = generate_model_analysis(m, input_csv, output_folder, log_callback, progress_callback)
                            if ok:
                                logger.info(f"  - Analysis complete for model: {m}")
                            else:
                                logger.warning(f"  - Analysis failed for model: {m}")
                        except Exception as e:
                            logger.warning(f"  - Exception during analysis of {m}: {e}")

            if mode != 'map_only':
                if selected_model is None:
                    # default to top-ranked model
                    selected_model = report_df.loc[0, "Model"]
                    logger.info(f"\nBest model: {selected_model}")
                else:
                    logger.info(f"\nUsing user-selected model: {selected_model}")

        # Stage 3: Map Generation
        logger.info("\nStarting Stage 3: Map Generation")
        if progress_callback:
            progress_callback(60)
        map_success = generate_final_map(selected_model, raster_folder, output_folder,
                                        log_callback, progress_callback)
        if not map_success:
            logger.error("Map generation failed!")
            return False
        logger.info("Map generation completed successfully")
        if progress_callback:
            progress_callback(80 if mode == 'full' else 50)

        # Stage 4: Model Analysis
        logger.info("\nStarting Stage 4: Model Analysis")
        if progress_callback:
            progress_callback(85)
        analysis_success = generate_model_analysis(selected_model, input_csv, output_folder,
                                                   log_callback, progress_callback)
        if not analysis_success:
            logger.error("Model analysis failed!")
            return False
        logger.info("Model analysis completed successfully")
        if progress_callback:
            progress_callback(100)

        # Summarize tuned and analyzed models
        try:
            report_df = pd.read_csv(os.path.join(output_folder, "model_comparison_report.csv"))
            tuned_models = report_df['Model'].tolist()
            logger.info(f"\nTuned models: {', '.join(tuned_models)}")
            # If analyze_all_models was requested earlier, all of them were analyzed
            analyzed_models = []
            analysis_folder = os.path.join(output_folder, 'model_analysis')
            if os.path.exists(analysis_folder):
                for f in os.listdir(analysis_folder):
                    if f.startswith('performance_summary_') and f.endswith('.csv'):
                        analyzed_models.append(f.replace('performance_summary_', '').replace('.csv', '').replace('_', ' '))
            if analyzed_models:
                logger.info(f"Analyzed models: {', '.join(analyzed_models)}")
            else:
                logger.info(f"No per-model analysis files found in {analysis_folder} (only selected model was analyzed)")
        except Exception as e:
            logger.warning(f"Could not summarize tuned/analyzed models: {e}")

        # Generate combined HTML report for convenience
        try:
            html_path = generate_combined_html_report(output_folder)
            if html_path:
                logger.info(f"Combined HTML report: {html_path}")
        except Exception as e:
            logger.warning(f"Failed to generate combined HTML report: {e}")

        logger.info("\n" + "="*70)
        logger.info("ALL STAGES COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"Output files saved to: {output_folder}")
        logger.info("\nGenerated outputs:")
        logger.info("  - fr_analysis/: Frequency ratio plots and rankings")
        logger.info("  - tuned_models/: Trained models and scalers")
        logger.info("  - model_analysis/: Performance metrics and visualizations")
        logger.info(f"  - Landslide_Susceptibility_Score_{selected_model.replace(' ', '_')}.tif")
        logger.info("  - model_comparison_report.csv")
        logger.info("  - analysis.log")

        return True
    except KeyboardInterrupt:
        logger.error("Process interrupted by user!")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return False

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Landslide Susceptibility Analysis Engine v2.0"
    )
    parser.add_argument(
        "--input-csv",
        default=os.path.join("input_data", "LS_Data.csv"),
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--raster-folder",
        default="input_rasters",
        help="Path to folder containing raster files"
    )
    parser.add_argument(
        "--output-folder",
        default="output",
        help="Path to output folder"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model to use (Random Forest, XGBoost, Support Vector Machine)"
    )
    parser.add_argument(
        "--balance-strategy",
        choices=['smote', 'undersample', 'none'],
        default='smote',
        help="Class imbalance handling strategy"
    )
    parser.add_argument(
        "--no-spatial-cv",
        action="store_true",
        help="Disable spatial cross-validation"
    )
    parser.add_argument(
        "--use-stacking",
        action="store_true",
        help="Enable stacking ensemble"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for raster processing"
    )
    parser.add_argument(
        "--analyze-all-models",
        action="store_true",
        help="Run detailed model analysis for all tuned models in model_comparison_report.csv"
    )
    parser.add_argument(
        "--require-coordinates",
        action="store_true",
        help="Fail if X/Y coordinates are not present in the input CSV (no fallback to Elevation/Slope proxy)"
    )

    args = parser.parse_args()

    # Update configuration based on arguments
    Config.BALANCE_STRATEGY = args.balance_strategy
    Config.USE_SPATIAL_CV = not args.no_spatial_cv
    Config.USE_STACKING = args.use_stacking
    Config.PARALLEL_WORKERS = args.workers
    Config.REQUIRE_COORDINATES = args.require_coordinates

    print("="*70)
    print("ENHANCED LANDSLIDE SUSCEPTIBILITY ANALYSIS ENGINE v2.0")
    print("="*70)
    print("\nChecking input files...")

    # Validate input files
    if not os.path.exists(args.input_csv):
        print(f"❌ ERROR: Input CSV file not found: {args.input_csv}")
        exit(1)

    if not os.path.exists(args.raster_folder):
        print(f"❌ ERROR: Raster folder not found: {args.raster_folder}")
        exit(1)

    required_rasters = [f"{name}.tif" for name in Config.FEATURE_NAMES]
    missing = [r for r in required_rasters
               if not os.path.exists(os.path.join(args.raster_folder, r))]
    if missing:
        print(f"❌ ERROR: Missing raster files: {missing}")
        exit(1)

    print("All input files found!")

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Run analysis
    success = run_complete_analysis(
        args.input_csv,
        args.raster_folder,
        args.output_folder,
        args.model,
        analyze_all_models=args.analyze_all_models
    )

    exit(0 if success else 1)