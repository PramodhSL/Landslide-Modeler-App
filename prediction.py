# =============================================================================
# File: prediction.py
# Author: Pramodh Gamage (with AI assistance)
# Version: 2.0
# Description: Final map generation with parallel processing
# =============================================================================

import os
import time
import logging
from typing import Callable, Tuple, List, Dict, Any, Optional

import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

from config import Config
from utils import safe_read_window, close_srcs, validate_file_exists, log_error_details, log_memory_usage

# =============================================================================
# UNCERTAINTY QUANTIFICATION
# =============================================================================

def predict_with_uncertainty(model: Any, X: pd.DataFrame, scaler: StandardScaler,
                            use_scaled: bool, n_bootstrap: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions with bootstrap confidence intervals.

    Returns:
        mean_predictions, std_predictions
    """
    predictions = []
    n_samples = len(X)

    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X.iloc[indices]

        # Predict
        if use_scaled:
            X_boot_processed = scaler.transform(X_boot)
        else:
            X_boot_processed = X_boot

        pred = model.predict_proba(X_boot_processed)[:, 1]
        predictions.append(pred)

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)

    return mean_pred, std_pred

# =============================================================================
# PREDICTION MONITORING
# =============================================================================

def monitor_prediction_distribution(predictions: np.ndarray, output_folder: str,
                                   model_name: str) -> None:
    """Monitor and visualize prediction distributions."""
    try:
        valid_preds = predictions[predictions != Config.NODATA_VALUE]

        if len(valid_preds) == 0:
            logging.warning("No valid predictions to monitor")
            return

        plt.figure(figsize=(12, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(valid_preds, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Susceptibility Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predictions')
        plt.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'Mean: {np.mean(valid_preds):.3f}\n'
        stats_text += f'Median: {np.median(valid_preds):.3f}\n'
        stats_text += f'Std: {np.std(valid_preds):.3f}\n'
        stats_text += f'Min: {np.min(valid_preds):.3f}\n'
        stats_text += f'Max: {np.max(valid_preds):.3f}'

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=9)

        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(valid_preds, vert=True)
        plt.ylabel('Susceptibility Probability')
        plt.title('Prediction Distribution (Box Plot)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'prediction_distribution_{model_name.replace(" ", "_")}.png'),
                   dpi=Config.FIGURE_DPI)
        plt.close()

        logging.info(f"  - Prediction statistics: Mean={np.mean(valid_preds):.3f}, "
                    f"Std={np.std(valid_preds):.3f}")

    except Exception as e:
        logging.warning(f"Prediction monitoring failed: {e}")

# =============================================================================
# FINAL MAP GENERATION WITH PARALLEL PROCESSING
# =============================================================================

def process_raster_block(block_info: Dict, raster_paths: List[str], model: Any,
                        scaler: StandardScaler, use_scaled: bool,
                        feature_order: List[str], nodata_value: float,
                        model_feature_list: Optional[List[str]] = None) -> Tuple[np.ndarray, Window]:
    """Process a single raster block for parallel execution.

    model_feature_list: list of features used at training time (may include coords).
    feature_order: list of raster feature names (should match raster_paths order).
    """
    # Use the provided model_feature_list from the main thread; do not try to load files
    feature_list = model_feature_list if model_feature_list is not None else feature_order

    # If feature_order (raster features) doesn't match what the model expects (feature_list),
    # ensure we predict using only the raster-backed features (exclude coordinates)
    if len(feature_order) != len(feature_list):
        feature_order = [f for f in feature_list if f in Config.FEATURE_NAMES]
    try:
        window = block_info['window']
        # Open raster files locally in this worker to avoid sharing dataset handles
        srcs_local = [rasterio.open(p) for p in raster_paths]
        try:
            # Read all features for this block
            arrays = [safe_read_window(src, window) for src in srcs_local]
        finally:
            for s in srcs_local:
                try:
                    s.close()
                except Exception:
                    pass

        height, width = arrays[0].shape

        # Stack into dataframe
        flattened_arrays = [arr.flatten() for arr in arrays]
        data_for_df = np.column_stack(flattened_arrays)
        df_chunk = pd.DataFrame(data_for_df, columns=feature_order if feature_order is not None else Config.FEATURE_NAMES)

        # Build nodata mask
        combined_nodata_mask = df_chunk.isnull().any(axis=1).values

        # Initialize predictions with nodata
        predictions = np.full(df_chunk.shape[0], nodata_value, dtype=np.float32)
        valid_idx = np.where(~combined_nodata_mask)[0]

        if valid_idx.size > 0:
            valid_data = df_chunk.iloc[valid_idx].copy()

            # Check for unexpected NaN/inf
            if valid_data.isnull().any(axis=None) or not np.isfinite(valid_data.values).all():
                problematic_mask = valid_data.isnull().any(axis=1) | ~np.isfinite(valid_data.values).all(axis=1)
                problematic_indices = valid_idx[problematic_mask]
                predictions[problematic_indices] = nodata_value

                clean_mask = ~problematic_mask
                if clean_mask.any():
                    valid_data = valid_data[clean_mask]
                    clean_valid_idx = valid_idx[clean_mask]
                else:
                    return predictions.reshape(height, width), window
            else:
                clean_valid_idx = valid_idx

            # Predict
            if clean_valid_idx.size > 0:
                # Get expected feature names from the model
                model_feats = None
                try:
                    model_feats = list(model.feature_names_in_)
                except Exception:
                    # If model doesn't have feature_names_in_, try feature_list or Config.FEATURE_NAMES
                    if feature_list:
                        model_feats = [f for f in feature_list if f in Config.FEATURE_NAMES]
                    else:
                        model_feats = Config.FEATURE_NAMES

                if model_feats and len(model_feats) > 0:
                    missing_feats = [f for f in model_feats if f not in valid_data.columns]
                    if missing_feats:
                        logging.getLogger('LandslideAnalysis').warning(
                            f"Model expects features not present in raster block: {missing_feats}. Filling with 0.0"
                        )
                        for mf in missing_feats:
                            valid_data[mf] = 0.0
                    # Reorder columns to model order
                    valid_data = valid_data[model_feats]

                Xpred = valid_data

                if use_scaled:
                    # Ensure scaler expects same number of features as Xpred
                    try:
                        Xpred = scaler.transform(Xpred)
                    except Exception as e:
                        logging.getLogger('LandslideAnalysis').warning(
                            f"Scaler transform failed for block: {e}. Attempting to align features to scaler."
                        )
                        # Try aligning to scaler's feature count by trimming or padding
                        if hasattr(scaler, 'n_features_in_'):
                            n_needed = int(scaler.n_features_in_)
                            if Xpred.shape[1] > n_needed:
                                Xpred = Xpred[:, :n_needed]
                            elif Xpred.shape[1] < n_needed:
                                pad = np.zeros((Xpred.shape[0], n_needed - Xpred.shape[1]), dtype=Xpred.dtype)
                                Xpred = np.hstack([Xpred, pad])
                            try:
                                Xpred = scaler.transform(Xpred)
                            except Exception:
                                logging.getLogger('LandslideAnalysis').warning("Final scaler.transform attempt failed; skipping block predictions")
                                Xpred = None

                if Xpred is not None and hasattr(model, "predict_proba"):
                    probs = model.predict_proba(Xpred)[:, 1]
                    predictions[clean_valid_idx] = probs.astype(np.float32)

        return predictions.reshape(height, width), window

    except Exception as e:
        logging.warning(f"Error processing block: {e}")
        return np.full((block_info['height'], block_info['width']), nodata_value, dtype=np.float32), block_info['window']

def generate_final_map(selected_model_name: str, raster_folder: str,
                      output_folder: str, log_callback: Callable,
                      progress_callback: Callable,
                      block_size: int = Config.DEFAULT_BLOCK_SIZE) -> bool:
    """Generates final landslide susceptibility map with parallel processing."""
    start_time = time.time()
    src_files = []

    try:
        logger = logging.getLogger('LandslideAnalysis')
        logger.info("\n" + "="*50)
        logger.info("--- STAGE 3: Generating Final Susceptibility Map ---")
        logger.info("="*50)
        logger.info(f"  - Selected model: '{selected_model_name}'")

            # If user passed a numeric selection (e.g., '2'), map it to the model name
        try:
            sel_idx = int(selected_model_name)
            report_path = os.path.join(output_folder, "model_comparison_report.csv")
            if not os.path.exists(report_path):
                logger.error("Model comparison report not found. Please run Stage 2 (model tuning) first.")
                return False
            try:
                df_report = pd.read_csv(report_path)
                if sel_idx < 1 or sel_idx > len(df_report):
                    logger.error(f"Invalid model selection {sel_idx}. Must be between 1 and {len(df_report)}")
                    logger.error("Available models:")
                    for idx, model in enumerate(df_report['Model'], 1):
                        logger.error(f"  {idx}: {model}")
                    return False
                mapped = df_report.loc[sel_idx - 1, 'Model']
                logger.info(f"Mapped numeric model selection {sel_idx} -> '{mapped}'")
                selected_model_name = mapped
            except Exception as e:
                logger.error(f"Could not read model report or map selection: {e}")
                return False
        except ValueError:
            # not an integer, proceed as before
            pass        # Load model and scaler
        model_filename = os.path.join(output_folder, "tuned_models",
                                     f"tuned_{selected_model_name.replace(' ', '_').lower()}.joblib")
        metadata_filename = os.path.join(output_folder, "tuned_models",
                                        f"tuned_{selected_model_name.replace(' ', '_').lower()}_metadata.joblib")
        scaler_filename = os.path.join(output_folder, "tuned_models", "final_scaler.joblib")

        validate_file_exists(model_filename, "model file")
        validate_file_exists(scaler_filename, "scaler file")

        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)

        # Load metadata
        use_scaled = isinstance(model, SVC)
        feature_order = Config.FEATURE_NAMES

        if os.path.exists(metadata_filename):
            metadata = joblib.load(metadata_filename)
            use_scaled = metadata.get("use_scaled", use_scaled)
            feature_order = metadata.get("feature_order", Config.FEATURE_NAMES)

        # Build raster paths
        raster_paths = [os.path.join(raster_folder, f"{name}.tif") for name in feature_order]
        for p in raster_paths:
            validate_file_exists(p, "raster file")

        # Check consistency (open briefly to validate and read profile)
        from utils import check_raster_consistency
        src_files = check_raster_consistency(raster_paths)

        # Copy profile then close the temporary opened datasets so workers open files themselves
        profile = src_files[0].profile.copy()
        profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=Config.NODATA_VALUE)
        close_srcs(src_files)
        src_files = []

        output_map_path = os.path.join(output_folder,
                                      f"Landslide_Susceptibility_Score_{selected_model_name.replace(' ', '_')}.tif")
        logger.info(f"\n--- Generating output map: {output_map_path} ---")
        logger.info(f"  - Raster dimensions: {profile['width']} x {profile['height']} pixels")

        # Adaptive block size
        if block_size <= 0:
            block_size = Config.DEFAULT_BLOCK_SIZE
        bs = min(block_size, profile['width'], profile['height'])

        # Memory check
        max_pixels_per_block = Config.MAX_PIXELS_PER_BLOCK
        max_block_size = int(np.sqrt(max_pixels_per_block))
        bs = min(bs, max_block_size)

        # Create block list
        blocks = []
        for j in range(0, profile['height'], bs):
            for i in range(0, profile['width'], bs):
                win_width = min(bs, profile['width'] - i)
                win_height = min(bs, profile['height'] - j)
                window = Window(i, j, win_width, win_height)
                blocks.append({
                    'window': window,
                    'height': win_height,
                    'width': win_width
                })

        total_blocks = len(blocks)
        logger.info(f"  - Total blocks to process: {total_blocks} (block size {bs})")
        logger.info(f"  - Using {Config.PARALLEL_WORKERS} parallel workers")

        # Load model feature list once and pass to workers
        feature_list_path = os.path.join(output_folder, "tuned_models", "feature_list.joblib")
        model_feature_list = None
        if os.path.exists(feature_list_path):
            try:
                model_feature_list = joblib.load(feature_list_path)
            except Exception:
                logger.warning("Could not load model feature list for raster prediction; proceeding with raster feature names")

        # Process blocks in parallel
        with rasterio.open(output_map_path, 'w', **profile) as dst:
            completed_blocks = 0

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=Config.PARALLEL_WORKERS) as executor:
                # Submit all blocks
                # Pass raster file paths (strings) to each worker so they open files locally
                future_to_block = {
                    executor.submit(
                        process_raster_block,
                        block,
                        raster_paths,
                        model,
                        scaler,
                        use_scaled,
                        feature_order,
                        Config.NODATA_VALUE,
                        model_feature_list
                    ): block for block in blocks
                }

                # Process completed blocks
                for future in as_completed(future_to_block):
                    try:
                        out_chunk, window = future.result()
                        dst.write(out_chunk.astype(rasterio.float32), 1, window=window)

                        completed_blocks += 1
                        progress_callback(completed_blocks / total_blocks)

                        if completed_blocks % 100 == 0:
                            logger.info(f"  - Processed {completed_blocks}/{total_blocks} blocks "
                                      f"({completed_blocks/total_blocks*100:.1f}%)")

                    except Exception as e:
                        logger.warning(f"Error processing block: {e}")

        # Read predictions for monitoring
        with rasterio.open(output_map_path, 'r') as src:
            predictions = src.read(1)

        # Monitor prediction distribution
        monitor_prediction_distribution(predictions, output_folder, selected_model_name)

        total_time = (time.time() - start_time) / 60
        logger.info(f"\n--- Map Generation Complete! ---")
        logger.info(f"  - Output saved to: {output_map_path}")
        logger.info(f"  - Processing time: {total_time:.2f} minutes")

        return True

    except Exception as e:
        log_error_details(logging.getLogger('LandslideAnalysis'), "MAP GENERATION", e)
        return False

    finally:
        close_srcs(src_files)