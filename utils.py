# =============================================================================
# File: utils.py
# Author: Pramodh Gamage (with AI assistance)
# Version: 2.0
# Description: Helper functions for landslide analysis engine
# =============================================================================

import os
import time
import warnings
import traceback
import logging
import psutil
from contextlib import contextmanager
from typing import Callable, Optional, Tuple, List, Dict, Any, Union

import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioIOError
from rasterio.crs import CRS

from sklearn.preprocessing import StandardScaler

from config import Config

# Optional dependencies
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCE_AVAILABLE = True
except ImportError:
    IMBALANCE_AVAILABLE = False
    warnings.warn("imbalanced-learn not available. Install with: pip install imbalanced-learn")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(output_folder: str) -> logging.Logger:
    """Setup comprehensive logging with file and console handlers."""
    os.makedirs(output_folder, exist_ok=True)
    log_file = os.path.join(output_folder, 'analysis.log')

    # Create logger
    logger = logging.getLogger('LandslideAnalysis')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler (wrap stdout with UTF-8 encoding to avoid Windows cp1252 errors)
    try:
        import sys, io
        console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        console_handler = logging.StreamHandler(console_stream)
    except Exception:
        console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
            'vms': memory_info.vms / (1024 * 1024),  # VMS in MB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / (1024 * 1024)  # Available in MB
        }
    except Exception as e:
        logging.warning(f"Could not get memory stats: {e}")
        return {}


@contextmanager
def temp_random_seed(seed: int):
    """Temporarily set numpy random seed inside a context manager.

    Restores the previous numpy RNG state on exit.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def log_memory_usage(logger: logging.Logger, stage: str) -> None:
    """Log current memory usage."""
    mem_stats = get_memory_usage()
    if mem_stats:
        logger.info(f"Memory usage at {stage}:")
        logger.info(f"  - RSS: {mem_stats['rss']:.1f} MB")
        logger.info(f"  - Memory Usage: {mem_stats['percent']:.1f}%")
        logger.info(f"  - Available: {mem_stats['available']:.1f} MB")

        # Warning if memory usage is high
        if mem_stats['percent'] > 80:
            logger.warning("High memory usage detected!")

def format_interval_label(interval):
    """Formats a pandas Interval object for cleaner plot labels."""
    return f"({interval.left:.2f} - {interval.right:.2f}]" if isinstance(interval, pd.Interval) else str(interval)

def validate_raster_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a raster file for corruption and readability.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with rasterio.open(file_path) as src:
            # Check if file can be read
            window = Window(0, 0, min(512, src.width), min(512, src.height))
            data = src.read(1, window=window)

            # Check for basic metadata
            if not src.crs:
                return False, "Missing coordinate reference system (CRS)"
            if not src.transform:
                return False, "Missing spatial transform"
            if src.dtypes[0] not in ['float32', 'float64', 'int16', 'int32', 'uint8', 'uint16']:
                return False, f"Unsupported data type: {src.dtypes[0]}"

            # Check for valid bounds
            if not all(isinstance(x, (int, float)) for x in src.bounds):
                return False, "Invalid bounds"

            return True, None
    except RasterioIOError as e:
        return False, f"IO Error: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_crs_consistency(raster_paths: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate that all rasters have the same and valid CRS.

    Returns:
        Tuple of (is_consistent, error_message)
    """
    if not raster_paths:
        return False, "No raster paths provided"

    try:
        reference_crs = None
        for path in raster_paths:
            with rasterio.open(path) as src:
                if not src.crs:
                    return False, f"Missing CRS in {path}"

                if reference_crs is None:
                    reference_crs = src.crs
                elif src.crs != reference_crs:
                    return False, f"CRS mismatch in {path}: {src.crs} != {reference_crs}"

        return True, None
    except Exception as e:
        return False, f"CRS validation error: {str(e)}"

def validate_file_exists(file_path: str, file_type: str = "file") -> bool:
    """Validates that a file exists and raises descriptive error if not."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERROR: {file_type.title()} not found: {file_path}")
    return True

def check_raster_consistency(paths: List[str]) -> List[rasterio.DatasetReader]:
    """Check raster consistency: same CRS, transform, width/height."""
    srcs = [rasterio.open(p) for p in paths]
    try:
        base = srcs[0]
        base_crs = base.crs
        base_transform = base.transform
        base_width, base_height = base.width, base.height

        for i, src in enumerate(srcs[1:], start=1):
            if src.crs != base_crs:
                raise ValueError(f"ERROR: CRS mismatch: {paths[i]} has {src.crs}, expected {base_crs}")
            if src.transform != base_transform:
                raise ValueError(f"ERROR: Transform mismatch: {paths[i]}")
            if (src.width, src.height) != (base_width, base_height):
                raise ValueError(f"ERROR: Shape mismatch: {paths[i]} is ({src.width},{src.height}), expected ({base_width},{base_height})")
        return srcs
    except Exception:
        for s in srcs:
            try:
                s.close()
            except Exception as close_error:
                logging.warning(f"Error closing raster source: {close_error}")
        raise

def safe_read_window(src: rasterio.DatasetReader, window: Window) -> np.ndarray:
    """Read window from rasterio dataset robustly."""
    arr = src.read(1, window=window, masked=False)
    nod = src.nodata

    if nod is not None:
        arr = arr.astype(np.float32)
        arr[arr == nod] = np.nan
    else:
        arr = arr.astype(np.float32)

    arr[~np.isfinite(arr)] = np.nan
    return arr

def close_srcs(srcs: List[rasterio.DatasetReader]) -> None:
    """Close raster sources safely."""
    for s in srcs:
        try:
            s.close()
        except Exception as e:
            logging.warning(f"Error closing raster source: {e}")

def log_error_details(logger: logging.Logger, stage_name: str, error: Exception) -> None:
    """Standardized error logging for all stages."""
    logger.error(f"\n--- ERROR IN {stage_name.upper()} ---")
    logger.error(f"Error details: {str(error)}")
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Traceback: {traceback.format_exc()}")

def identify_coordinate_columns(df: pd.DataFrame) -> List[str]:
    """Identify coordinate columns in the dataframe."""
    coord_pairs = [
        ['X_coord', 'Y_coord'],
        ['x', 'y'],
        ['X', 'Y']
    ]
    for x_col, y_col in coord_pairs:
        if x_col in df.columns and y_col in df.columns:
            return [x_col, y_col]
    return []

def validate_input_data(df: pd.DataFrame, logger: logging.Logger = None) -> bool:
    """Comprehensive validation of input data quality and structure."""
    if logger is None:
        logger = logging.getLogger('LandslideAnalysis')

    try:
        # Check if dataframe is empty
        if df.empty:
            raise ValueError("Input dataset is empty")

        # Check required columns
        required_columns = ['Type'] + Config.FEATURE_NAMES
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check data types
        if not pd.api.types.is_numeric_dtype(df['Type']):
            raise ValueError("'Type' column must be numeric")

        # Check for valid Type values
        unique_types = df['Type'].unique()
        invalid_types = [t for t in unique_types if t not in [0, 1]]
        if invalid_types:
            raise ValueError(f"Invalid values in 'Type' column: {invalid_types}. Must be 0 or 1.")

        # Check for sufficient data
        if len(df) < 10:
            raise ValueError(f"Insufficient data: only {len(df)} rows. Need at least 10.")

        # Check for landslides
        landslide_count = (df['Type'] == 1).sum()
        if landslide_count == 0:
            raise ValueError("No landslide points found (Type=1)")

        # Check for non-landslides
        non_landslide_count = (df['Type'] == 0).sum()
        if non_landslide_count == 0:
            raise ValueError("No non-landslide points found (Type=0)")

        # Check class balance
        imbalance_ratio = max(landslide_count, non_landslide_count) / min(landslide_count, non_landslide_count)
        if imbalance_ratio > 10:
            logger.warning(f"Severe class imbalance detected: ratio = {imbalance_ratio:.1f}:1")

        # Check for excessive missing values
        missing_percentage = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_percentage[missing_percentage > 50]
        if not high_missing.empty:
            logger.warning(f"High missing values in columns: {high_missing.to_dict()}")

        # Check for infinite values
        inf_columns = []
        for col in Config.FEATURE_NAMES:
            if col in df.columns and np.isinf(df[col]).any():
                inf_columns.append(col)
        if inf_columns:
            logger.warning(f"Infinite values found in columns: {inf_columns}")

        return True
    except Exception as e:
        logger.error(f"Input data validation failed: {e}")
        logger.debug(traceback.format_exc())
        return False