# =============================================================================
# File: config.py
# Author: Pramodh Gamage (with AI assistance)
# Version: 2.0
# Description: Configuration class and constants for landslide analysis engine
# =============================================================================

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================
class Config:
    """Enhanced configuration class with validation."""

    # Random seed for reproducibility
    RANDOM_STATE = 42

    # Feature names in the correct order
    FEATURE_NAMES = ['Elevation', 'TWI', 'STI', 'SPI', 'Slope',
                     'Profile_curvature', 'Plan_curvature', 'EWC', 'Aspect']

    # Memory and performance settings
    MAX_MEMORY_PERCENT = 80  # Maximum memory usage percentage
    BATCH_SIZE = 100000  # Number of rows to process at once for large datasets
    CACHE_DIR = '.cache'  # Directory for caching intermediate results

    # Raster processing optimization
    AUTO_BLOCK_SIZE = True  # Automatically optimize block size based on memory
    MIN_BLOCK_SIZE = 128
    MAX_BLOCK_SIZE = 2048
    TARGET_BLOCK_MEMORY_MB = 100  # Target memory usage per block in MB

    # Raster processing settings
    DEFAULT_BLOCK_SIZE = 512
    MAX_PIXELS_PER_BLOCK = 2_700_000  # ~100MB per block
    NODATA_VALUE = -9999.0
    PARALLEL_WORKERS = 4  # Number of parallel workers for raster processing

    # Require coordinates (fail if not present)
    REQUIRE_COORDINATES = True

    # Model training settings
    TEST_SIZE = 0.2
    CV_FOLDS = 5  # Increased from 3 for better validation
    BAYES_ITERATIONS = 32

    # Class imbalance handling
    BALANCE_STRATEGY = 'smote'  # Options: 'smote', 'undersample', 'none'
    SMOTE_K_NEIGHBORS = 5

    # Spatial validation
    USE_SPATIAL_CV = True
    SPATIAL_CLUSTERS = 5

    # Uncertainty quantification
    BOOTSTRAP_SAMPLES = 100
    CONFIDENCE_LEVEL = 0.95

    # SHAP settings
    SHAP_SAMPLE_SIZE = 1000
    SHAP_BACKGROUND_SIZE = 200

    # Model ensemble
    USE_STACKING = False  # Set to True to use ensemble stacking

    # Threshold optimization
    THRESHOLD_METRIC = 'f1'  # Options: 'f1', 'youden'

    # Plotting settings
    FIGURE_DPI = 300
    PLOT_STYLE = "whitegrid"

    # File extensions
    MODEL_EXTENSION = ".joblib"
    RASTER_EXTENSION = ".tif"
    CSV_EXTENSION = ".csv"
    PNG_EXTENSION = ".png"

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration parameters."""
        try:
            assert 0 < cls.TEST_SIZE < 1, "TEST_SIZE must be between 0 and 1"
            assert cls.CV_FOLDS >= 2, "CV_FOLDS must be at least 2"
            assert cls.RANDOM_STATE >= 0, "RANDOM_STATE must be non-negative"
            assert len(cls.FEATURE_NAMES) > 0, "FEATURE_NAMES cannot be empty"
            assert cls.BALANCE_STRATEGY in ['smote', 'undersample', 'none'], \
                "BALANCE_STRATEGY must be 'smote', 'undersample', or 'none'"
            assert cls.THRESHOLD_METRIC in ['f1', 'youden'], \
                "THRESHOLD_METRIC must be 'f1' or 'youden'"
            assert cls.PARALLEL_WORKERS > 0, "PARALLEL_WORKERS must be positive"
            return True
        except AssertionError as e:
            import logging
            logging.error(f"Configuration validation failed: {e}")
            return False

# Legacy constants for backward compatibility
RANDOM_STATE = Config.RANDOM_STATE
FEATURE_NAMES = Config.FEATURE_NAMES
DEFAULT_BLOCK_SIZE = Config.DEFAULT_BLOCK_SIZE