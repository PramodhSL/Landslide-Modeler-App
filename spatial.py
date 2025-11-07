# =============================================================================
# File: spatial.py
# Author: Pramodh Gamage (with AI assistance)
# Version: 2.0
# Description: Spatial analysis and frequency ratio functions
# =============================================================================

import os
import logging
from typing import Callable, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans

from config import Config
from utils import format_interval_label, validate_input_data, log_error_details

# =============================================================================
# SPATIAL ANALYSIS FUNCTIONS
# =============================================================================

def create_spatial_groups(df: pd.DataFrame, n_groups: int = 5) -> np.ndarray:
    """
    Create spatial groups for spatial cross-validation.
    Uses k-means clustering on feature space as proxy for spatial proximity.

    Note: If actual coordinates are available, use them instead of feature-based clustering.
    """
    # If actual coordinates are available, use them for spatial grouping
    coord_cols = None

    # Accept multiple common names: X_coord/Y_coord, x/y, X/Y
    # Use case-sensitive column matching
    coord_pairs = [
        ['X_coord', 'Y_coord'],
        ['x', 'y'],
        ['X', 'Y']
    ]

    # Find first matching coordinate pair
    for x_col, y_col in coord_pairs:
        if x_col in df.columns and y_col in df.columns:
            coord_cols = [x_col, y_col]
            break

    if coord_cols is not None:
        spatial_features = df[coord_cols].values
    else:
        # No coordinate columns found: fall back to feature-based clustering
        logging.getLogger('LandslideAnalysis').warning(
            "Coordinate columns not found; falling back to feature-based spatial grouping"
        )
        # Use numeric features from df (prefer Config.FEATURE_NAMES when present)
        fallback_cols = [c for c in Config.FEATURE_NAMES if c in df.columns]
        if fallback_cols:
            spatial_features = df[fallback_cols].values
        else:
            # Last resort: use all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No suitable columns for spatial grouping found")
            spatial_features = df[numeric_cols].values

    kmeans = KMeans(n_clusters=n_groups, random_state=Config.RANDOM_STATE, n_init=10)
    groups = kmeans.fit_predict(spatial_features)

    return groups

def get_cv_splitter(X: pd.DataFrame, use_spatial: bool = True):
    """Get appropriate cross-validation splitter."""
    if use_spatial and Config.USE_SPATIAL_CV:
        spatial_groups = create_spatial_groups(X, n_groups=Config.SPATIAL_CLUSTERS)
        return GroupKFold(n_splits=Config.CV_FOLDS), spatial_groups
    else:
        return Config.CV_FOLDS, None

# =============================================================================
# FREQUENCY RATIO ANALYSIS
# =============================================================================

def classify_aspect(df: pd.DataFrame) -> pd.DataFrame:
    """Classifies the 'Aspect' column into 9 geographical directions."""
    if 'Aspect' not in df.columns:
        return df

    df['Aspect_class'] = 'Flat'
    conditions = [
        (df['Aspect'] > 337.5) | ((df['Aspect'] >= 0) & (df['Aspect'] <= 22.5)),
        (df['Aspect'] > 22.5) & (df['Aspect'] <= 67.5),
        (df['Aspect'] > 67.5) & (df['Aspect'] <= 112.5),
        (df['Aspect'] > 112.5) & (df['Aspect'] <= 157.5),
        (df['Aspect'] > 157.5) & (df['Aspect'] <= 202.5),
        (df['Aspect'] > 202.5) & (df['Aspect'] <= 247.5),
        (df['Aspect'] > 247.5) & (df['Aspect'] <= 292.5),
        (df['Aspect'] > 292.5) & (df['Aspect'] <= 337.5)
    ]
    choices = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    df['Aspect_class'] = np.select(conditions, choices, default='Flat')
    return df

def calculate_fr(data: pd.DataFrame, factor_column: str,
                is_categorical: bool = False, bins: int = 10) -> pd.DataFrame:
    """Calculates Frequency Ratio for a given factor."""
    class_col_name = factor_column if is_categorical else factor_column + '_class'

    if not is_categorical:
        try:
            unique_values = data[factor_column].nunique()
            if unique_values < bins:
                data[class_col_name] = pd.cut(data[factor_column], bins=unique_values, duplicates='drop')
            else:
                data[class_col_name] = pd.qcut(data[factor_column], q=bins, duplicates='drop')
        except Exception as e:
            logging.warning(f"Could not create bins for {factor_column}: {e}")
            return pd.DataFrame()

    total_landslides = data[data['Type'] == 1].shape[0]
    total_points = data.shape[0]

    if total_landslides == 0 or total_points == 0:
        logging.warning("No landslides or no data points for FR calculation")
        return pd.DataFrame()

    fr_list = []
    for class_label in data[class_col_name].unique():
        class_data = data[data[class_col_name] == class_label]
        landslides_in_class = class_data[class_data['Type'] == 1].shape[0]
        points_in_class = class_data.shape[0]

        if points_in_class > 0:
            fr = (landslides_in_class / total_landslides) / (points_in_class / total_points)
        else:
            fr = 0.0

        fr_list.append((class_label, fr))

    return pd.DataFrame(fr_list, columns=[class_col_name, 'FR'])

def plot_fr(factor_name: str, fr_table: pd.DataFrame, output_dir: str) -> None:
    """Generates and saves a bar plot for a factor's FR values."""
    is_aspect = (factor_name == 'Aspect')
    class_col = 'Aspect_class' if is_aspect else factor_name + '_class'
    order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'Flat'] if is_aspect else None

    if class_col not in fr_table.columns:
        return

    fr_table['formatted_label'] = fr_table[class_col].apply(format_interval_label)

    if not is_aspect:
        try:
            fr_table.sort_values(by=class_col, inplace=True)
        except Exception:
            pass

    xlabel = f"{factor_name} Classes" + ("" if is_aspect else " (Value Ranges)")

    plt.figure(figsize=(12, 7))
    sns.set_style(Config.PLOT_STYLE)
    barplot = sns.barplot(x='formatted_label', y='FR', data=fr_table,
                         palette='viridis', order=order)
    plt.axhline(y=1, color='r', linestyle='--', linewidth=2,
               label='FR = 1 (Average Susceptibility)')

    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9),
                        textcoords='offset points')

    plt.xlabel(xlabel)
    plt.ylabel("Frequency Ratio (FR)")
    plt.title(f"Frequency Ratio Analysis for {factor_name}", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout(pad=2.0)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{factor_name}_FR_plot.png"),
               dpi=Config.FIGURE_DPI)
    plt.close()

def perform_fr_analysis(input_csv: str, output_folder: str,
                       log_callback: Callable, progress_callback: Callable) -> bool:
    """Performs comprehensive Frequency Ratio analysis."""
    try:
        logger = logging.getLogger('LandslideAnalysis')
        logger.info("\n" + "="*50)
        logger.info("--- STAGE 1: Performing Frequency Ratio Analysis ---")
        logger.info("="*50)

        output_fr_folder = os.path.join(output_folder, "fr_analysis")
        os.makedirs(output_fr_folder, exist_ok=True)

        df = pd.read_csv(input_csv)

        if not validate_input_data(df, logger):
            return False

        df.dropna(inplace=True)
        df = classify_aspect(df)

        continuous_factors = [f for f in Config.FEATURE_NAMES if f != 'Aspect']
        fr_dict = {}
        total_factors = len(Config.FEATURE_NAMES)

        logger.info("  - Calculating FR for continuous factors...")
        for i, factor in enumerate(continuous_factors):
            if factor in df.columns:
                fr_df = calculate_fr(df.copy(), factor, bins=10)
                fr_dict[factor] = fr_df
                logger.info(f"    - Completed: {factor}")
            progress_callback((i + 1) / total_factors)

        logger.info("  - Calculating FR for categorical factor: Aspect...")
        if 'Aspect_class' in df.columns:
            fr_dict['Aspect'] = calculate_fr(df, 'Aspect_class', is_categorical=True)
        progress_callback(1.0)

        summary_list = []
        for factor, fr_table in fr_dict.items():
            if not fr_table.empty:
                summary_list.append({'Factor': factor, 'Max_FR': fr_table['FR'].max()})
                plot_fr(factor, fr_table, output_fr_folder)

        summary_df = pd.DataFrame(summary_list).sort_values(by='Max_FR', ascending=False)
        summary_df.to_csv(os.path.join(output_fr_folder, "fr_importance_ranking.csv"), index=False)

        logger.info("\n--- FR Analysis Complete ---")
        logger.info(f"Importance Ranking:\n{summary_df.to_string()}")
        logger.info(f"Plots and report saved to '{output_fr_folder}'")
        return True

    except Exception as e:
        log_error_details(logging.getLogger('LandslideAnalysis'), "FR ANALYSIS", e)
        return False