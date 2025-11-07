# =============================================================================
# File: preprocessing.py
# Author: Pramodh Gamage (with AI assistance)
# Version: 2.0
# Description: Preprocessing functions for class imbalance, threshold optimization, feature engineering
# =============================================================================

import logging
from typing import Tuple, List

import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

from config import Config
from utils import IMBALANCE_AVAILABLE, SMOTE, RandomUnderSampler

# =============================================================================
# CLASS IMBALANCE HANDLING
# =============================================================================

def handle_class_imbalance(X_train: pd.DataFrame, y_train: pd.Series,
                          strategy: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using various strategies.

    Args:
        X_train: Training features
        y_train: Training labels
        strategy: 'smote', 'undersample', or 'none'

    Returns:
        Resampled X_train and y_train
    """
    if strategy == 'none':
        return X_train, y_train

    if not IMBALANCE_AVAILABLE:
        logging.warning("imbalanced-learn not available. Skipping resampling.")
        return X_train, y_train

    try:
        if strategy == 'smote':
            # Calculate appropriate k_neighbors
            min_class_size = min(np.bincount(y_train))
            k_neighbors = min(Config.SMOTE_K_NEIGHBORS, min_class_size - 1)

            if k_neighbors < 1:
                logging.warning("Not enough samples for SMOTE. Skipping resampling.")
                return X_train, y_train

            smote = SMOTE(random_state=Config.RANDOM_STATE, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            logging.info(f"SMOTE applied: {len(y_train)} -> {len(y_resampled)} samples")

        elif strategy == 'undersample':
            undersampler = RandomUnderSampler(random_state=Config.RANDOM_STATE)
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
            logging.info(f"Undersampling applied: {len(y_train)} -> {len(y_resampled)} samples")

        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)

    except Exception as e:
        logging.warning(f"Resampling failed: {e}. Using original data.")
        return X_train, y_train

# =============================================================================
# THRESHOLD OPTIMIZATION
# =============================================================================

def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal probability threshold based on metric.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: 'f1' or 'youden'

    Returns:
        optimal_threshold, optimal_score
    """
    if metric == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # Avoid division by zero
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_scores = f1_scores[:-1]  # precision_recall_curve returns n+1 values
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx], f1_scores[optimal_idx]

    elif metric == 'youden':
        # Youden's J statistic (sensitivity + specificity - 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx], j_scores[optimal_idx]

    else:
        return 0.5, 0.0

# =============================================================================
# FEATURE ENGINEERING AND SELECTION
# =============================================================================

def generate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate interaction features between important numerical columns."""
    result = df.copy()

    # Generate meaningful combinations
    result['Slope_Elevation'] = df['Slope'] * df['Elevation']
    result['TWI_Slope'] = df['TWI'] * df['Slope']
    result['Curvature_Combined'] = df['Profile_curvature'] * df['Plan_curvature']

    # Generate ratio features
    result['SPI_TWI_Ratio'] = df['SPI'] / (df['TWI'] + 1e-6)  # Avoid division by zero

    return result

def select_features_by_correlation(df: pd.DataFrame, target: pd.Series,
                                 threshold: float = 0.7) -> List[str]:
    """Select features by removing highly correlated ones while keeping most important."""
    correlation = df.corr().abs()
    target_corr = df.apply(lambda x: abs(x.corr(target)))

    selected_features = []
    candidate_features = target_corr.sort_values(ascending=False).index.tolist()

    for feature in candidate_features:
        if feature not in selected_features:
            correlated_features = correlation.index[correlation[feature] > threshold].tolist()
            correlated_features = [f for f in correlated_features if f in candidate_features]

            # Keep the feature most correlated with target
            best_feature = max(correlated_features,
                             key=lambda x: target_corr[x])
            if best_feature not in selected_features:
                selected_features.append(best_feature)

    return selected_features

def rank_features_by_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rank features by importance using multiple methods."""
    from sklearn.ensemble import RandomForestClassifier

    rankings = {}

    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
    rf.fit(X, y)
    rankings['rf_importance'] = rf.feature_importances_

    # Correlation with target
    rankings['correlation'] = abs(X.apply(lambda x: x.corr(y)))

    # Combine rankings
    ranking_df = pd.DataFrame(rankings, index=X.columns)
    ranking_df['mean_rank'] = ranking_df.mean(axis=1)

    return ranking_df.sort_values('mean_rank', ascending=False)