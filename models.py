# =============================================================================
# File: models.py
# Author: Pramodh Gamage (with AI assistance)
# Version: 2.0
# Description: Model tuning, analysis, diagnostics, and SHAP functions
# =============================================================================

import os
import logging
import traceback
from typing import Callable, Tuple, List, Dict, Any, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                            confusion_matrix, precision_recall_curve,
                            roc_curve, make_scorer)
from sklearn.calibration import calibration_curve

import xgboost as xgb

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from config import Config
from utils import SHAP_AVAILABLE, temp_random_seed, validate_file_exists, log_error_details, identify_coordinate_columns, validate_input_data
from spatial import get_cv_splitter
from preprocessing import handle_class_imbalance, find_optimal_threshold

# =============================================================================
# ENSEMBLE STACKING
# =============================================================================

def create_stacked_model() -> StackingClassifier:
    """Create stacked ensemble of all models."""
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=Config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )),
        ('xgb', xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=Config.RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        )),
        ('svm', SVC(
            C=10,
            gamma=0.1,
            probability=True,
            random_state=Config.RANDOM_STATE
        ))
    ]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=xgb.XGBClassifier(
            n_estimators=100,
            random_state=Config.RANDOM_STATE,
                eval_metric='logloss',
                verbosity=0
        ),
        cv=Config.CV_FOLDS,
        n_jobs=-1
    )

    return stack

# =============================================================================
# MODEL TUNING
# =============================================================================

def perform_model_tuning(input_csv: str, output_folder: str,
                        log_callback: Callable, progress_callback: Callable) -> bool:
    """Performs advanced machine learning model tuning with all enhancements."""
    try:
        logger = logging.getLogger('LandslideAnalysis')
        logger.info("\n" + "="*50)
        logger.info("--- STAGE 2: Enhanced Model Training and Evaluation ---")
        logger.info("="*50)

        output_model_folder = os.path.join(output_folder, "tuned_models")
        os.makedirs(output_model_folder, exist_ok=True)

        # Load data in batches if necessary
        df_iterator = pd.read_csv(input_csv, chunksize=Config.BATCH_SIZE) if os.path.getsize(input_csv) > 1e8 else [pd.read_csv(input_csv)]

        all_data = []
        for chunk in df_iterator:
            # Process each batch
            all_data.append(chunk)

        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(df)} samples for processing")

        # If coordinates are required, ensure the CSV contains them before further processing
        if Config.REQUIRE_COORDINATES:
            if not identify_coordinate_columns(df):
                logger.error("Config.REQUIRE_COORDINATES is True but X/Y coordinate columns are missing in the input CSV. Aborting.")
                return False

        if not validate_input_data(df, logger):
            return False

        df.dropna(inplace=True)

        # Identify coordinate columns and save them for later use
        coord_columns = identify_coordinate_columns(df)

        # Save the complete feature list for model use
        # For modeling, use only raster-derived features (do not include coordinates)
        training_feature_list = Config.FEATURE_NAMES
        joblib.dump(training_feature_list, os.path.join(output_model_folder, "feature_list.joblib"))
        logger.info(f"  - Using training features: {training_feature_list}")

        # Features used for model training (exclude coordinates)
        X = df[training_feature_list]
        y = df['Type']

        # Preserve coordinate columns (if present) so spatial grouping can use them
        coords_df = df[coord_columns] if coord_columns else None

        # Train/test split with stratification. If coordinates exist, include them so
        # we can compute spatial groups on the original training samples (before resampling).
        if coords_df is not None:
            X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
                X, y, coords_df, test_size=Config.TEST_SIZE,
                random_state=Config.RANDOM_STATE,
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.TEST_SIZE,
                random_state=Config.RANDOM_STATE,
                stratify=y
            )

        # Handle class imbalance
        logger.info(f"\n  - Applying class imbalance handling: {Config.BALANCE_STRATEGY}")
        X_train_balanced, y_train_balanced = handle_class_imbalance(
            X_train, y_train, strategy=Config.BALANCE_STRATEGY
        )
          # Defensive cleaning: coerce types, drop NaNs and invalid labels
        # Ensure X is DataFrame and y is Series
        if not isinstance(X_train_balanced, pd.DataFrame):
            try:
                X_train_balanced = pd.DataFrame(X_train_balanced, columns=X.columns)
            except Exception:
                X_train_balanced = pd.DataFrame(X_train_balanced)

        if not isinstance(y_train_balanced, pd.Series):
            y_train_balanced = pd.Series(y_train_balanced)

        # Drop NaNs in y
        if y_train_balanced.isnull().any():
            logger.warning("NaN values found in y_train after resampling — dropping corresponding rows")
            mask = ~y_train_balanced.isnull()
            X_train_balanced = X_train_balanced.loc[mask].reset_index(drop=True)
            y_train_balanced = y_train_balanced.loc[mask].reset_index(drop=True)

        # Keep only valid labels (0/1)
        valid_mask = y_train_balanced.isin([0, 1])
        if not valid_mask.all():
            logger.warning("Invalid labels found in y_train after resampling — dropping them")
            X_train_balanced = X_train_balanced.loc[valid_mask].reset_index(drop=True)
            y_train_balanced = y_train_balanced.loc[valid_mask].reset_index(drop=True)

        # Coerce y to integer labels (0/1)
        try:
            y_train_balanced = y_train_balanced.astype(int)
        except Exception:
            y_train_balanced = y_train_balanced.map(lambda v: 1 if v == 1 else 0).astype(int)

        # If resampling removed too many samples, fallback to original training set
        if len(y_train_balanced) < 10 or y_train_balanced.nunique() < 2:
            logger.warning("Resampled training set too small or single-class after cleaning — falling back to original training data")
            X_train_balanced = X_train.copy().reset_index(drop=True)
            y_train_balanced = y_train.copy().reset_index(drop=True)

        # Fit scaler on balanced training data
        with temp_random_seed(Config.RANDOM_STATE):
            scaler = StandardScaler().fit(X_train_balanced)
        X_train_scaled = scaler.transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, os.path.join(output_model_folder, "final_scaler.joblib"))
        logger.info("  - Data scaler has been saved.")

        # Calculate scale_pos_weight for XGBoost
        pos = np.sum(y_train_balanced == 1)
        neg = np.sum(y_train_balanced == 0)
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        # Compute spatial groups for cross-validation on the original training set
        # (use coordinates when available). Do this before resampling so groups
        # reflect true spatial structure.
        if coords_df is not None:
            # Reconstruct a training dataframe that includes the coordinate columns
            try:
                X_train_for_cv = pd.concat([X_train.reset_index(drop=True),
                                            coords_train.reset_index(drop=True)], axis=1)
            except Exception:
                # Fallback: if concat fails, just use X_train
                X_train_for_cv = X_train
        else:
            X_train_for_cv = X_train

        cv_splitter, spatial_groups = get_cv_splitter(X_train_for_cv)

        if spatial_groups is not None:
            logger.info(f"  - Using spatial cross-validation with {Config.SPATIAL_CLUSTERS} clusters")

        # Define models and search spaces
        models_and_spaces = {
            "Random Forest": {
                "model": RandomForestClassifier(
                    random_state=Config.RANDOM_STATE,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                "space": {
                    'n_estimators': Integer(100, 500),
                    'max_depth': Integer(10, 30),
                    'max_features': Categorical(['sqrt', 'log2']),
                    'min_samples_split': Integer(2, 10),
                    'min_samples_leaf': Integer(1, 4)
                },
                "use_scaled": False
            },
            "XGBoost": {
                "model": xgb.XGBClassifier(
                    random_state=Config.RANDOM_STATE,
                    eval_metric='logloss',
                    verbosity=0,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight
                ),
                "space": {
                    'n_estimators': Integer(100, 500),
                    'learning_rate': Real(0.01, 0.2, 'log-uniform'),
                    'max_depth': Integer(3, 8),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0)
                },
                "use_scaled": False
            },
            "Support Vector Machine": {
                "model": SVC(
                    random_state=Config.RANDOM_STATE,
                    probability=True
                ),
                "space": {
                    'C': Real(1e-1, 1e+2, 'log-uniform'),
                    'gamma': Real(1e-3, 1e+1, 'log-uniform'),
                    'kernel': Categorical(['rbf'])
                },
                "use_scaled": True
            }
        }

        # Add stacking ensemble if enabled
        if Config.USE_STACKING:
            models_and_spaces["Stacking Ensemble"] = {
                "model": create_stacked_model(),
                "space": {},  # No hyperparameter tuning for ensemble
                "use_scaled": False
            }

        results_list = []
        total_iterations = sum(
            Config.BAYES_ITERATIONS if mp["space"] else 1
            for mp in models_and_spaces.values()
        )
        completed_iterations = 0

        for name, mp in models_and_spaces.items():
            logger.info(f"\n--- Tuning {name} ---")

            # Select appropriate training data for final model fitting (may be resampled/scaled)
            X_train_model = X_train_scaled if mp["use_scaled"] else X_train_balanced

            # For hyperparameter tuning, use the original (non-resampled) training set so
            # cross-validation reflects real data distribution. Use scaling if required.
            if mp["use_scaled"]:
                # Fit a temporary scaler on original training data (X_train)
                tmp_scaler = StandardScaler().fit(X_train)
                X_train_for_tuning = tmp_scaler.transform(X_train)
            else:
                X_train_for_tuning = X_train

            if mp["space"]:  # Has hyperparameters to tune
                # Prepare CV parameters and validate folds
                use_cv = cv_splitter

                # Validate folds using original training labels and groups
                if spatial_groups is not None:
                    try:
                        splits = list(cv_splitter.split(X_train_for_tuning, y_train, spatial_groups))
                    except Exception:
                        splits = list(cv_splitter.split(X_train_for_tuning, y_train)) if hasattr(cv_splitter, 'split') else []

                    # Check each fold contains both classes
                    invalid_fold = False
                    for tr_idx, te_idx in splits:
                        if y_train.iloc[tr_idx].nunique() < 2 or y_train.iloc[te_idx].nunique() < 2:
                            invalid_fold = True
                            break
                    if invalid_fold:
                        from sklearn.model_selection import StratifiedKFold
                        logger.warning("Spatial CV produced single-class fold; falling back to StratifiedKFold")
                        use_cv = StratifiedKFold(n_splits=Config.CV_FOLDS)

                bayes_search = BayesSearchCV(
                    mp["model"],
                    mp["space"],
                    n_iter=Config.BAYES_ITERATIONS,
                    cv=use_cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=Config.RANDOM_STATE,
                    verbose=0
                )

                def on_step(optim_result):
                    nonlocal completed_iterations
                    completed_iterations += 1
                    progress_callback(completed_iterations / total_iterations)

                # Try Bayesian tuning: pass groups to fit when spatial_groups are available
                try:
                    with temp_random_seed(Config.RANDOM_STATE):
                        if spatial_groups is not None:
                            bayes_search.fit(X_train_for_tuning, y_train.values, groups=spatial_groups, callback=on_step)
                        else:
                            bayes_search.fit(X_train_for_tuning, y_train.values, callback=on_step)
                    best_model = bayes_search.best_estimator_
                    best_params = bayes_search.best_params_
                    logger.info(f"  - Best parameters: {best_params}")
                except Exception as e:
                    logger.warning(f"BayesSearchCV failed ({e}); falling back to default model fit")
                    best_model = mp["model"]
                    try:
                        with temp_random_seed(Config.RANDOM_STATE):
                            best_model.fit(X_train_model, y_train_balanced)
                    except Exception as fit_e:
                        logger.error(f"Fallback model fit failed: {fit_e}")
                        raise
                    best_params = {}
            else:
                # No tuning needed (e.g., stacking ensemble)
                best_model = mp["model"]
                with temp_random_seed(Config.RANDOM_STATE):
                    best_model.fit(X_train_model, y_train_balanced)
                best_params = {}
                completed_iterations += 1
                progress_callback(completed_iterations / total_iterations)

            # Save model
            model_filename = os.path.join(output_model_folder,
                                          f"tuned_{name.replace(' ', '_').lower()}.joblib")
            joblib.dump(best_model, model_filename)

            # Save metadata
            model_metadata = {
                "model_name": name,
                "use_scaled": mp["use_scaled"],
                "best_params": best_params,
                "feature_order": Config.FEATURE_NAMES,
                "random_state": Config.RANDOM_STATE,
                "balance_strategy": Config.BALANCE_STRATEGY,
                "spatial_cv": Config.USE_SPATIAL_CV
            }
            metadata_filename = os.path.join(output_model_folder,
                                            f"tuned_{name.replace(' ', '_').lower()}_metadata.joblib")
            joblib.dump(model_metadata, metadata_filename)
            logger.info(f"  - Tuned model saved to: '{model_filename}'")

            # Evaluate on test set
            X_test_model = X_test_scaled if mp["use_scaled"] else X_test
            y_pred_proba = best_model.predict_proba(X_test_model)[:, 1]

            # Find optimal threshold
            optimal_threshold, optimal_score = find_optimal_threshold(
                y_test, y_pred_proba, metric=Config.THRESHOLD_METRIC
            )
            logger.info(f"  - Optimal threshold ({Config.THRESHOLD_METRIC}): {optimal_threshold:.3f} "
                       f"(score: {optimal_score:.3f})")

            # Make predictions with optimal threshold
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)

            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results_list.append({
                "Model": name,
                "ROC AUC": roc_auc,
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Optimal Threshold": optimal_threshold
            })

            logger.info(f"  - ROC AUC: {roc_auc:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

        # Save comparison report
        results_df = pd.DataFrame(results_list).sort_values(by="ROC AUC", ascending=False)
        report_filename = os.path.join(output_folder, "model_comparison_report.csv")
        results_df.to_csv(report_filename, index=False)

        logger.info("\n\n--- Bayesian Tuned Model Performance Comparison ---")
        logger.info("\n" + results_df.to_string(index=False))
        logger.info(f"\nDetailed report saved to '{report_filename}'")

        return True

    except Exception as e:
        log_error_details(logging.getLogger('LandslideAnalysis'), "MODEL TUNING", e)
        return False

# =============================================================================
# MODEL INTERPRETABILITY AND EVALUATION
# =============================================================================

def generate_model_diagnostics(model: Any, X: pd.DataFrame, y: pd.Series,
                             output_folder: str, model_name: str) -> None:
    """Generate comprehensive model diagnostics and calibration plots."""
    try:
        # Calibration curve
        plt.figure(figsize=(10, 6))
        prob_true, prob_pred = calibration_curve(y, model.predict_proba(X)[:, 1], n_bins=10)

        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.plot(prob_pred, prob_true, "s-", label=f"{model_name}")

        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(output_folder, f"calibration_{model_name.replace(' ', '_')}.png"),
                   dpi=Config.FIGURE_DPI)
        plt.close()

        # Feature correlation analysis
        plt.figure(figsize=(12, 8))
        correlation_matrix = X.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()

        plt.savefig(os.path.join(output_folder, f"feature_correlation_{model_name.replace(' ', '_')}.png"),
                   dpi=Config.FIGURE_DPI)
        plt.close()

        # Save feature importance ranking
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            importance_df.to_csv(os.path.join(output_folder,
                                            f"feature_importance_{model_name.replace(' ', '_')}.csv"),
                               index=False)

    except Exception as e:
        logging.warning(f"Model diagnostics generation failed: {e}")

def generate_shap_analysis(model: Any, X_test: pd.DataFrame,
                          output_folder: str, model_name: str) -> bool:
    """Generate comprehensive SHAP analysis with detailed feature interactions."""
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not available. Skipping SHAP analysis.")
        return False

    try:
        logging.info("  - Generating enhanced SHAP analysis...")

        # Sample data if too large (configurable)
        sample_size = min(Config.SHAP_SAMPLE_SIZE, len(X_test))
        with temp_random_seed(Config.RANDOM_STATE):
            X_sample = X_test.sample(n=sample_size, random_state=Config.RANDOM_STATE)

        # Create explainer based on model type
        if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
            # Use a smaller background sample for TreeExplainer for performance
            with temp_random_seed(Config.RANDOM_STATE):
                background = shap.sample(X_test, min(Config.SHAP_BACKGROUND_SIZE, len(X_test)))
            explainer = shap.TreeExplainer(model, data=background)
            shap_values = explainer.shap_values(X_sample)

            # For binary classification, use class 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Approximate CI by re-sampling a few times (fast)
            shap_cis = []
            for i in range(5):
                with temp_random_seed(Config.RANDOM_STATE + i):
                    sample_i = X_sample.sample(frac=0.8, random_state=Config.RANDOM_STATE + i)
                    vals = explainer.shap_values(sample_i)
                    if isinstance(vals, list):
                        vals = vals[1]
                    shap_cis.append(vals)
            shap_std = np.std(shap_cis, axis=0)
        else:
            # Use KernelExplainer for other models
            with temp_random_seed(Config.RANDOM_STATE):
                background = shap.sample(X_test, min(50, len(X_test)))
            explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background)
            shap_values = explainer.shap_values(X_sample)

            # Approximate confidence intervals for kernel explainer (very few resamples)
            shap_cis = []
            for i in range(3):
                with temp_random_seed(Config.RANDOM_STATE + i):
                    sample_i = X_sample.sample(frac=0.8, random_state=Config.RANDOM_STATE + i)
                    vals = explainer.shap_values(sample_i)
                    shap_cis.append(vals)
            shap_std = np.std(shap_cis, axis=0)

        # Enhanced summary plot with confidence intervals
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample,
                         feature_names=list(X_sample.columns),  # Use actual column names
                         show=False,
                         plot_size=(12, 8),
                         random_state=42)  # Use random_state instead of seed
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"shap_summary_{model_name.replace(' ', '_')}.png"),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        # Feature importance plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample,
                         feature_names=list(X_sample.columns),  # Use actual column names
                         plot_type="bar", show=False,
                         random_state=42)  # Use random_state instead of seed
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"shap_importance_{model_name.replace(' ', '_')}.png"),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        logging.info("  - SHAP analysis completed successfully")
        return True

    except Exception as e:
        logging.warning(f"SHAP analysis failed: {e}")
        return False

# =============================================================================
# MODEL ANALYSIS
# =============================================================================

def generate_model_analysis(best_model_name: str, input_csv: str,
                           output_folder: str, log_callback: Callable,
                           progress_callback: Callable) -> bool:
    """Generates comprehensive model performance analysis with enhancements."""
    try:
        logger = logging.getLogger('LandslideAnalysis')
        logger.info("\n" + "="*50)
        logger.info("--- GENERATING MODEL ANALYSIS ---")
        logger.info("="*50)

        # If user passed a numeric selection (e.g., '2'), map it to the model name
        try:
            sel_idx = int(best_model_name)
            report_path = os.path.join(output_folder, "model_comparison_report.csv")
            if os.path.exists(report_path):
                try:
                    df_report = pd.read_csv(report_path)
                    if 1 <= sel_idx <= len(df_report):
                        mapped = df_report.loc[sel_idx - 1, 'Model']
                        logger.info(f"Mapped numeric model selection {sel_idx} -> '{mapped}'")
                        best_model_name = mapped
                except Exception:
                    logger.warning("Could not map numeric model selection to report; proceeding with given value")
        except Exception:
            # not an integer, proceed as before
            pass

        analysis_folder = os.path.join(output_folder, "model_analysis")
        os.makedirs(analysis_folder, exist_ok=True)

        df = pd.read_csv(input_csv)

        from utils import validate_input_data
        if not validate_input_data(df, logger):
            return False

        df.dropna(inplace=True)

        # Load feature list used during training
        feature_list_path = os.path.join(output_folder, "tuned_models", "feature_list.joblib")
        if not os.path.exists(feature_list_path):
            logger.error("Feature list not found. Model needs to be retrained.")
            return False

        feature_list = joblib.load(feature_list_path)
        logger.info(f"Using features from training: {feature_list}")

        # Use the same features as during training
        X = df[feature_list]
        y = df['Type']

        # Load model and scaler
        model_filename = os.path.join(output_folder, "tuned_models",
                                     f"tuned_{best_model_name.replace(' ', '_').lower()}.joblib")
        scaler_filename = os.path.join(output_folder, "tuned_models", "final_scaler.joblib")

        validate_file_exists(model_filename, "model file")
        validate_file_exists(scaler_filename, "scaler file")

        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)

        # Load metadata
        metadata_filename = os.path.join(output_folder, "tuned_models",
                                        f"tuned_{best_model_name.replace(' ', '_').lower()}_metadata.joblib")
        use_scaled = isinstance(model, SVC)
        optimal_threshold = 0.5

        if os.path.exists(metadata_filename):
            metadata = joblib.load(metadata_filename)
            use_scaled = metadata.get("use_scaled", use_scaled)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y
        )

        # Process test data
        X_test_processed = scaler.transform(X_test) if use_scaled else X_test

        # Ensure predict_proba exists
        if not hasattr(model, "predict_proba"):
            logger.error("ERROR: Selected model does not support probability estimates")
            return False

        # Get predictions
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

        # Find optimal threshold
        optimal_threshold, optimal_score = find_optimal_threshold(
            y_test, y_pred_proba, metric=Config.THRESHOLD_METRIC
        )
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        logger.info(f"  - Using optimal threshold: {optimal_threshold:.3f}")

        # Confusion matrix
        logger.info("  - Generating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Landslide', 'Landslide'],
                   yticklabels=['Non-Landslide', 'Landslide'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        plt.figtext(0.02, 0.02,
                   f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\n'
                   f'Recall: {recall:.3f}\nF1-Score: {f1:.3f}\n'
                   f'Threshold: {optimal_threshold:.3f}',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_folder,
                                f"confusion_matrix_{best_model_name.replace(' ', '_')}.png"),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        # Calibration curve
        logger.info("  - Generating calibration curve...")
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                label=f"{best_model_name}")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve - {best_model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_folder,
                                f"calibration_curve_{best_model_name.replace(' ', '_')}.png"),
                   dpi=Config.FIGURE_DPI)
        plt.close()

        # Feature importance
        logger.info("  - Generating feature importance/coefficient plot...")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names_clean = [name.replace('_', ' ').title() for name in Config.FEATURE_NAMES]

            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(importances)), importances[indices])
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title(f'Feature Importance - {best_model_name}')
            plt.xticks(range(len(importances)),
                      [feature_names_clean[i] for i in indices],
                      rotation=45, ha='right')
            colors = plt.cm.viridis(np.linspace(0, 1, len(importances)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_folder,
                                    f"feature_importance_{best_model_name.replace(' ', '_')}.png"),
                       dpi=Config.FIGURE_DPI, bbox_inches='tight')
            plt.close()

            importance_df = pd.DataFrame({
                'Feature': [Config.FEATURE_NAMES[i] for i in indices],
                'Importance': importances[indices]
            })
            importance_df.to_csv(os.path.join(analysis_folder,
                                             f"feature_importance_{best_model_name.replace(' ', '_')}.csv"),
                                index=False)

        elif hasattr(model, 'coef_'):
            coef = model.coef_[0]
            indices = np.argsort(np.abs(coef))[::-1]
            feature_names_clean = [name.replace('_', ' ').title() for name in Config.FEATURE_NAMES]

            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(coef)), coef[indices])
            plt.xlabel('Features')
            plt.ylabel('Coefficient Value')
            plt.title(f'Feature Coefficients - {best_model_name}')
            plt.xticks(range(len(coef)),
                      [feature_names_clean[i] for i in indices],
                      rotation=45, ha='right')
            colors = ['red' if x < 0 else 'blue' for x in coef[indices]]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_folder,
                                    f"feature_coefficients_{best_model_name.replace(' ', '_')}.png"),
                       dpi=Config.FIGURE_DPI, bbox_inches='tight')
            plt.close()

        # SHAP analysis
        generate_shap_analysis(model, X_test, analysis_folder, best_model_name)

        # ROC curve
        logger.info("  - Generating ROC curve...")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {best_model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_folder,
                                f"roc_curve_{best_model_name.replace(' ', '_')}.png"),
                   dpi=Config.FIGURE_DPI)
        plt.close()

        # Precision-Recall curve
        logger.info("  - Generating Precision-Recall curve...")
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {best_model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_folder,
                                f"precision_recall_curve_{best_model_name.replace(' ', '_')}.png"),
                   dpi=Config.FIGURE_DPI)
        plt.close()

        # Performance summary
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        performance_summary = {
            'Model': best_model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall (Sensitivity)': sensitivity,
            'Specificity': specificity,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Optimal Threshold': optimal_threshold,
            'True Positives': int(tp),
            'True Negatives': int(tn),
            'False Positives': int(fp),
            'False Negatives': int(fn)
        }

        summary_df = pd.DataFrame([performance_summary])
        summary_df.to_csv(os.path.join(analysis_folder,
                                      f"performance_summary_{best_model_name.replace(' ', '_')}.csv"),
                         index=False)

        logger.info(f"\n--- Model Analysis Complete! ---")
        logger.info(f"Analysis files saved to: {analysis_folder}")
        for key, value in performance_summary.items():
            if isinstance(value, float):
                logger.info(f"  - {key}: {value:.3f}")
            else:
                logger.info(f"  - {key}: {value}")

        return True

    except Exception as e:
        log_error_details(logging.getLogger('LandslideAnalysis'), "MODEL ANALYSIS", e)
        return False