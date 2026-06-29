from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    f1_score,
    get_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from experiment_utils import (
    MODEL_CONFIGS,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_CONFIGS,
    build_best_params_lookup,
    build_model_from_best_params,
    build_pipeline,
    clean_param_names,
    compute_summary,
    deserialize_params,
    inverse_predictions,
    load_feature_matrix,
    load_processed_movies,
    load_saved_folds,
    load_saved_stratification_bins,
    make_holdout_split,
    serialize_params,
)
from tmdb_feature_enrichment_utils import TMDB_EXTENDED_PROCESSED_DATA_PATH

REVENUE_BAND_LABELS = [
    "Muito baixa receita",
    "Baixa receita",
    "Média receita",
    "Alta receita",
    "Muito alta receita",
]

TMDB_EXTENDED_NO_TRANSFORM_ARTIFACT_DIR = (
    PROJECT_ROOT / "data" / "revenue_model_selection_tmdb_extended_no_transform"
)
ORACLE_BAND_ARTIFACT_DIR = PROJECT_ROOT / "data" / "revenue_oracle_band_models_tmdb_extended"
ROBUST_LOSS_ARTIFACT_DIR = PROJECT_ROOT / "data" / "revenue_robust_losses_tmdb_extended_promising"
HYBRID_CLASSIFICATION_REGRESSION_ARTIFACT_DIR = (
    PROJECT_ROOT / "data" / "revenue_hybrid_classification_regression_tmdb_extended"
)
SOFT_ROUTING_CLASSIFICATION_REGRESSION_ARTIFACT_DIR = (
    PROJECT_ROOT / "data" / "revenue_soft_routing_tmdb_extended"
)

ORACLE_LOCAL_MODEL_NAMES = [
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "XGBoost Regressor",
]

ROBUST_MODEL_CONFIGS = {
    "Decision Tree (MSE)": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_SEED, criterion="squared_error"),
        "param_grid": {
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
    },
    "Decision Tree (MAE)": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_SEED, criterion="absolute_error"),
        "param_grid": {
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
    },
    "Random Forest (MSE)": {
        "estimator": RandomForestRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            criterion="squared_error",
        ),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    },
    "Random Forest (MAE)": {
        "estimator": RandomForestRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            criterion="absolute_error",
        ),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    },
    "Gradient Boosting (Squared Error)": {
        "estimator": GradientBoostingRegressor(
            random_state=RANDOM_SEED,
            loss="squared_error",
        ),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
    },
    "Gradient Boosting (Absolute Error)": {
        "estimator": GradientBoostingRegressor(
            random_state=RANDOM_SEED,
            loss="absolute_error",
        ),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
    },
    "Gradient Boosting (Huber)": {
        "estimator": GradientBoostingRegressor(
            random_state=RANDOM_SEED,
            loss="huber",
        ),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
            "model__alpha": [0.8, 0.9],
        },
    },
    "Gradient Boosting (Quantile Median)": {
        "estimator": GradientBoostingRegressor(
            random_state=RANDOM_SEED,
            loss="quantile",
        ),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
            "model__alpha": [0.5],
        },
    },
    "XGBoost (Squared Error)": {
        "estimator": xgb.XGBRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            objective="reg:squarederror",
        ),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        },
    },
    "XGBoost (MAE)": {
        "estimator": xgb.XGBRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            objective="reg:absoluteerror",
        ),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        },
    },
    "XGBoost (Pseudo-Huber)": {
        "estimator": xgb.XGBRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            objective="reg:pseudohubererror",
        ),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        },
    },
}

ROBUST_MODEL_FAMILY_VARIANTS = {
    "Decision Tree Regressor": [
        "Decision Tree (MSE)",
        "Decision Tree (MAE)",
    ],
    "Random Forest Regressor": [
        "Random Forest (MSE)",
        "Random Forest (MAE)",
    ],
    "Gradient Boosting Regressor": [
        "Gradient Boosting (Squared Error)",
        "Gradient Boosting (Absolute Error)",
        "Gradient Boosting (Huber)",
        "Gradient Boosting (Quantile Median)",
    ],
    "XGBoost Regressor": [
        "XGBoost (Squared Error)",
        "XGBoost (MAE)",
        "XGBoost (Pseudo-Huber)",
    ],
}

HYBRID_CLASSIFIER_CONFIGS = {
    "Random Forest Classifier": {
        "estimator": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        "regressor_model_name": "Random Forest Regressor",
    },
    "Gradient Boosting Classifier": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
        "regressor_model_name": "Gradient Boosting Regressor",
    },
    "XGBoost Classifier": {
        "estimator": xgb.XGBClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=len(REVENUE_BAND_LABELS),
        ),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        },
        "regressor_model_name": "XGBoost Regressor",
    },
}

def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def select_robust_model_configs_from_global_best(
    artifact_dir: str | Path = TMDB_EXTENDED_NO_TRANSFORM_ARTIFACT_DIR,
    *,
    target_name: str = "Sem transformação",
) -> dict[str, Any]:
    _, _, summary_df = load_results_from_artifact_dir(artifact_dir)
    summary_df = summary_df.loc[summary_df["target_version"] == target_name].copy()

    if summary_df.empty:
        raise ValueError(
            f"Nenhum resultado encontrado em {artifact_dir} para o alvo '{target_name}'."
        )

    best_global_row = summary_df.sort_values(["mean_rmse", "mean_mae", "model"]).iloc[0]
    best_global_model_name = str(best_global_row["model"])
    selected_model_names = ROBUST_MODEL_FAMILY_VARIANTS.get(best_global_model_name)

    if selected_model_names is None:
        available = ", ".join(sorted(ROBUST_MODEL_FAMILY_VARIANTS))
        raise ValueError(
            "Nao foi possivel mapear variantes robustas para o melhor modelo global "
            f"'{best_global_model_name}'. Modelos suportados: {available}."
        )

    return {
        "best_global_model_name": best_global_model_name,
        "selected_model_names": list(selected_model_names),
        "model_configs": {
            model_name: ROBUST_MODEL_CONFIGS[model_name]
            for model_name in selected_model_names
        },
    }

def load_tmdb_extended_context(
    dataset_path: str | Path = TMDB_EXTENDED_PROCESSED_DATA_PATH,
) -> dict[str, Any]:
    df_movies = load_processed_movies(dataset_path)
    X, y = load_feature_matrix(df_movies)
    saved_fold_assignments_df, folds_df, _ = load_saved_folds(df_movies)
    revenue_bins = load_saved_stratification_bins()
    revenue_band_series = build_revenue_band_series(y, revenue_bins)
    return {
        "df_movies": df_movies,
        "X": X,
        "y": y,
        "saved_fold_assignments_df": saved_fold_assignments_df,
        "folds_df": folds_df,
        "revenue_bins": revenue_bins,
        "revenue_band_series": revenue_band_series,
    }

def build_revenue_band_series(
    values: np.ndarray | pd.Series,
    revenue_bins: np.ndarray,
    labels: list[str] | None = None,
) -> pd.Series:
    labels = labels or REVENUE_BAND_LABELS
    series = pd.Series(np.asarray(values), copy=False)
    bands = pd.cut(
        series,
        bins=revenue_bins,
        labels=labels,
        include_lowest=True,
    )
    return pd.Series(
        pd.Categorical(bands, categories=labels, ordered=True),
        index=series.index,
        name="faixa_receita",
    )

def build_revenue_band_codes(
    values: np.ndarray | pd.Series,
    revenue_bins: np.ndarray,
    labels: list[str] | None = None,
) -> np.ndarray:
    band_series = build_revenue_band_series(values, revenue_bins, labels=labels)
    return band_series.cat.codes.to_numpy()

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def compute_classification_metrics(y_true_codes: np.ndarray, y_pred_codes: np.ndarray) -> dict[str, float]:
    return {
        "band_macro_f1": float(f1_score(y_true_codes, y_pred_codes, average="macro", zero_division=0)),
    }

def make_stratified_holdout_split(
    labels: np.ndarray | pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels_array = np.asarray(labels)
    positions = np.arange(len(labels_array))
    stratify = labels_array if np.unique(labels_array).size > 1 else None
    train_pos, val_pos = train_test_split(
        positions,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return [(train_pos, val_pos)]

def save_artifact_tables(
    artifact_dir: str | Path,
    *,
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict[str, Path]:
    artifact_dir = ensure_directory(artifact_dir)
    results_path = artifact_dir / "model_selection_results.csv"
    predictions_path = artifact_dir / "model_selection_predictions.csv"
    summary_path = artifact_dir / "model_selection_summary.csv"

    results_df.to_csv(results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return {
        "results_path": results_path,
        "predictions_path": predictions_path,
        "summary_path": summary_path,
    }

def normalize_summary_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(summary_df.index, pd.MultiIndex):
        return summary_df.reset_index()
    return summary_df.copy()

def build_summary_comparison(
    reference_summary_df: pd.DataFrame,
    candidate_summary_df: pd.DataFrame,
    *,
    reference_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    reference_df = normalize_summary_frame(reference_summary_df)
    candidate_df = normalize_summary_frame(candidate_summary_df)
    merged_df = reference_df.merge(
        candidate_df,
        on=["target_version", "model"],
        how="inner",
        suffixes=(f"_{reference_label}", f"_{candidate_label}"),
    )
    merged_df["delta_rmse"] = (
        merged_df[f"mean_rmse_{candidate_label}"] - merged_df[f"mean_rmse_{reference_label}"]
    )
    merged_df["delta_mae"] = (
        merged_df[f"mean_mae_{candidate_label}"] - merged_df[f"mean_mae_{reference_label}"]
    )
    merged_df["delta_r2"] = (
        merged_df[f"mean_r2_{candidate_label}"] - merged_df[f"mean_r2_{reference_label}"]
    )
    return merged_df

def summarize_errors_by_band(
    predictions_df: pd.DataFrame,
    revenue_bins: np.ndarray,
    *,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    labels = labels or REVENUE_BAND_LABELS
    working_df = predictions_df.copy().reset_index(drop=True)
    working_df["faixa_receita"] = build_revenue_band_series(
        working_df["y_true"].to_numpy(),
        revenue_bins,
        labels=labels,
    )
    grouped_df = (
        working_df
        .groupby("faixa_receita", observed=False)
        .agg(
            quantidade_filmes=("row_index", "size"),
            receita_minima=("y_true", "min"),
            receita_maxima=("y_true", "max"),
            mae_medio=("abs_error", "mean"),
            mse=("residual", lambda values: np.mean(np.square(values))),
            residuo_medio=("residual", "mean"),
            mediana_erro_absoluto=("abs_error", "median"),
            percentual_subestimados=("residual", lambda values: (values > 0).mean() * 100),
            percentual_superestimados=("residual", lambda values: (values < 0).mean() * 100),
        )
        .reset_index()
    )
    grouped_df["rmse"] = np.sqrt(grouped_df.pop("mse"))
    return grouped_df

def run_regression_model_selection(
    *,
    df_movies: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    folds_df: pd.DataFrame,
    model_configs: dict[str, dict[str, Any]],
    target_name: str = "Sem transformação",
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_config = TARGET_CONFIGS[target_name]
    fold_results: list[dict[str, Any]] = []
    prediction_records: list[pd.DataFrame] = []

    tasks = [
        {"model_name": model_name, "config": config, "row": row}
        for model_name, config in model_configs.items()
        for _, row in folds_df.iterrows()
    ]

    progress_bar = _make_candidate_progress_bar(
        tasks,
        show_progress=show_progress,
        desc="Regressão robusta",
    )

    for task in tasks:
        model_name = task["model_name"]
        config = task["config"]
        row = task["row"]
        fold = int(row["fold"])
        train_idx = row["train_index"]
        test_idx = row["test_index"]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_model = target_config["forward"](y_train)

        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"modelo={model_name} | fold={fold} | refit=nao",
                refresh=True,
            )

        best_model, best_params = _run_manual_single_split_search(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            scoring=config.get("scoring", "neg_mean_squared_error"),
            X_train=X_train,
            y_train=y_train_model,
            progress_bar=progress_bar,
            task_label=f"{model_name} | fold {fold}",
        )
        y_pred_model = best_model.predict(X_test)
        y_pred = inverse_predictions(y_pred_model, target_name)

        metrics = compute_regression_metrics(y_test, y_pred)
        fold_results.append(
            {
                "target_version": target_name,
                "model": model_name,
                "fold": fold,
                **metrics,
                "best_params_json": serialize_params(best_params),
            }
        )

        fold_predictions = pd.DataFrame(
            {
                "row_index": test_idx,
                "id_tmdb": df_movies.iloc[test_idx]["id_tmdb"].values,
                "title": df_movies.iloc[test_idx]["title"].values,
                "target_version": target_name,
                "model": model_name,
                "fold": fold,
                "y_true": y_test,
                "y_pred": y_pred,
            }
        )
        fold_predictions["residual"] = fold_predictions["y_true"] - fold_predictions["y_pred"]
        fold_predictions["abs_error"] = fold_predictions["residual"].abs()
        prediction_records.append(fold_predictions)

    if progress_bar is not None:
        progress_bar.close()

    results_df = pd.DataFrame(fold_results)
    predictions_df = pd.concat(prediction_records, ignore_index=True)
    summary_df = compute_summary(results_df).reset_index()
    return results_df, predictions_df, summary_df

def load_results_from_artifact_dir(artifact_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifact_dir = Path(artifact_dir)
    results_df = pd.read_csv(artifact_dir / "model_selection_results.csv")
    predictions_df = pd.read_csv(artifact_dir / "model_selection_predictions.csv")
    summary_df = pd.read_csv(artifact_dir / "model_selection_summary.csv")
    return results_df, predictions_df, summary_df

def load_best_params_lookup_from_artifact_dir(
    artifact_dir: str | Path = TMDB_EXTENDED_NO_TRANSFORM_ARTIFACT_DIR,
    *,
    target_name: str = "Sem transformação",
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    artifact_dir = Path(artifact_dir)
    results_df = pd.read_csv(artifact_dir / "model_selection_results.csv")
    results_df["best_params"] = results_df["best_params_json"].apply(deserialize_params)
    results_df = results_df.loc[results_df["target_version"] == target_name].copy()
    return build_best_params_lookup(results_df)

def fit_local_band_regressors_from_lookup(
    *,
    model_name: str,
    fold: int,
    best_params_lookup: dict[str, dict[str, list[dict[str, Any]]]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    train_band_labels: pd.Series,
    target_name: str = "Sem transformação",
    min_band_samples: int = 80,
) -> tuple[dict[str, Any], Any, pd.DataFrame]:
    global_best_params = best_params_lookup[target_name][model_name][fold]
    fallback_model = build_model_from_best_params(model_name, global_best_params)
    fallback_model.fit(X_train, TARGET_CONFIGS[target_name]["forward"](y_train))

    local_models: dict[str, Any] = {}
    training_rows: list[dict[str, Any]] = []
    for band_label in REVENUE_BAND_LABELS:
        band_mask = (train_band_labels == band_label).to_numpy()
        n_samples = int(band_mask.sum())
        training_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "faixa_receita": band_label,
                "n_train": n_samples,
                "uses_fallback": int(n_samples < min_band_samples),
            }
        )

        if n_samples < min_band_samples:
            local_models[band_label] = None
            continue

        local_model = build_model_from_best_params(model_name, global_best_params)
        local_model.fit(
            X_train.loc[band_mask],
            TARGET_CONFIGS[target_name]["forward"](y_train[band_mask]),
        )
        local_models[band_label] = local_model

    return local_models, fallback_model, pd.DataFrame(training_rows)

def predict_with_local_band_regressors(
    *,
    local_models: dict[str, Any],
    fallback_model: Any,
    X_test: pd.DataFrame,
    routing_bands: pd.Series | np.ndarray,
    target_name: str = "Sem transformação",
) -> np.ndarray:
    routing_series = pd.Series(np.asarray(routing_bands))
    predictions = np.zeros(len(X_test), dtype=float)

    for band_label, positions in routing_series.groupby(routing_series).groups.items():
        selected_positions = np.asarray(list(positions), dtype=int)
        estimator = local_models.get(str(band_label)) or fallback_model
        y_pred_model = estimator.predict(X_test.iloc[selected_positions])
        predictions[selected_positions] = inverse_predictions(y_pred_model, target_name)

    return predictions

def build_band_probability_frame(
    probabilities: np.ndarray,
    class_codes: np.ndarray | list[int],
    *,
    index: pd.Index | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    labels = labels or REVENUE_BAND_LABELS
    probability_df = pd.DataFrame(0.0, index=index, columns=labels)

    for position, class_code in enumerate(np.asarray(class_codes, dtype=int)):
        probability_df.iloc[:, class_code] = probabilities[:, position]

    row_sums = probability_df.sum(axis=1).to_numpy(dtype=float)
    valid_rows = row_sums > 0
    if not np.all(valid_rows):
        probability_df.loc[~valid_rows, :] = 1.0 / len(labels)
        row_sums = probability_df.sum(axis=1).to_numpy(dtype=float)

    probability_df = probability_df.div(row_sums, axis=0)
    return probability_df

def predict_with_soft_band_regressors(
    *,
    local_models: dict[str, Any],
    fallback_model: Any,
    X_test: pd.DataFrame,
    band_probabilities: pd.DataFrame,
    target_name: str = "Sem transformação",
) -> np.ndarray:
    aligned_probabilities = band_probabilities.reindex(columns=REVENUE_BAND_LABELS, fill_value=0.0)
    weighted_predictions = np.zeros(len(X_test), dtype=float)

    for band_label in REVENUE_BAND_LABELS:
        estimator = local_models.get(band_label) or fallback_model
        y_pred_model = estimator.predict(X_test)
        y_pred = inverse_predictions(y_pred_model, target_name)
        weighted_predictions += aligned_probabilities[band_label].to_numpy(dtype=float) * y_pred

    return weighted_predictions

def run_oracle_band_experiment(
    *,
    df_movies: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    folds_df: pd.DataFrame,
    revenue_bins: np.ndarray,
    best_params_lookup: dict[str, dict[str, list[dict[str, Any]]]],
    model_names: list[str] | None = None,
    target_name: str = "Sem transformação",
    min_band_samples: int = 80,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_names = model_names or ORACLE_LOCAL_MODEL_NAMES
    results_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []

    tasks = [
        {"model_name": model_name, "row": row}
        for model_name in model_names
        for _, row in folds_df.iterrows()
    ]

    for task in _progress(tasks, show_progress=show_progress, desc="Oracle por faixa"):
        model_name = task["model_name"]
        row = task["row"]
        fold = int(row["fold"])
        train_idx = row["train_index"]
        test_idx = row["test_index"]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_bands = build_revenue_band_series(y_train, revenue_bins)
        test_bands = build_revenue_band_series(y_test, revenue_bins)

        local_models, fallback_model, training_df = fit_local_band_regressors_from_lookup(
            model_name=model_name,
            fold=fold,
            best_params_lookup=best_params_lookup,
            X_train=X_train,
            y_train=y_train,
            train_band_labels=train_bands,
            target_name=target_name,
            min_band_samples=min_band_samples,
        )
        training_frames.append(training_df)

        y_pred = predict_with_local_band_regressors(
            local_models=local_models,
            fallback_model=fallback_model,
            X_test=X_test,
            routing_bands=test_bands.reset_index(drop=True),
            target_name=target_name,
        )

        metrics = compute_regression_metrics(y_test, y_pred)
        results_rows.append(
            {
                "target_version": target_name,
                "model": model_name,
                "fold": fold,
                **metrics,
            }
        )

        fold_predictions = pd.DataFrame(
            {
                "row_index": test_idx,
                "id_tmdb": df_movies.iloc[test_idx]["id_tmdb"].values,
                "title": df_movies.iloc[test_idx]["title"].values,
                "target_version": target_name,
                "model": model_name,
                "fold": fold,
                "y_true": y_test,
                "y_pred": y_pred,
                "true_band": test_bands.astype(str).values,
                "routing_band": test_bands.astype(str).values,
                "routing_strategy": "oracle_true_band",
            }
        )
        fold_predictions["residual"] = fold_predictions["y_true"] - fold_predictions["y_pred"]
        fold_predictions["abs_error"] = fold_predictions["residual"].abs()
        prediction_frames.append(fold_predictions)

    results_df = pd.DataFrame(results_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df = compute_summary(results_df).reset_index()
    training_sizes_df = pd.concat(training_frames, ignore_index=True)
    return results_df, predictions_df, summary_df, training_sizes_df

def compute_hybrid_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df
        .groupby(["target_version", "model", "classifier_model", "regressor_model"])
        .agg(
            mean_mse=("mse", "mean"),
            std_mse=("mse", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_band_macro_f1=("band_macro_f1", "mean"),
            std_band_macro_f1=("band_macro_f1", "std"),
        )
        .reset_index()
        .sort_values(["mean_rmse", "mean_mae", "model"])
        .reset_index(drop=True)
    )

def build_classifier_best_params_lookup(
    results_df: pd.DataFrame,
    *,
    params_column: str = "classifier_best_params",
) -> dict[str, dict[int, dict[str, Any]]]:
    classifier_lookup: dict[str, dict[int, dict[str, Any]]] = {}

    for row in results_df.itertuples(index=False):
        classifier_name = str(getattr(row, "classifier_model"))
        fold = int(getattr(row, "fold"))
        params = getattr(row, params_column)
        classifier_lookup.setdefault(classifier_name, {})[fold] = dict(params)

    return classifier_lookup

def load_classifier_best_params_lookup_from_artifact_dir(
    artifact_dir: str | Path = HYBRID_CLASSIFICATION_REGRESSION_ARTIFACT_DIR,
    *,
    target_name: str = "Sem transformação",
) -> dict[str, dict[int, dict[str, Any]]]:
    artifact_dir = Path(artifact_dir)
    results_df = pd.read_csv(artifact_dir / "model_selection_results.csv")
    results_df = results_df.loc[results_df["target_version"] == target_name].copy()

    if results_df.empty:
        raise ValueError(
            f"Nenhum resultado encontrado em {artifact_dir} para o alvo '{target_name}'."
        )

    results_df["classifier_best_params"] = results_df["classifier_best_params_json"].apply(
        deserialize_params
    )
    deduplicated_df = results_df.drop_duplicates(
        subset=["classifier_model", "fold"],
        keep="first",
    )
    return build_classifier_best_params_lookup(
        deduplicated_df,
        params_column="classifier_best_params",
    )

def _prefix_model_params(params: dict[str, Any]) -> dict[str, Any]:
    return {f"model__{key}": value for key, value in params.items()}

def _fit_classifier_for_fold(
    *,
    classifier_name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    progress_bar=None,
    fold: int,
    classifier_best_params_lookup: dict[str, dict[int, dict[str, Any]]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    precomputed_params = None
    if classifier_best_params_lookup is not None:
        precomputed_params = classifier_best_params_lookup.get(classifier_name, {}).get(fold)

    if precomputed_params is not None:
        classifier = build_pipeline(clone(estimator))
        classifier.set_params(**_prefix_model_params(precomputed_params))
        classifier.fit(X_train, y_train)

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"{classifier_name} | fold {fold} | macro-F1=reuso | refit=sim",
                refresh=True,
            )

        return classifier, dict(precomputed_params)

    return _run_manual_single_split_search(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1_macro",
        X_train=X_train,
        y_train=y_train,
        cv_splits=make_stratified_holdout_split(y_train),
        progress_bar=progress_bar,
        task_label=f"{classifier_name} | fold {fold}",
    )

def run_hybrid_classification_regression_experiment(
    *,
    df_movies: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    folds_df: pd.DataFrame,
    revenue_bins: np.ndarray,
    best_params_lookup: dict[str, dict[str, list[dict[str, Any]]]],
    classifier_configs: dict[str, dict[str, Any]] | None = None,
    classifier_best_params_lookup: dict[str, dict[int, dict[str, Any]]] | None = None,
    target_name: str = "Sem transformação",
    min_band_samples: int = 80,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    classifier_configs = classifier_configs or HYBRID_CLASSIFIER_CONFIGS
    results_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []

    tasks = [
        {
            "classifier_name": classifier_name,
            "config": config,
            "row": row,
            "reuse_params": bool(
                classifier_best_params_lookup
                and classifier_best_params_lookup.get(classifier_name, {}).get(int(row["fold"]))
            ),
        }
        for classifier_name, config in classifier_configs.items()
        for _, row in folds_df.iterrows()
    ]

    progress_bar = _make_candidate_progress_bar(
        tasks,
        show_progress=show_progress,
        desc="Classificação + regressão",
    )

    for task in tasks:
        classifier_name = task["classifier_name"]
        config = task["config"]
        row = task["row"]
        regressor_model_name = config["regressor_model_name"]
        strategy_name = f"{classifier_name} + {regressor_model_name}"

        fold = int(row["fold"])
        train_idx = row["train_index"]
        test_idx = row["test_index"]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_bands = build_revenue_band_series(y_train, revenue_bins)
        test_bands = build_revenue_band_series(y_test, revenue_bins)
        y_train_band_codes = train_bands.cat.codes.to_numpy()
        y_test_band_codes = test_bands.cat.codes.to_numpy()

        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"classificador={classifier_name} | fold={fold} | refit=nao",
                refresh=True,
            )

        best_classifier, classifier_best_params = _fit_classifier_for_fold(
            classifier_name=classifier_name,
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            X_train=X_train,
            y_train=y_train_band_codes,
            progress_bar=progress_bar,
            fold=fold,
            classifier_best_params_lookup=classifier_best_params_lookup,
        )
        y_pred_band_codes = np.asarray(best_classifier.predict(X_test), dtype=int)
        y_pred_bands = pd.Categorical.from_codes(
            y_pred_band_codes,
            categories=REVENUE_BAND_LABELS,
            ordered=True,
        )

        local_models, fallback_model, training_df = fit_local_band_regressors_from_lookup(
            model_name=regressor_model_name,
            fold=fold,
            best_params_lookup=best_params_lookup,
            X_train=X_train,
            y_train=y_train,
            train_band_labels=train_bands,
            target_name=target_name,
            min_band_samples=min_band_samples,
        )
        training_df["classifier_model"] = classifier_name
        training_frames.append(training_df)

        y_pred = predict_with_local_band_regressors(
            local_models=local_models,
            fallback_model=fallback_model,
            X_test=X_test,
            routing_bands=pd.Series(y_pred_bands, index=X_test.index),
            target_name=target_name,
        )

        regression_metrics = compute_regression_metrics(y_test, y_pred)
        classification_metrics = compute_classification_metrics(
            y_test_band_codes,
            y_pred_band_codes,
        )

        results_rows.append(
            {
                "target_version": target_name,
                "model": strategy_name,
                "classifier_model": classifier_name,
                "regressor_model": regressor_model_name,
                "fold": fold,
                **regression_metrics,
                **classification_metrics,
                "classifier_best_params_json": serialize_params(classifier_best_params),
            }
        )

        fold_predictions = pd.DataFrame(
            {
                "row_index": test_idx,
                "id_tmdb": df_movies.iloc[test_idx]["id_tmdb"].values,
                "title": df_movies.iloc[test_idx]["title"].values,
                "target_version": target_name,
                "model": strategy_name,
                "classifier_model": classifier_name,
                "regressor_model": regressor_model_name,
                "fold": fold,
                "y_true": y_test,
                "y_pred": y_pred,
                "true_band": test_bands.astype(str).values,
                "predicted_band": pd.Series(y_pred_bands).astype(str).values,
            }
        )
        fold_predictions["residual"] = fold_predictions["y_true"] - fold_predictions["y_pred"]
        fold_predictions["abs_error"] = fold_predictions["residual"].abs()
        prediction_frames.append(fold_predictions)

    if progress_bar is not None:
        progress_bar.close()

    results_df = pd.DataFrame(results_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df = compute_hybrid_summary(results_df)
    training_sizes_df = pd.concat(training_frames, ignore_index=True)
    return results_df, predictions_df, summary_df, training_sizes_df

def run_soft_routing_classification_regression_experiment(
    *,
    df_movies: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    folds_df: pd.DataFrame,
    revenue_bins: np.ndarray,
    best_params_lookup: dict[str, dict[str, list[dict[str, Any]]]],
    classifier_configs: dict[str, dict[str, Any]] | None = None,
    classifier_best_params_lookup: dict[str, dict[int, dict[str, Any]]] | None = None,
    target_name: str = "Sem transformação",
    min_band_samples: int = 80,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    classifier_configs = classifier_configs or HYBRID_CLASSIFIER_CONFIGS
    results_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []

    tasks = [
        {
            "classifier_name": classifier_name,
            "config": config,
            "row": row,
            "reuse_params": bool(
                classifier_best_params_lookup
                and classifier_best_params_lookup.get(classifier_name, {}).get(int(row["fold"]))
            ),
        }
        for classifier_name, config in classifier_configs.items()
        for _, row in folds_df.iterrows()
    ]

    progress_bar = _make_candidate_progress_bar(
        tasks,
        show_progress=show_progress,
        desc="Soft routing",
    )

    for task in tasks:
        classifier_name = task["classifier_name"]
        config = task["config"]
        row = task["row"]
        regressor_model_name = config["regressor_model_name"]
        strategy_name = f"{classifier_name} + {regressor_model_name}"

        fold = int(row["fold"])
        train_idx = row["train_index"]
        test_idx = row["test_index"]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_bands = build_revenue_band_series(y_train, revenue_bins)
        test_bands = build_revenue_band_series(y_test, revenue_bins)
        y_train_band_codes = train_bands.cat.codes.to_numpy()
        y_test_band_codes = test_bands.cat.codes.to_numpy()

        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"classificador={classifier_name} | fold={fold} | refit=nao",
                refresh=True,
            )

        best_classifier, classifier_best_params = _fit_classifier_for_fold(
            classifier_name=classifier_name,
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            X_train=X_train,
            y_train=y_train_band_codes,
            progress_bar=progress_bar,
            fold=fold,
            classifier_best_params_lookup=classifier_best_params_lookup,
        )
        y_pred_band_codes = np.asarray(best_classifier.predict(X_test), dtype=int)
        y_pred_bands = pd.Categorical.from_codes(
            y_pred_band_codes,
            categories=REVENUE_BAND_LABELS,
            ordered=True,
        )
        band_probabilities = build_band_probability_frame(
            np.asarray(best_classifier.predict_proba(X_test), dtype=float),
            np.asarray(best_classifier.classes_, dtype=int),
            index=X_test.index,
        )

        local_models, fallback_model, training_df = fit_local_band_regressors_from_lookup(
            model_name=regressor_model_name,
            fold=fold,
            best_params_lookup=best_params_lookup,
            X_train=X_train,
            y_train=y_train,
            train_band_labels=train_bands,
            target_name=target_name,
            min_band_samples=min_band_samples,
        )
        training_df["classifier_model"] = classifier_name
        training_frames.append(training_df)

        y_pred = predict_with_soft_band_regressors(
            local_models=local_models,
            fallback_model=fallback_model,
            X_test=X_test,
            band_probabilities=band_probabilities,
            target_name=target_name,
        )

        regression_metrics = compute_regression_metrics(y_test, y_pred)
        classification_metrics = compute_classification_metrics(
            y_test_band_codes,
            y_pred_band_codes,
        )

        results_rows.append(
            {
                "target_version": target_name,
                "model": strategy_name,
                "classifier_model": classifier_name,
                "regressor_model": regressor_model_name,
                "fold": fold,
                **regression_metrics,
                **classification_metrics,
                "classifier_best_params_json": serialize_params(classifier_best_params),
            }
        )

        fold_predictions = pd.DataFrame(
            {
                "row_index": test_idx,
                "id_tmdb": df_movies.iloc[test_idx]["id_tmdb"].values,
                "title": df_movies.iloc[test_idx]["title"].values,
                "target_version": target_name,
                "model": strategy_name,
                "classifier_model": classifier_name,
                "regressor_model": regressor_model_name,
                "fold": fold,
                "y_true": y_test,
                "y_pred": y_pred,
                "true_band": test_bands.astype(str).values,
                "predicted_band": pd.Series(y_pred_bands).astype(str).values,
                "classifier_confidence": band_probabilities.max(axis=1).to_numpy(dtype=float),
                "routing_strategy": "soft_probability_weighted",
            }
        )
        fold_predictions["residual"] = fold_predictions["y_true"] - fold_predictions["y_pred"]
        fold_predictions["abs_error"] = fold_predictions["residual"].abs()
        prediction_frames.append(fold_predictions)

    if progress_bar is not None:
        progress_bar.close()

    results_df = pd.DataFrame(results_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df = compute_hybrid_summary(results_df)
    training_sizes_df = pd.concat(training_frames, ignore_index=True)
    return results_df, predictions_df, summary_df, training_sizes_df

def save_additional_table(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)
    return output_path

def _progress(items: list[dict[str, Any]], *, show_progress: bool, desc: str):
    if not show_progress:
        return items
    return tqdm(items, desc=desc, unit="fit", dynamic_ncols=True)

def _make_candidate_progress_bar(
    tasks: list[dict[str, Any]],
    *,
    show_progress: bool,
    desc: str,
):
    if not show_progress:
        return None

    total_candidates = sum(
        1
        if task.get("reuse_params")
        else len(ParameterGrid(task["config"]["param_grid"])) + 1
        for task in tasks
    )
    return tqdm(
        total=total_candidates,
        desc=desc,
        unit="ajuste",
        dynamic_ncols=True,
    )

def _run_manual_single_split_search(
    *,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    scoring: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    progress_bar=None,
    task_label: str = "",
) -> tuple[Any, dict[str, Any]]:
    scorer = get_scorer(scoring)
    splits = cv_splits or make_holdout_split(X_train)
    train_positions, val_positions = splits[0]
    X_fit = X_train.iloc[train_positions]
    y_fit = y_train[train_positions]
    X_val = X_train.iloc[val_positions]
    y_val = y_train[val_positions]

    best_score = -np.inf
    best_params_with_prefix: dict[str, Any] | None = None

    for params in ParameterGrid(param_grid):
        candidate_model = build_pipeline(clone(estimator))
        candidate_model.set_params(**params)
        candidate_model.fit(X_fit, y_fit)
        score = float(scorer(candidate_model, X_val, y_val))

        if score > best_score:
            best_score = score
            best_params_with_prefix = dict(params)

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"{task_label} | melhor={best_score:.4f} | refit=nao",
                refresh=True,
            )

    if best_params_with_prefix is None:
        raise ValueError("A busca manual nao avaliou nenhuma combinacao de hiperparametros.")

    best_model = build_pipeline(clone(estimator))
    best_model.set_params(**best_params_with_prefix)
    best_model.fit(X_train, y_train)

    if progress_bar is not None:
        progress_bar.update(1)
        progress_bar.set_postfix_str(
            f"{task_label} | melhor={best_score:.4f} | refit=sim",
            refresh=True,
        )

    return best_model, clean_param_names(best_params_with_prefix)
