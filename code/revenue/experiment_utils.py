from __future__ import annotations

import ast
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def _discover_project_root(start_path: Path) -> Path:
    current = start_path.resolve()
    while not (current / "data").exists() and current != current.parent:
        current = current.parent
    if not (current / "data").exists():
        raise FileNotFoundError("Não foi possível localizar a raiz do projeto a partir do módulo de revenue.")
    return current

RANDOM_SEED = 222050006
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _discover_project_root(MODULE_DIR)
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "TMDB_movies_processed.csv"
FOLDS_PATH = PROJECT_ROOT / "data" / "revenue_folds.csv"
STRATIFICATION_BINS_PATH = PROJECT_ROOT / "data" / "revenue_stratification_bins.csv"
IMAGES_DIR = PROJECT_ROOT / "code" / "revenue" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ARTICLE_NOTEBOOK_DPI = 170
ARTICLE_EXPORT_DPI = 320

ARTICLE_FIGURE_SIZES = {
    "axis_label": 22,
    "tick": 18,
    "legend": 15,
    "legend_title": 15,
    "annotation": 18,
    "annotation_small": 15,
    "heatmap_annotation": 18,
    "colorbar_tick": 16,
}
ARTICLE_FIGURE_RC = {
    "axes.labelsize": ARTICLE_FIGURE_SIZES["axis_label"],
    "xtick.labelsize": ARTICLE_FIGURE_SIZES["tick"],
    "ytick.labelsize": ARTICLE_FIGURE_SIZES["tick"],
    "legend.fontsize": ARTICLE_FIGURE_SIZES["legend"],
    "legend.title_fontsize": ARTICLE_FIGURE_SIZES["legend_title"],
    "figure.dpi": ARTICLE_NOTEBOOK_DPI,
    "savefig.dpi": ARTICLE_EXPORT_DPI,
}

TARGET_CONFIGS = {
    "Sem transformação": {
        "forward": lambda values: values,
        "inverse": lambda values: values,
    },
}

MODEL_CONFIGS = {
    "Dummy Regressor": {
        "estimator": DummyRegressor(),
        "param_grid": [
            {"model__strategy": ["mean", "median"]},
            {"model__strategy": ["quantile"], "model__quantile": [0.25, 0.5, 0.75]},
        ],
        "supports_native_importance": False,
    },
    "Linear Regression": {
        "estimator": LinearRegression(),
        "param_grid": {
            "model__fit_intercept": [True, False],
            "model__positive": [False, True],
        },
        "supports_native_importance": False,
    },
    "KNN Regressor": {
        "estimator": KNeighborsRegressor(),
        "param_grid": {
            "model__n_neighbors": [3, 5, 11, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "supports_native_importance": False,
    },
    "SVR": {
        "estimator": SVR(),
        "param_grid": [
            {
                "model__kernel": ["linear"],
                "model__C": [0.1, 1.0, 10.0],
                "model__epsilon": [0.01, 0.1],
            },
            {
                "model__kernel": ["rbf"],
                "model__C": [1.0, 10.0, 50.0],
                "model__epsilon": [0.01, 0.1],
                "model__gamma": ["scale", 0.1],
            },
        ],
        "supports_native_importance": False,
    },
    "Decision Tree Regressor": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_SEED),
        "param_grid": {
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "supports_native_importance": True,
    },
    "Random Forest Regressor": {
        "estimator": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        "supports_native_importance": True,
    },
    "Gradient Boosting Regressor": {
        "estimator": GradientBoostingRegressor(random_state=RANDOM_SEED),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
        "supports_native_importance": True,
    },
    "XGBoost Regressor": {
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
        "supports_native_importance": True,
    },
}

TREE_BASED_MODEL_NAMES = [
    model_name
    for model_name, config in MODEL_CONFIGS.items()
    if config["supports_native_importance"]
]

ARTIFACT_DIR = PROJECT_ROOT / "data" / "revenue_model_selection"
RESULTS_PATH = ARTIFACT_DIR / "model_selection_results.csv"
PREDICTIONS_PATH = ARTIFACT_DIR / "model_selection_predictions.csv"
SUMMARY_PATH = ARTIFACT_DIR / "model_selection_summary.csv"
MODEL_SELECTION_ARTIFACT_DIR_CANDIDATES = [
    ARTIFACT_DIR,
    PROJECT_ROOT / "bkp" / "data copy" / "revenue_model_selection",
]

def _slugify_plot_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = "".join(char.lower() if char.isalnum() else "_" for char in normalized)
    return "_".join(part for part in slug.split("_") if part)

def build_image_path(name: str) -> Path:
    return IMAGES_DIR / f"{_slugify_plot_name(name)}.png"

def announce_figure(title: str) -> None:
    print(f"Figura: {title}")

def apply_article_figure_theme() -> None:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk", rc=ARTICLE_FIGURE_RC)

def save_current_figure(name: str, dpi: int = ARTICLE_EXPORT_DPI) -> Path:
    import matplotlib.pyplot as plt

    image_path = build_image_path(name)
    plt.savefig(image_path, dpi=dpi, bbox_inches="tight")
    return image_path

def show_and_save_current_figure(
    name: str,
    *,
    notebook_title: str | None = None,
    dpi: int = ARTICLE_EXPORT_DPI,
) -> Path:
    import matplotlib.pyplot as plt

    if notebook_title:
        announce_figure(notebook_title)
    image_path = save_current_figure(name, dpi=dpi)
    plt.show()
    plt.close()
    return image_path

def load_processed_movies(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path is not None else PROCESSED_DATA_PATH
    return pd.read_csv(dataset_path)

def load_feature_matrix(df_movies: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df_movies.drop(columns=["id_tmdb", "title", "revenue"])
    y = df_movies["revenue"].to_numpy()
    return X, y

def load_saved_stratification_bins(path: str | Path | None = None) -> np.ndarray:
    bins_path = Path(path) if path is not None else STRATIFICATION_BINS_PATH
    bins_df = pd.read_csv(bins_path)
    return bins_df["bin_edge"].to_numpy()

def load_saved_folds(
    df_movies: pd.DataFrame,
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    folds_path = Path(path) if path is not None else FOLDS_PATH
    saved_fold_assignments = pd.read_csv(folds_path)
    fold_assignments = df_movies[["id_tmdb"]].merge(
        saved_fold_assignments,
        on="id_tmdb",
        how="left",
        validate="one_to_one",
    )

    if fold_assignments["fold"].isna().any():
        missing_ids = fold_assignments.loc[fold_assignments["fold"].isna(), "id_tmdb"].tolist()
        raise ValueError(f"Existem filmes sem fold atribuído: {missing_ids[:5]}")

    fold_assignments["fold"] = fold_assignments["fold"].astype(int)
    positions = np.arange(len(df_movies))
    fold_labels = fold_assignments["fold"].to_numpy()

    folds_data = []
    for fold in sorted(np.unique(fold_labels)):
        test_index = positions[fold_labels == fold]
        train_index = positions[fold_labels != fold]
        folds_data.append(
            {
                "fold": int(fold),
                "train_index": train_index,
                "test_index": test_index,
            }
        )

    return saved_fold_assignments, pd.DataFrame(folds_data), fold_labels

def make_holdout_split(
    X_train: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    positions = np.arange(len(X_train))
    train_pos, val_pos = train_test_split(
        positions,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return [(train_pos, val_pos)]

def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("model", estimator),
        ]
    )

def clean_param_names(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key.replace("model__", ""): value
        for key, value in params.items()
    }

def _normalize_param_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value

def serialize_params(params: dict[str, Any]) -> str:
    normalized = {
        key: _normalize_param_value(value)
        for key, value in params.items()
    }
    return json.dumps(normalized, sort_keys=True)

def deserialize_params(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if pd.isna(payload):
        return {}

    text = str(payload)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)

def build_model_from_best_params(model_name: str, best_params: dict[str, Any]) -> Pipeline:
    estimator = clone(MODEL_CONFIGS[model_name]["estimator"])
    estimator.set_params(**best_params)
    return build_pipeline(estimator)

def inverse_predictions(values: np.ndarray, target_name: str) -> np.ndarray:
    original_scale_values = TARGET_CONFIGS[target_name]["inverse"](values)
    return np.clip(original_scale_values, a_min=0, a_max=None)

def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df
        .groupby(["target_version", "model"])
        .agg(
            mean_mse=("mse", "mean"),
            std_mse=("mse", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
        )
        .sort_values(by="mean_rmse")
    )

def format_metric_value(value: float, metric_key: str) -> str:
    if pd.isna(value):
        return "-"

    normalized_key = metric_key.lower()

    if "r2" in normalized_key:
        return f"{value:.3f}"

    if "percentage" in normalized_key:
        return f"{value:.2f}%"

    if "mse" in normalized_key and "rmse" not in normalized_key:
        return f"{value:.2e}"

    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f} mi"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f} mil"

    return f"{value:.2f}"

def format_metric_mean_std(mean_value: float, std_value: float, metric_key: str) -> str:
    return (
        f"{format_metric_value(mean_value, metric_key)} "
        f"(± {format_metric_value(std_value, metric_key)})"
    )

def format_summary_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_display_df = summary_df.reset_index().copy()
    summary_display_df["MSE"] = summary_display_df.apply(
        lambda row: format_metric_mean_std(row["mean_mse"], row["std_mse"], "mse"),
        axis=1,
    )
    summary_display_df["RMSE"] = summary_display_df.apply(
        lambda row: format_metric_mean_std(row["mean_rmse"], row["std_rmse"], "rmse"),
        axis=1,
    )
    summary_display_df["MAE"] = summary_display_df.apply(
        lambda row: format_metric_mean_std(row["mean_mae"], row["std_mae"], "mae"),
        axis=1,
    )
    summary_display_df["R²"] = summary_display_df.apply(
        lambda row: format_metric_mean_std(row["mean_r2"], row["std_r2"], "r2"),
        axis=1,
    )

    return (
        summary_display_df
        .rename(
            columns={
                "target_version": "Versão do alvo",
                "model": "Modelo",
            }
        )
        .set_index(["Versão do alvo", "Modelo"])[["MSE", "RMSE", "MAE", "R²"]]
    )

def build_best_params_lookup(results_df: pd.DataFrame) -> dict[str, dict[str, list[dict[str, Any]]]]:
    results_with_params = results_df.copy()
    if "best_params" not in results_with_params.columns and "best_params_json" in results_with_params.columns:
        results_with_params["best_params"] = results_with_params["best_params_json"].apply(deserialize_params)

    lookup: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for target_name, target_group in results_with_params.groupby("target_version"):
        lookup[target_name] = {}
        for model_name, model_group in target_group.groupby("model"):
            lookup[target_name][model_name] = (
                model_group
                .sort_values("fold")["best_params"]
                .tolist()
            )

    return lookup

def resolve_model_selection_artifact_paths() -> tuple[Path, Path, Path]:
    for artifact_dir in MODEL_SELECTION_ARTIFACT_DIR_CANDIDATES:
        results_path = artifact_dir / "model_selection_results.csv"
        predictions_path = artifact_dir / "model_selection_predictions.csv"
        summary_path = artifact_dir / "model_selection_summary.csv"

        if results_path.exists() and predictions_path.exists():
            return results_path, predictions_path, summary_path

    raise FileNotFoundError(
        "Os artefatos de seleção de modelos não foram encontrados. "
        "Execute 02_model_grid_search.ipynb primeiro."
    )

def load_model_selection_artifacts(
    target_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_path, predictions_path, summary_path = resolve_model_selection_artifact_paths()

    results_df = pd.read_csv(results_path)
    predictions_df = pd.read_csv(predictions_path)
    if "best_params_json" in results_df.columns:
        results_df["best_params"] = results_df["best_params_json"].apply(deserialize_params)

    if target_name is not None:
        results_df = results_df.loc[results_df["target_version"] == target_name].copy()
        predictions_df = predictions_df.loc[predictions_df["target_version"] == target_name].copy()

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if target_name is not None:
            summary_df = summary_df.loc[summary_df["target_version"] == target_name].copy()
        summary_df = summary_df.set_index(["target_version", "model"]).sort_values(by="mean_rmse")
    else:
        summary_df = compute_summary(results_df)

    return results_df, predictions_df, summary_df