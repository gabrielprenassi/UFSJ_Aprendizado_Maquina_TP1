from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from tmdb_api_utils import TMDB_V3_BASE_URL


def _discover_project_root(start_path: Path) -> Path:
    current = start_path.resolve()
    while not (current / "data").exists() and current != current.parent:
        current = current.parent
    if not (current / "data").exists():
        raise FileNotFoundError("Não foi possível localizar a raiz do projeto.")
    return current


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _discover_project_root(MODULE_DIR)

TMDB_ADDITIONAL_METADATA_PATH = PROJECT_ROOT / "data" / "TMDB_movies_additional_metadata.csv"
TMDB_EXTENDED_PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "TMDB_movies_processed_tmdb_extended.csv"

DEFAULT_TOP_CAST_MEMBERS = 3
DEFAULT_TOP_CAST_LABELS = 100
DEFAULT_TOP_DIRECTOR_LABELS = 50
DEFAULT_TOP_COMPANY_LABELS = 50

TERMINAL_FETCH_STATUSES = {
    "ok",
    "missing_release_date",
    "http_error",
}


def build_tmdb_session() -> requests.Session:
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=(429, 500, 502, 503, 504),
        backoff_factor=1.0,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({"accept": "application/json"})
    return session


def load_movies_for_tmdb_enrichment(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    movies_df = pd.read_csv(dataset_path)

    required_columns = {"id_tmdb", "title", "revenue"}
    missing_columns = sorted(required_columns - set(movies_df.columns))
    if missing_columns:
        raise ValueError(
            "A base processada precisa conter as colunas "
            f"{sorted(required_columns)}. Ausentes: {missing_columns}"
        )

    if not movies_df["id_tmdb"].is_unique:
        raise ValueError("A base processada precisa ter id_tmdb unico para o enriquecimento do TMDB.")

    return movies_df


def _prepare_cache(cache_path: str | Path, expected_columns: list[str]) -> pd.DataFrame:
    normalized_path = Path(cache_path)
    if not normalized_path.exists():
        return pd.DataFrame(columns=expected_columns)

    cache_df = pd.read_csv(normalized_path)
    missing_columns = [column for column in expected_columns if column not in cache_df.columns]
    for column in missing_columns:
        cache_df[column] = np.nan

    cache_df = cache_df[expected_columns].copy()
    if "id_tmdb" in cache_df.columns:
        cache_df["id_tmdb"] = cache_df["id_tmdb"].astype("Int64")
    return cache_df


def _flush_cache(
    existing_df: pd.DataFrame,
    new_records: list[dict[str, Any]],
    output_path: str | Path,
) -> pd.DataFrame:
    if not new_records:
        return existing_df

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    updates_df = pd.DataFrame(new_records)
    combined_df = pd.concat([existing_df, updates_df], ignore_index=True)
    combined_df["id_tmdb"] = combined_df["id_tmdb"].astype("Int64")
    combined_df = combined_df.drop_duplicates(subset="id_tmdb", keep="last").sort_values("id_tmdb")
    combined_df.to_csv(output_path, index=False)
    new_records.clear()
    return combined_df


def _pipe_join(values: list[str]) -> str:
    cleaned_values = [str(value).strip() for value in values if str(value).strip()]
    return "|".join(cleaned_values)


def _release_period_label(month: int | None) -> str:
    if month is None or pd.isna(month):
        return "desconhecido"
    month = int(month)
    if month in {12, 1, 2}:
        return "verao"
    if month in {3, 4, 5}:
        return "outono"
    if month in {6, 7, 8}:
        return "inverno"
    return "primavera"


def _normalize_label(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in ascii_value)
    return "_".join(part for part in normalized.split("_") if part)


def _extract_director_name(crew_items: list[dict[str, Any]]) -> str:
    for crew_member in crew_items:
        if crew_member.get("job") == "Director" and crew_member.get("name"):
            return str(crew_member["name"]).strip()
    return ""


def _extract_top_cast_names(cast_items: list[dict[str, Any]], top_cast_members: int) -> list[str]:
    sorted_cast = sorted(
        cast_items,
        key=lambda item: (
            item.get("order", 10_000),
            item.get("cast_id", 10_000),
        ),
    )
    cast_names = []
    for cast_member in sorted_cast:
        cast_name = str(cast_member.get("name") or "").strip()
        if cast_name:
            cast_names.append(cast_name)
        if len(cast_names) >= top_cast_members:
            break
    return cast_names


def fetch_movie_additional_metadata(
    *,
    session: requests.Session,
    movie_id: int,
    title: str,
    api_key: str,
    top_cast_members: int = DEFAULT_TOP_CAST_MEMBERS,
) -> dict[str, Any]:
    url = f"{TMDB_V3_BASE_URL}/movie/{movie_id}"

    try:
        response = session.get(
            url,
            params={
                "api_key": api_key,
                "language": "en-US",
                "append_to_response": "credits",
            },
            timeout=(10, 25),
        )
        http_status = response.status_code
        payload = response.json()

        if response.status_code != 200:
            return {
                "id_tmdb": movie_id,
                "title": title,
                "tmdb_title": payload.get("title", ""),
                "release_date": "",
                "release_year": np.nan,
                "release_month": np.nan,
                "release_quarter": "",
                "release_period": "",
                "is_summer_release": 0,
                "is_holiday_release": 0,
                "director_name": "",
                "top_billed_cast": "",
                "production_company_names": "",
                "main_production_company": "",
                "production_company_count": 0,
                "cast_member_count": 0,
                "tmdb_fetch_status": "http_error",
                "tmdb_http_status": http_status,
                "tmdb_error": payload.get("status_message", response.text[:300]),
            }

        release_date = str(payload.get("release_date") or "").strip()
        parsed_release_date = pd.to_datetime(release_date, errors="coerce")
        release_month = int(parsed_release_date.month) if pd.notna(parsed_release_date) else np.nan
        release_quarter = f"Q{int(parsed_release_date.quarter)}" if pd.notna(parsed_release_date) else ""
        production_companies = payload.get("production_companies") or []
        production_company_names = [
            str(company.get("name") or "").strip()
            for company in production_companies
            if str(company.get("name") or "").strip()
        ]

        credits = payload.get("credits") or {}
        cast_items = credits.get("cast") or []
        crew_items = credits.get("crew") or []
        top_cast_names = _extract_top_cast_names(cast_items, top_cast_members)
        director_name = _extract_director_name(crew_items)

        release_month_int = None if pd.isna(release_month) else int(release_month)

        return {
            "id_tmdb": movie_id,
            "title": title,
            "tmdb_title": payload.get("title") or payload.get("original_title") or "",
            "release_date": release_date,
            "release_year": int(parsed_release_date.year) if pd.notna(parsed_release_date) else np.nan,
            "release_month": release_month,
            "release_quarter": release_quarter,
            "release_period": _release_period_label(release_month_int),
            "is_summer_release": int(release_month_int in {5, 6, 7, 8}) if release_month_int is not None else 0,
            "is_holiday_release": int(release_month_int in {11, 12}) if release_month_int is not None else 0,
            "director_name": director_name,
            "top_billed_cast": _pipe_join(top_cast_names),
            "production_company_names": _pipe_join(production_company_names),
            "main_production_company": production_company_names[0] if production_company_names else "",
            "production_company_count": len(production_company_names),
            "cast_member_count": len(cast_items),
            "tmdb_fetch_status": "ok" if release_date else "missing_release_date",
            "tmdb_http_status": http_status,
            "tmdb_error": "",
        }
    except requests.RequestException as exc:
        return {
            "id_tmdb": movie_id,
            "title": title,
            "tmdb_title": "",
            "release_date": "",
            "release_year": np.nan,
            "release_month": np.nan,
            "release_quarter": "",
            "release_period": "",
            "is_summer_release": 0,
            "is_holiday_release": 0,
            "director_name": "",
            "top_billed_cast": "",
            "production_company_names": "",
            "main_production_company": "",
            "production_company_count": 0,
            "cast_member_count": 0,
            "tmdb_fetch_status": "request_error",
            "tmdb_http_status": np.nan,
            "tmdb_error": str(exc)[:300],
        }


def collect_tmdb_additional_metadata(
    movies_df: pd.DataFrame,
    *,
    api_key: str,
    output_path: str | Path = TMDB_ADDITIONAL_METADATA_PATH,
    pause_seconds: float = 0.25,
    overwrite_existing: bool = False,
    max_movies: int | None = None,
    flush_every: int = 50,
    top_cast_members: int = DEFAULT_TOP_CAST_MEMBERS,
    show_progress: bool = True,
) -> pd.DataFrame:
    expected_columns = [
        "id_tmdb",
        "title",
        "tmdb_title",
        "release_date",
        "release_year",
        "release_month",
        "release_quarter",
        "release_period",
        "is_summer_release",
        "is_holiday_release",
        "director_name",
        "top_billed_cast",
        "production_company_names",
        "main_production_company",
        "production_company_count",
        "cast_member_count",
        "tmdb_fetch_status",
        "tmdb_http_status",
        "tmdb_error",
    ]

    if overwrite_existing and Path(output_path).exists():
        existing_df = pd.DataFrame(columns=expected_columns)
    else:
        existing_df = _prepare_cache(output_path, expected_columns)

    cached_ids = set(
        existing_df.loc[
            existing_df["tmdb_fetch_status"].fillna("").isin(TERMINAL_FETCH_STATUSES),
            "id_tmdb",
        ].dropna().astype(int)
    )

    pending_df = movies_df.loc[~movies_df["id_tmdb"].isin(cached_ids), ["id_tmdb", "title"]].copy()
    if max_movies is not None:
        pending_df = pending_df.head(max_movies)

    session = build_tmdb_session()
    new_records: list[dict[str, Any]] = []

    progress_iterable = pending_df.itertuples(index=False)
    if show_progress:
        progress_iterable = tqdm(
            progress_iterable,
            total=len(pending_df),
            desc="TMDB metadata enrichment",
        )

    for position, row in enumerate(progress_iterable, start=1):
        new_records.append(
            fetch_movie_additional_metadata(
                session=session,
                movie_id=int(row.id_tmdb),
                title=str(row.title),
                api_key=api_key,
                top_cast_members=top_cast_members,
            )
        )

        if position % flush_every == 0:
            existing_df = _flush_cache(existing_df, new_records, output_path)

        if pause_seconds > 0:
            from time import sleep
            sleep(pause_seconds)

    existing_df = _flush_cache(existing_df, new_records, output_path)
    return existing_df


def _split_pipe_values(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    if not str(value).strip():
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _build_top_label_set(values_series: pd.Series, top_k: int) -> set[str]:
    counts = values_series.value_counts()
    return set(counts.head(top_k).index.tolist())


def _build_top_multilabel_set(values_series: pd.Series, top_k: int) -> set[str]:
    exploded = values_series.explode()
    exploded = exploded.loc[exploded.notna() & (exploded.astype(str).str.strip() != "")]
    counts = exploded.value_counts()
    return set(counts.head(top_k).index.tolist())


def _multihot_encode(
    values_series: pd.Series,
    *,
    prefix: str,
    top_k: int,
) -> pd.DataFrame:
    parsed_values = values_series.apply(_split_pipe_values)
    allowed_labels = _build_top_multilabel_set(parsed_values, top_k)
    filtered_values = parsed_values.apply(
        lambda labels: [label for label in labels if label in allowed_labels]
    )

    if not allowed_labels or filtered_values.apply(len).sum() == 0:
        return pd.DataFrame(index=values_series.index)

    mlb = MultiLabelBinarizer()
    encoded = pd.DataFrame(
        mlb.fit_transform(filtered_values),
        columns=[f"{prefix}_{_normalize_label(label)}" for label in mlb.classes_],
        index=values_series.index,
    )
    return encoded


def _onehot_encode_top_values(
    values_series: pd.Series,
    *,
    prefix: str,
    top_k: int,
) -> pd.DataFrame:
    normalized_values = values_series.fillna("").astype(str).str.strip()
    allowed_labels = _build_top_label_set(normalized_values.loc[normalized_values != ""], top_k)
    filtered = normalized_values.where(normalized_values.isin(allowed_labels), other="")

    if filtered.eq("").all():
        return pd.DataFrame(index=values_series.index)

    encoded = pd.get_dummies(filtered, prefix=prefix)
    if f"{prefix}_" in encoded.columns:
        encoded = encoded.drop(columns=[f"{prefix}_"], errors="ignore")
    encoded.columns = [f"{prefix}_{_normalize_label(column.split(f'{prefix}_', 1)[1])}" for column in encoded.columns]
    return encoded


def merge_tmdb_additional_features_with_processed_dataset(
    *,
    processed_data_path: str | Path,
    metadata_path: str | Path = TMDB_ADDITIONAL_METADATA_PATH,
    output_path: str | Path = TMDB_EXTENDED_PROCESSED_DATA_PATH,
    top_cast_labels: int = DEFAULT_TOP_CAST_LABELS,
    top_director_labels: int = DEFAULT_TOP_DIRECTOR_LABELS,
    top_company_labels: int = DEFAULT_TOP_COMPANY_LABELS,
) -> pd.DataFrame:
    processed_df = pd.read_csv(processed_data_path)
    metadata_df = pd.read_csv(metadata_path)

    metadata_features_df = metadata_df.copy()

    metadata_features_df["release_month"] = pd.to_numeric(metadata_features_df["release_month"], errors="coerce")
    metadata_features_df["release_year"] = pd.to_numeric(metadata_features_df["release_year"], errors="coerce")
    metadata_features_df["release_month"] = metadata_features_df["release_month"].fillna(0).astype(int)
    metadata_features_df["release_year"] = metadata_features_df["release_year"].fillna(0).astype(int)
    metadata_features_df["production_company_count"] = (
        pd.to_numeric(metadata_features_df["production_company_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    metadata_features_df["cast_member_count"] = (
        pd.to_numeric(metadata_features_df["cast_member_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    metadata_features_df["is_summer_release"] = (
        pd.to_numeric(metadata_features_df["is_summer_release"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    metadata_features_df["is_holiday_release"] = (
        pd.to_numeric(metadata_features_df["is_holiday_release"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    director_dummies = _onehot_encode_top_values(
        metadata_features_df["director_name"],
        prefix="director",
        top_k=top_director_labels,
    )
    cast_dummies = _multihot_encode(
        metadata_features_df["top_billed_cast"],
        prefix="cast",
        top_k=top_cast_labels,
    )
    company_dummies = _multihot_encode(
        metadata_features_df["production_company_names"],
        prefix="production_company",
        top_k=top_company_labels,
    )
    release_quarter_dummies = pd.get_dummies(
        metadata_features_df["release_quarter"].fillna("").astype(str),
        prefix="release_quarter",
    ).drop(columns=["release_quarter_"], errors="ignore")
    release_period_dummies = pd.get_dummies(
        metadata_features_df["release_period"].fillna("").astype(str),
        prefix="release_period",
    ).drop(columns=["release_period_"], errors="ignore")

    additional_numeric_df = metadata_features_df[
        [
            "id_tmdb",
            "release_year",
            "release_month",
            "production_company_count",
            "cast_member_count",
            "is_summer_release",
            "is_holiday_release",
        ]
    ].copy()

    additional_features_df = pd.concat(
        [
            additional_numeric_df.drop(columns=["id_tmdb"]),
            director_dummies,
            cast_dummies,
            company_dummies,
            release_quarter_dummies,
            release_period_dummies,
        ],
        axis=1,
    )
    additional_features_df.insert(0, "id_tmdb", metadata_features_df["id_tmdb"].astype(int))

    merged_df = processed_df.merge(
        additional_features_df,
        on="id_tmdb",
        how="left",
        validate="one_to_one",
    )

    new_feature_columns = [column for column in merged_df.columns if column not in processed_df.columns]
    for column in new_feature_columns:
        merged_df[column] = merged_df[column].fillna(0)

    reordered_columns = [column for column in merged_df.columns if column != "revenue"] + ["revenue"]
    merged_df = merged_df[reordered_columns]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    return merged_df
