from __future__ import annotations

import os
from pathlib import Path

TMDB_V3_BASE_URL = "https://api.themoviedb.org/3"


def _discover_project_root(start_path: Path) -> Path:
    current = start_path.resolve()
    while not (current / "data").exists() and current != current.parent:
        current = current.parent
    if not (current / "data").exists():
        raise FileNotFoundError("Não foi possível localizar a raiz do projeto.")
    return current


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _discover_project_root(MODULE_DIR)
ENV_FILE_PATH = PROJECT_ROOT / ".env"


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[len("export "):].strip()

    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    return key, value


def load_project_env_file(
    env_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    normalized_path = Path(env_path) if env_path is not None else ENV_FILE_PATH
    if not normalized_path.exists():
        return {}

    loaded_values: dict[str, str] = {}
    for raw_line in normalized_path.read_text(encoding="utf-8").splitlines():
        parsed_line = _parse_env_line(raw_line)
        if parsed_line is None:
            continue

        key, value = parsed_line
        if overwrite or key not in os.environ:
            os.environ[key] = value
            loaded_values[key] = value

    return loaded_values


def resolve_tmdb_api_key(
    explicit_api_key: str | None = None,
    env_var_name: str = "TMDB_API_KEY",
) -> str:
    load_project_env_file(overwrite=True)
    api_key = os.getenv(env_var_name) or explicit_api_key
    if not api_key:
        raise ValueError(
            "Defina a chave da API do TMDB no arquivo .env na raiz do projeto. "
            f"Como alternativa, use a variavel de ambiente {env_var_name} "
            "ou um valor explícito apenas como fallback."
        )
    return api_key
