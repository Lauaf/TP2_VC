from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import logging
from dotenv import find_dotenv, load_dotenv


def _resolve_env_path() -> Path | None:
    module_env = Path(__file__).resolve().parents[1] / ".env"
    if module_env.is_file():
        return module_env.resolve()
    env_path = find_dotenv(filename=".env", usecwd=True)
    if env_path:
        return Path(env_path).resolve()
    return None


_ENV_FILE = _resolve_env_path()
if _ENV_FILE is not None:
    load_dotenv(_ENV_FILE)
else:
    load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _settings_base_dir() -> Path:
    if _ENV_FILE is not None:
        return _ENV_FILE.parent
    return Path.cwd().resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE is not None else ".env",
        env_file_encoding="utf-8",
    )

    cors_origins: str = Field(
        default="*",
        description="Comma-separated allowed origins, or * for any.",
    )
    app_name: str = "Dog Breed Recognition TP2"

    # Modelo de embeddings seleccionado: baseline | resnet18_finetuned | cnn_custom
    embedding_model: str = "baseline"
    model_path: Path = Path("models")
    resnet18_model_name: str = "resnet18_finetuned.pth"
    cnn_custom_model_name: str = "cnn_custom.pth"

    # Busqueda por similitud
    similarity_metric: str = "cosine"
    similarity_threshold: float = 0.55
    top_k: int = 10
    image_size: int = 224
    embedding_dim: int = 512

    # Configuracion de YOLO (Etapa 3)
    yolo_model: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.25
    yolo_dog_class_id: int = 16

    # Paths
    embeddings_path: Path = Path("data/embeddings.json")
    data_path: Path = Path("data")
    dataset_path: Path = Path("data/dataset")
    output_path: Path = Path("output")
    max_workers: int = 2

    # PostgreSQL / pgvector
    use_pgvector: bool = True
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "dogs"
    postgres_user: str = "dogs_user"
    postgres_password: str = "dogs_pass"

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "Settings":
        base_dir = _settings_base_dir()
        for field_name in (
            "model_path",
            "embeddings_path",
            "data_path",
            "dataset_path",
            "output_path",
        ):
            value = getattr(self, field_name)
            if not value.is_absolute():
                setattr(self, field_name, (base_dir / value).resolve())
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
