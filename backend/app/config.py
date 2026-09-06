from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "change-me-in-production"

    database_url: str | None = None
    enable_sqlite_fallback: bool = True
    sqlite_path: str = "./backend/data/vision_companion.db"

    cors_origins: str = "http://localhost:8081,http://127.0.0.1:8081"

    max_upload_bytes: int = 8_388_608
    allowed_image_types: str = "image/jpeg,image/png,image/webp"

    device: str = "cpu"
    yolo_model: str = "yolo11n.pt"
    yolo_world_model: str = "yolov8s-worldv2.pt"
    yolo_confidence: float = 0.35
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    whisper_model: str = "tiny"
    enable_ocr: bool = True
    enable_yolo_world: bool = False
    enable_depth: bool = True
    enable_tracking: bool = True

    depth_higher_means_farther: bool = True
    depth_near_threshold: float = 0.35
    depth_mid_threshold: float = 0.65

    rate_limit_analyze: str = "20/minute"
    rate_limit_speech: str = "30/minute"

    store_images: bool = False
    log_image_metadata_only: bool = True
    enable_model_warmup: bool = True
    enable_rate_limit: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def allowed_image_type_set(self) -> set[str]:
        return {item.strip() for item in self.allowed_image_types.split(",") if item.strip()}

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def weights_dir(self) -> Path:
        path = Path(__file__).resolve().parents[1] / "models" / "weights"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_database_url(self) -> str:
        if self.database_url and self.database_url.strip():
            url = self.database_url.strip()
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
        if not self.enable_sqlite_fallback:
            raise RuntimeError("DATABASE_URL is required when SQLite fallback is disabled.")
        sqlite_path = Path(self.sqlite_path)
        if not sqlite_path.is_absolute():
            base_dir = self.project_root if (self.project_root / "backend").exists() else Path(__file__).resolve().parents[1]
            sqlite_path = base_dir / sqlite_path
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{sqlite_path.as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
