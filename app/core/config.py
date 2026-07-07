from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://runify:runify@localhost:5432/runify"
    secret_key: str = "dev-secret-change-in-production"
    session_expire_days: int = 30
    storage_dir: Path = Path("./storage")
    debug: bool = True
    auto_create_tables: bool = True
    sync_uploads: bool = False

    garmin_client_id: str = ""
    garmin_client_secret: str = ""
    wahoo_client_id: str = ""
    wahoo_client_secret: str = ""
    google_client_id: str = ""
    apple_client_id: str = ""

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def exports_dir(self) -> Path:
        return self.storage_dir / "exports"


@lru_cache
def get_settings() -> Settings:
    return Settings()
