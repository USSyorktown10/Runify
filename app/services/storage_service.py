from pathlib import Path

from app.core.config import get_settings


class StorageService:
    def __init__(self) -> None:
        settings = get_settings()
        self.uploads_dir = settings.uploads_dir
        self.exports_dir = settings.exports_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, upload_id: str, filename: str, content: bytes) -> str:
        dest_dir = self.uploads_dir / upload_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        dest.write_bytes(content)
        return str(dest)

    def read_file(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def export_path(self, name: str) -> Path:
        return self.exports_dir / name


storage_service = StorageService()
