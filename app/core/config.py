from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Runify"
    DATABASE_URL: str 
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    SESSION_EXPIRE_DAYS: int = 30
    STORAGE_DIR: str = "./storage"
    DEBUG: bool = True

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings() # type: ignore