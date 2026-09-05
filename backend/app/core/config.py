from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Synthetic APAC Patient Face Dataset Generator"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str

    # Storage
    STORAGE_DIR: str = str(BASE_DIR / "storage")
    GENERATED_IMAGES_DIR: str = str(
        BASE_DIR / "storage" / "generated"
    )

    # Authentication
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # Generation
    GENERATION_MODEL: str = "MOCK"
    IMAGE_WIDTH: int = 512
    IMAGE_HEIGHT: int = 512
    MAX_GENERATION_ATTEMPTS: int = 100

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()