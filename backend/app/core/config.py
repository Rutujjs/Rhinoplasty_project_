"""
Application settings loaded from environment variables / .env file.

All fields that existed before this change keep their original names and
defaults so that the existing .env and any running instance are unaffected.
New fields carry safe defaults so the app starts without requiring changes
to the .env file (except when actually enabling Kaggle batch generation).
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root  (synthetic/)
BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # Application                                                          #
    # ------------------------------------------------------------------ #
    APP_NAME: str = "Synthetic APAC Patient Face Dataset Generator"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # ------------------------------------------------------------------ #
    # Database                                                             #
    # ------------------------------------------------------------------ #
    DATABASE_URL: str  # required – must be set in .env

    # ------------------------------------------------------------------ #
    # Storage directories                                                  #
    # ------------------------------------------------------------------ #
    STORAGE_DIR: str = str(BASE_DIR / "storage")
    GENERATED_IMAGES_DIR: str = str(BASE_DIR / "storage" / "generated")

    # Temporary directory for raw Kaggle artifact downloads.
    ARTIFACT_DOWNLOAD_DIR: str = str(BASE_DIR / "storage" / "artifacts_tmp")

    # Validated images that are ready for export.
    VALIDATED_IMAGES_DIR: str = str(BASE_DIR / "storage" / "validated")

    # Rejected images kept for inspection (subject to retention policy).
    REJECTED_IMAGES_DIR: str = str(BASE_DIR / "storage" / "rejected")

    # Final ZIP exports.
    EXPORT_DIR: str = str(BASE_DIR / "storage" / "exports")

    # Frontend directory containing HTML/JS UI assets.
    FRONTEND_DIR: str = str(BASE_DIR / "frontend")

    # ------------------------------------------------------------------ #
    # Authentication (kept from original)                                 #
    # ------------------------------------------------------------------ #
    JWT_SECRET: str = "CHANGE_THIS_SECRET_IN_PRODUCTION"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # ------------------------------------------------------------------ #
    # Generation – provider selection                                      #
    # ------------------------------------------------------------------ #
    # MOCK   : deterministic placeholder PNG – no GPU required
    # KAGGLE_BATCH : submits a Kaggle notebook run and retrieves output
    GENERATION_PROVIDER: str = "MOCK"

    # Legacy alias kept for backward compatibility with existing .env files.
    # If both are set, GENERATION_PROVIDER takes precedence.
    GENERATION_MODEL: str = "MOCK"

    # Maximum images that may be requested in a single dataset.
    MAX_REQUESTED_IMAGES: int = 10_000

    # Preferred output dimensions (used in manifest and validation).
    IMAGE_WIDTH: int = 512
    IMAGE_HEIGHT: int = 512

    # Allowed output image formats (comma-separated, lowercase).
    ALLOWED_IMAGE_FORMATS: str = "png,jpg,jpeg,webp"

    # Maximum accepted file size per image in bytes (default 20 MB).
    MAX_IMAGE_FILE_BYTES: int = 20 * 1024 * 1024

    # Maximum generation attempts before a job is marked FAILED.
    MAX_GENERATION_ATTEMPTS: int = 100

    # How many images per Kaggle batch submission.
    BATCH_SIZE: int = 50

    # ------------------------------------------------------------------ #
    # Background worker                                                    #
    # ------------------------------------------------------------------ #
    # Whether to run the embedded background worker inside the FastAPI process.
    RUN_EMBEDDED_WORKER: bool = True

    # How often (seconds) the worker polls the job queue.
    WORKER_POLL_INTERVAL_SECONDS: int = 10

    # After this many seconds without a heartbeat, a job is considered
    # stale and eligible for re-claiming by another worker.
    JOB_LEASE_SECONDS: int = 300  # 5 minutes

    # How many times a job may be retried before being marked FAILED.
    JOB_MAX_RETRIES: int = 3

    # ------------------------------------------------------------------ #
    # Kaggle integration                                                   #
    # ------------------------------------------------------------------ #
    # Kaggle username – required when GENERATION_PROVIDER=KAGGLE_BATCH
    KAGGLE_USERNAME: str = ""

    # Path to kaggle.json credentials file.
    # Defaults to the standard Kaggle location.
    KAGGLE_CONFIG_DIR: str = str(Path.home() / ".kaggle")

    # Kaggle kernel/notebook slug in the form  "username/kernel-slug"
    KAGGLE_KERNEL_SLUG: str = ""

    # Local path to the .ipynb notebook file to push.
    # This file is pushed to Kaggle on every submit_batch() call.
    # Default points to the tested notebook in kaggle_notebook_check/.
    KAGGLE_NOTEBOOK_PATH: str = str(
        BASE_DIR / "kaggle_notebook_check" / "apac_face_generator.ipynb"
    )

    # Local path to the kernel-metadata.json for the notebook above.
    KAGGLE_KERNEL_METADATA_PATH: str = str(
        BASE_DIR / "kaggle_notebook_check" / "kernel-metadata.json"
    )

    # Slug of the private Kaggle dataset used to pass the manifest JSON
    # to the notebook.  Format: "username/dataset-slug".
    # This dataset must be created once manually (see setup instructions).
    KAGGLE_MANIFEST_DATASET_SLUG: str = ""

    # How often (seconds) the Kaggle adapter polls for notebook status.
    KAGGLE_POLL_INTERVAL_SECONDS: int = 30

    # Overall timeout (seconds) for a Kaggle notebook run.
    # Default 2 hours; increase for large batches.
    KAGGLE_RUN_TIMEOUT_SECONDS: int = 7_200

    # Maximum consecutive Kaggle poll errors before the job is failed.
    KAGGLE_MAX_POLL_ERRORS: int = 5

    # Seconds to wait after kernel_push before first status poll.
    # The Kaggle API may briefly show the previous run's status.
    KAGGLE_POST_PUSH_DELAY_SECONDS: int = 90

    # ------------------------------------------------------------------ #
    # Validation pipeline                                                  #
    # ------------------------------------------------------------------ #
    # Set to True to require face detection for every accepted image.
    # Requires opencv-python or equivalent installed.
    REQUIRE_FACE_DETECTION: bool = False

    # Set to True to enable near-duplicate detection (perceptual hash).
    ENABLE_DUPLICATE_DETECTION: bool = True

    # Maximum allowed perceptual-hash distance before images are considered
    # duplicates (0 = identical; 10 is a reasonable threshold).
    DUPLICATE_HASH_THRESHOLD: int = 10

    # ------------------------------------------------------------------ #
    # Export / retention                                                   #
    # ------------------------------------------------------------------ #
    # Days to keep rejected images and temporary artifacts before cleanup.
    ARTIFACT_RETENTION_DAYS: int = 7

    model_config = SettingsConfigDict(
        # The .env file lives in  backend/.env  (one level up from app/)
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def effective_provider(self) -> str:
        """
        Return the active generation provider name.

        GENERATION_PROVIDER takes priority.  If it is still the default
        'MOCK' but GENERATION_MODEL has been set to something else in an
        old .env, honour GENERATION_MODEL so existing deployments are not
        broken.
        """
        if self.GENERATION_PROVIDER != "MOCK":
            return self.GENERATION_PROVIDER
        if self.GENERATION_MODEL not in ("MOCK", ""):
            return self.GENERATION_MODEL
        return "MOCK"

    @property
    def allowed_formats(self) -> list[str]:
        """Return allowed image formats as a lowercase list."""
        return [
            fmt.strip().lower()
            for fmt in self.ALLOWED_IMAGE_FORMATS.split(",")
            if fmt.strip()
        ]

    @property
    def resolved_notebook_path(self) -> Path:
        """
        Return the absolute, verified Path to the Kaggle notebook.

        Handles:
        1. Absolute paths.
        2. Paths relative to current working directory.
        3. Paths relative to BASE_DIR (project root).
        4. Paths specified with leading '../' relative to backend/.
        """
        p = Path(self.KAGGLE_NOTEBOOK_PATH)
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p.resolve()
        if (BASE_DIR / self.KAGGLE_NOTEBOOK_PATH).exists():
            return (BASE_DIR / self.KAGGLE_NOTEBOOK_PATH).resolve()
        backend_dir = BASE_DIR / "backend"
        if (backend_dir / self.KAGGLE_NOTEBOOK_PATH).resolve().exists():
            return (backend_dir / self.KAGGLE_NOTEBOOK_PATH).resolve()
        clean = str(self.KAGGLE_NOTEBOOK_PATH).replace("../", "").replace("..\\", "")
        if (BASE_DIR / clean).exists():
            return (BASE_DIR / clean).resolve()
        # Default fallback to BASE_DIR joined path
        return (BASE_DIR / self.KAGGLE_NOTEBOOK_PATH).resolve()

    @property
    def resolved_kernel_metadata_path(self) -> Path:
        """
        Return the absolute, verified Path to kernel-metadata.json.
        """
        p = Path(self.KAGGLE_KERNEL_METADATA_PATH)
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p.resolve()
        if (BASE_DIR / self.KAGGLE_KERNEL_METADATA_PATH).exists():
            return (BASE_DIR / self.KAGGLE_KERNEL_METADATA_PATH).resolve()
        backend_dir = BASE_DIR / "backend"
        if (backend_dir / self.KAGGLE_KERNEL_METADATA_PATH).resolve().exists():
            return (backend_dir / self.KAGGLE_KERNEL_METADATA_PATH).resolve()
        clean = str(self.KAGGLE_KERNEL_METADATA_PATH).replace("../", "").replace("..\\", "")
        if (BASE_DIR / clean).exists():
            return (BASE_DIR / clean).resolve()
        return (BASE_DIR / self.KAGGLE_KERNEL_METADATA_PATH).resolve()

    @property
    def resolved_frontend_dir(self) -> Path:
        """
        Return the verified absolute Path to the frontend directory.
        """
        p = Path(self.FRONTEND_DIR)
        if p.is_absolute() and p.exists():
            return p
        if (BASE_DIR / "frontend").exists():
            return (BASE_DIR / "frontend").resolve()
        candidate = Path(__file__).resolve().parents[3] / "frontend"
        if candidate.exists():
            return candidate.resolve()
        if p.exists():
            return p.resolve()
        return (BASE_DIR / "frontend").resolve()


settings = Settings()