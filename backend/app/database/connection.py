import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from backend.app.core.config import BASE_DIR, settings

logger = logging.getLogger(__name__)


def _resolve_sqlite_url(url: str) -> str:
    """Normalize sqlite URL to an absolute path rooted at BASE_DIR if relative."""
    prefix = "sqlite:///"
    if url.startswith(prefix):
        raw_path = url[len(prefix):]
        if raw_path == ":memory:":
            return url
        p = Path(raw_path)
        if not p.is_absolute():
            clean = raw_path.replace("\\", "/")
            if clean.startswith("./"):
                clean = clean[2:]
            elif clean.startswith("../"):
                clean = clean.replace("../", "")
            p = (BASE_DIR / clean).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{p.as_posix()}"
    return url


def _init_engine():
    db_url = settings.DATABASE_URL.strip() if settings.DATABASE_URL else ""
    if not db_url:
        db_url = f"sqlite:///{(BASE_DIR / 'storage' / 'apac_face.db').as_posix()}"

    if db_url.startswith("sqlite"):
        resolved_url = _resolve_sqlite_url(db_url)
        return create_engine(
            resolved_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )

    # For PostgreSQL / other relational databases: test connection
    try:
        eng = create_engine(db_url, echo=False)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return eng
    except Exception as exc:
        fallback_path = BASE_DIR / "storage" / "apac_face.db"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_url = f"sqlite:///{fallback_path.as_posix()}"
        logger.warning(
            "Primary database connection (%s) failed: %s. "
            "Falling back to local SQLite database at %s.",
            db_url,
            exc,
            fallback_path,
        )
        print(
            f"[WARNING] Database connection failed ({exc}). "
            f"Falling back to local SQLite: {fallback_path}"
        )
        return create_engine(
            fallback_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )


engine = _init_engine()

def _auto_migrate_sqlite(eng):
    """Ensure newly added columns exist in existing SQLite databases."""
    try:
        with eng.connect() as conn:
            # Check datasets columns
            res = conn.execute(text("PRAGMA table_info(datasets)")).fetchall()
            if res:
                existing_cols = {row[1] for row in res}
                cols_to_add = [
                    ("morphology_type", "VARCHAR(50) DEFAULT 'saddle'"),
                    ("deformation_strength", "FLOAT DEFAULT 0.6"),
                    ("deviated_direction", "VARCHAR(20) DEFAULT 'right'"),
                ]
                for col_name, col_def in cols_to_add:
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE datasets ADD COLUMN {col_name} {col_def}"))
                conn.commit()

            # Check generated_images columns
            res_img = conn.execute(text("PRAGMA table_info(generated_images)")).fetchall()
            if res_img:
                existing_img_cols = {row[1] for row in res_img}
                img_cols_to_add = [
                    ("deformation_status", "VARCHAR(50) DEFAULT 'NOT_STARTED'"),
                    ("deformed_image_path", "VARCHAR(500)"),
                    ("mask_image_path", "VARCHAR(500)"),
                    ("deformation_type", "VARCHAR(50)"),
                    ("deformation_strength", "FLOAT"),
                    ("deformation_direction", "VARCHAR(20)"),
                    ("deformed_at", "DATETIME"),
                ]
                for col_name, col_def in img_cols_to_add:
                    if col_name not in existing_img_cols:
                        conn.execute(text(f"ALTER TABLE generated_images ADD COLUMN {col_name} {col_def}"))
                conn.commit()
    except Exception as exc:
        logger.debug("Auto-migrate SQLite columns skipped: %s", exc)

# Ensure database tables exist
try:
    from backend.app.database.base import Base
    import backend.app.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _auto_migrate_sqlite(engine)
except Exception as exc:
    logger.warning("Could not auto-create database tables on engine init: %s", exc)
