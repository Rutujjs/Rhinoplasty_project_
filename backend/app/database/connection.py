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

# Ensure database tables exist
try:
    from backend.app.database.base import Base
    import backend.app.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
except Exception as exc:
    logger.warning("Could not auto-create database tables on engine init: %s", exc)
