"""
Create (or extend) all database tables and add new columns safely.

Run from the project root:
    python backend/create_tables.py
"""
import sys
from pathlib import Path

# Ensure the project root is on the Python path when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import inspect, text
from backend.app.database.base import Base
from backend.app.database.connection import engine

# Import every model so SQLAlchemy knows about all tables.
from backend.app.models.dataset import Dataset          # noqa: F401
from backend.app.models.generated_image import GeneratedImage  # noqa: F401
from backend.app.models.generation_job import GenerationJob    # noqa: F401

Base.metadata.create_all(bind=engine)

def auto_upgrade_columns():
    """Safely add new multi-view and deformation columns if they are missing in existing tables."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        # Check generated_images columns
        if "generated_images" in inspector.get_table_names():
            existing_cols = {c["name"] for c in inspector.get_columns("generated_images")}
            col_definitions = [
                ("patient_id", "VARCHAR(50)"),
                ("viewpoint", "VARCHAR(50)"),
                ("viewpoint_name", "VARCHAR(100)"),
                ("deformation_status", "VARCHAR(50) DEFAULT 'NOT_STARTED'"),
                ("deformed_image_path", "VARCHAR(500)"),
                ("mask_image_path", "VARCHAR(500)"),
            ]
            for col_name, col_type in col_definitions:
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE generated_images ADD COLUMN {col_name} {col_type}"))
                    print(f"Added column generated_images.{col_name}")

        # Check datasets columns
        if "datasets" in inspector.get_table_names():
            existing_cols = {c["name"] for c in inspector.get_columns("datasets")}
            col_definitions = [
                ("patient_id", "VARCHAR(50)"),
                ("custom_prompt", "VARCHAR(4000)"),
                ("viewpoint_prompts", "VARCHAR(8000)"),
            ]
            for col_name, col_type in col_definitions:
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE datasets ADD COLUMN {col_name} {col_type}"))
                    print(f"Added column datasets.{col_name}")

try:
    auto_upgrade_columns()
except Exception as exc:
    print(f"Column check notice: {exc}")

print("Database tables created / verified successfully.")
print("Tables managed:")
for table_name in Base.metadata.tables:
    print(f"  • {table_name}")
