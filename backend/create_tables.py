from backend.app.database.base import Base
from backend.app.database.connection import engine
from backend.app.models.dataset import Dataset


Base.metadata.create_all(bind=engine)

print("Database tables created successfully.")