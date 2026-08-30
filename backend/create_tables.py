from app.database.base import Base
from app.database.connection import engine

# Import models so SQLAlchemy knows about them
from app.models.dataset import Dataset


Base.metadata.create_all(bind=engine)

print("Database tables created successfully.")