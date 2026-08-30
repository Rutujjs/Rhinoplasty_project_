from sqlalchemy.orm import sessionmaker, Session

from app.database.connection import engine


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


def get_db():
    db: Session = SessionLocal()

    try:
        yield db
    finally:
        db.close()