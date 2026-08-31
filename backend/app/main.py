from fastapi import FastAPI

from app.api.routes.datasets import router as dataset_router
from app.api.routes.generation import router as generation_router

app = FastAPI(
    title="Synthetic APAC Patient Face Dataset Generator",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Synthetic APAC Patient Face Dataset Generator API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


app.include_router(dataset_router)
app.include_router(generation_router)