from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.app.api.routes.datasets import router as dataset_router
from backend.app.api.routes.generation import router as generation_router
from backend.app.api.routes.images import router as image_router
from backend.app.core.exceptions import AppException

app = FastAPI(
    title="Synthetic APAC Patient Face Dataset Generator",
    version="1.0.0",
)


@app.exception_handler(AppException)
async def app_exception_handler(
    request: Request,
    exc: AppException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
        },
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
app.include_router(image_router)