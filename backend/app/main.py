from fastapi import FastAPI

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