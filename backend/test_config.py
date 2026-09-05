from backend.app.core.config import settings


print("Application:", settings.APP_NAME)
print("Environment:", settings.ENVIRONMENT)
print("Generation model:", settings.GENERATION_MODEL)
print("Image size:", settings.IMAGE_WIDTH, "x", settings.IMAGE_HEIGHT)
print("Max generation attempts:", settings.MAX_GENERATION_ATTEMPTS)
print("Storage:", settings.STORAGE_DIR)