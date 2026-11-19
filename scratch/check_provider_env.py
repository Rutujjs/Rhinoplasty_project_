import sys
sys.path.insert(0, ".")
from backend.app.core.config import settings

print("effective_provider:", settings.effective_provider)
print("GENERATION_PROVIDER:", settings.GENERATION_PROVIDER)
print("RUN_EMBEDDED_WORKER:", settings.RUN_EMBEDDED_WORKER)
