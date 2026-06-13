from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lib.api import router as dog_router
from lib.config import settings
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dog Breed Recognition Backend",
    version="0.1.0",
    description="Backend API for TP2 dog breed detection and classification system.",
)

_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dog_router)

@app.get("/health", tags=["health"])
async def health() -> dict[str, object]:
    checkpoints = {
        "resnet18_finetuned": (settings.model_path / settings.resnet18_model_name).exists(),
        "cnn_custom": (settings.model_path / settings.cnn_custom_model_name).exists(),
    }
    missing = [name for name, exists in checkpoints.items() if not exists]
    if missing:
        logger.warning(
            "Checkpoints no encontrados en %s: %s (se generan al entrenar en la Etapa 2)",
            settings.model_path,
            ", ".join(missing),
        )
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "yolo_model": settings.yolo_model,
        "use_pgvector": settings.use_pgvector,
        "checkpoints": checkpoints,
    }
