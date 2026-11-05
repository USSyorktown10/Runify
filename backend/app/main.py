from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import backend.app.models.activity

from app.api.v1 import router

application = FastAPI(
    title="Runify API",
    description="Runify Release 1 — Cool Thing",
    version="1.0.0",
)

@application.get("/health")
def health():
    return {"status": "ok"}

app = application