from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import backend.app.models.activity

from app.api.v1 import router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Runify Backend!"}

@app.get("/api/status")
async def get_status():
    return {"status": "online", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
