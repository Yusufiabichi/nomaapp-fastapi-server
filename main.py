

import os

from fastapi import FastAPI
from app.api.router import api_router
from app.models.loader import load_models
import logging
import uvicorn

app = FastAPI(
    title="NomaApp AI Service",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    logging.info("Starting NomaApp AI Service")
    load_models()
    logging.info("AI models loaded successfully")

app.include_router(api_router, prefix="/ai")

@app.get("/")
def root():
    return {"message": "FastAPI server is running 🚀"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)