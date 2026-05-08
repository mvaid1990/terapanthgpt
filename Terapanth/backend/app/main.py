# app/main.py
# FastAPI application entry point. Registers routers and global error handler.
# Run with: uvicorn app.main:app --reload

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.chat.router import router as chat_router
from app.core.exceptions import (
    ConversationNotFoundError,
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentTooLargeError,
    LLMError,
    TerapanthError,
)
from app.ingestion.router import router as ingestion_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Terapanth Learning API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(ingestion_router)


@app.exception_handler(DocumentNotFoundError)
@app.exception_handler(ConversationNotFoundError)
async def not_found_handler(request: Request, exc: TerapanthError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(DocumentTooLargeError)
async def too_large_handler(request: Request, exc: DocumentTooLargeError):
    return JSONResponse(status_code=413, content={"detail": str(exc)})


@app.exception_handler(DocumentProcessingError)
async def processing_error_handler(request: Request, exc: DocumentProcessingError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
async def health():
    return {"status": "ok"}
