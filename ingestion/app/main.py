import io
import json
import os
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urlparse

import requests
import weaviate
from fastapi import Depends, FastAPI, Header, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from minio import Minio
from pydantic import BaseModel, Field

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from faster_whisper import WhisperModel
from pdf2image import convert_from_bytes
from pypdf import PdfReader
import pytesseract
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi
from yt_dlp import YoutubeDL


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


WEAVIATE_URL = _env("WEAVIATE_URL")

STORAGE_BACKEND = _env("STORAGE_BACKEND", "minio").lower()  # minio|local
ARTIFACTS_DIR = _env("ARTIFACTS_DIR", "/data/artifacts")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "religious-rag")

OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL")
EMBED_MODEL = _env("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = _env("CHAT_MODEL", "llama3.1:8b")
INGESTION_API_KEY = _env("INGESTION_API_KEY")
SQLITE_PATH = _env("SQLITE_PATH", "/data/app.db")

OCR_ENABLED = _env("OCR_ENABLED", "true").lower() == "true"
OCR_MAX_PAGES = int(_env("OCR_MAX_PAGES", "5"))
WHISPER_MODEL_SIZE = _env("WHISPER_MODEL_SIZE", "small")


app = FastAPI()


def require_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != INGESTION_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
              id TEXT PRIMARY KEY,
              source_type TEXT NOT NULL,
              title TEXT,
              source_uri TEXT,
              status TEXT NOT NULL,
              tags_json TEXT NOT NULL,
              minio_prefix TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              approved_at INTEGER
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def minio_client() -> Minio:
    if not MINIO_ENDPOINT or not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise RuntimeError("MinIO env vars are not set")
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def ensure_bucket() -> None:
    if STORAGE_BACKEND != "minio":
        return
    client = minio_client()
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)


def ensure_artifacts_dir() -> None:
    if STORAGE_BACKEND != "local":
        return
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)


def _local_path(prefix: str, name: str) -> Path:
    # Keep same logical key layout as MinIO.
    return Path(ARTIFACTS_DIR) / prefix / name


def weaviate_client() -> weaviate.WeaviateClient:
    parsed = urlparse(WEAVIATE_URL)
    if not parsed.hostname:
        raise RuntimeError(f"Invalid WEAVIATE_URL: {WEAVIATE_URL}")
    http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    http_secure = parsed.scheme == "https"
    return weaviate.connect_to_custom(
        http_host=parsed.hostname,
        http_port=http_port,
        http_secure=http_secure,
        grpc_host=parsed.hostname,
        grpc_port=50051,
        grpc_secure=http_secure,
    )


def ensure_schema() -> None:
    client = weaviate_client()
    try:
        collections = client.collections.list_all(simple=True)
        if "Chunk" not in collections:
            client.collections.create(
                name="Chunk",
                vectorizer_config=None,
                properties=[
                    weaviate.classes.config.Property(name="source_id", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="title", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="chunk_index", data_type=weaviate.classes.config.DataType.INT),
                    weaviate.classes.config.Property(name="text", data_type=weaviate.classes.config.DataType.TEXT),
                ],
            )
    finally:
        client.close()


class SourceCreate(BaseModel):
    source_type: str = Field(description="youtube|pdf|epub|web|text")
    title: Optional[str] = None
    source_uri: Optional[str] = None
    text: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class SourceOut(BaseModel):
    id: str
    source_type: str
    title: Optional[str]
    source_uri: Optional[str]
    status: str
    tags: List[str]
    created_at: int
    updated_at: int
    approved_at: Optional[int]


def row_to_source(row: sqlite3.Row) -> SourceOut:
    return SourceOut(
        id=row["id"],
        source_type=row["source_type"],
        title=row["title"],
        source_uri=row["source_uri"],
        status=row["status"],
        tags=json.loads(row["tags_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        approved_at=row["approved_at"],
    )


def minio_put_text(prefix: str, name: str, text: str) -> None:
    if STORAGE_BACKEND == "local":
        p = _local_path(prefix, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return

    client = minio_client()
    data = text.encode("utf-8")
    client.put_object(
        MINIO_BUCKET,
        f"{prefix}/{name}",
        data=io.BytesIO(data),
        length=len(data),
        content_type="text/plain; charset=utf-8",
    )


def minio_put_bytes(prefix: str, name: str, data: bytes, content_type: str) -> None:
    if STORAGE_BACKEND == "local":
        p = _local_path(prefix, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return

    client = minio_client()
    client.put_object(
        MINIO_BUCKET,
        f"{prefix}/{name}",
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def minio_put_json(prefix: str, name: str, obj: Any) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

    if STORAGE_BACKEND == "local":
        p = _local_path(prefix, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return

    client = minio_client()
    client.put_object(
        MINIO_BUCKET,
        f"{prefix}/{name}",
        data=io.BytesIO(data),
        length=len(data),
        content_type="application/json",
    )


def minio_get_text(prefix: str, name: str) -> str:
    if STORAGE_BACKEND == "local":
        p = _local_path(prefix, name)
        return p.read_text(encoding="utf-8")

    client = minio_client()
    resp = client.get_object(MINIO_BUCKET, f"{prefix}/{name}")
    try:
        return resp.read().decode("utf-8")
    finally:
        resp.close()
        resp.release_conn()


def _download_bytes(url: str, timeout_s: int = 300) -> bytes:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.content


def _extract_youtube_id(url: str) -> str:
    # Basic support for youtu.be/<id> and youtube.com/watch?v=<id>
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.endswith("youtu.be"):
        vid = parsed.path.strip("/")
        if not vid:
            raise ValueError("Invalid youtu.be URL")
        return vid
    if "youtube.com" in host:
        qs = dict([p.split("=", 1) for p in parsed.query.split("&") if "=" in p])
        vid = qs.get("v")
        if not vid:
            raise ValueError("Missing v= parameter")
        return vid
    raise ValueError("Unsupported YouTube URL")


def _youtube_captions_text(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        raise
    lines = []
    for seg in transcript:
        t = (seg.get("text") or "").strip()
        if t:
            lines.append(t)
    return "\n".join(lines)


def _youtube_whisper_text(youtube_url: str) -> str:
    # Downloads audio with yt-dlp and transcribes with faster-whisper.
    with tempfile.TemporaryDirectory() as td:
        outtmpl = str(Path(td) / "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }
            ],
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        mp3_path = Path(td) / "audio.mp3"
        if not mp3_path.exists():
            raise RuntimeError("Audio download failed")

        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(mp3_path), vad_filter=True)
        lines: List[str] = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                lines.append(t)
        return "\n".join(lines)


def _pdf_extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = t.strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _pdf_ocr_text(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes)
    parts: List[str] = []
    for idx, img in enumerate(images[:OCR_MAX_PAGES]):
        text = pytesseract.image_to_string(img)
        text = (text or "").strip()
        if text:
            parts.append(text)
        if idx + 1 >= OCR_MAX_PAGES:
            break
    return "\n\n".join(parts)


def _epub_extract_text(epub_bytes: bytes) -> str:
    book = epub.read_epub(io.BytesIO(epub_bytes))
    parts: List[str] = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            txt = soup.get_text(" ")
            txt = (txt or "").strip()
            if txt:
                parts.append(txt)
    return "\n\n".join(parts)


def ollama_embed(text: str) -> List[float]:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data["embedding"]


def ollama_chat(messages: List[Dict[str, str]]) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": CHAT_MODEL, "messages": messages, "stream": False},
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


class IngestState(TypedDict, total=False):
    source_id: str
    minio_prefix: str
    title: Optional[str]
    text: str
    chunks: List[str]


splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)


def node_load_text(state: IngestState) -> IngestState:
    text = minio_get_text(state["minio_prefix"], "extracted.txt")
    return {**state, "text": text}


def node_chunk(state: IngestState) -> IngestState:
    docs = splitter.split_text(state["text"])
    return {**state, "chunks": docs}


def node_index(state: IngestState) -> IngestState:
    ensure_schema()
    client = weaviate_client()
    try:
        coll = client.collections.get("Chunk")
        for i, chunk in enumerate(state["chunks"]):
            vec = ollama_embed(chunk)
            coll.data.insert(
                properties={
                    "source_id": state["source_id"],
                    "title": state.get("title"),
                    "chunk_index": i,
                    "text": chunk,
                },
                vector=vec,
            )
    finally:
        client.close()
    return state


ingest_graph = StateGraph(IngestState)
ingest_graph.add_node("load_text", node_load_text)
ingest_graph.add_node("chunk", node_chunk)
ingest_graph.add_node("index", node_index)
ingest_graph.set_entry_point("load_text")
ingest_graph.add_edge("load_text", "chunk")
ingest_graph.add_edge("chunk", "index")
ingest_graph.add_edge("index", END)
ingest_runner = ingest_graph.compile()


@app.on_event("startup")
def _startup() -> None:
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    init_db()
    ensure_bucket()
    ensure_artifacts_dir()
    ensure_schema()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/admin/sources", response_model=SourceOut, dependencies=[Depends(require_api_key)])
def create_source(payload: SourceCreate) -> SourceOut:
    now = int(time.time())
    source_id = str(uuid.uuid4())
    minio_prefix = f"sources/{source_id}"

    extracted_text: str
    if payload.source_type == "text":
        if not payload.text:
            raise HTTPException(status_code=400, detail="text is required for source_type=text")
        extracted_text = payload.text

    elif payload.source_type == "youtube":
        if not payload.source_uri:
            raise HTTPException(status_code=400, detail="source_uri is required for source_type=youtube")
        try:
            video_id = _extract_youtube_id(payload.source_uri)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Captions-first
        transcript_method = "captions"
        try:
            extracted_text = _youtube_captions_text(video_id)
        except (TranscriptsDisabled, NoTranscriptFound):
            transcript_method = "whisper"
            extracted_text = _youtube_whisper_text(payload.source_uri)

        minio_put_json(
            minio_prefix,
            "youtube.json",
            {"video_id": video_id, "url": payload.source_uri, "method": transcript_method},
        )

    elif payload.source_type == "pdf":
        if not payload.source_uri:
            raise HTTPException(status_code=400, detail="source_uri is required for source_type=pdf")
        pdf_bytes = _download_bytes(payload.source_uri)
        minio_put_bytes(minio_prefix, "original.pdf", pdf_bytes, "application/pdf")

        extracted_text = _pdf_extract_text(pdf_bytes)
        if OCR_ENABLED and len(extracted_text.strip()) < 200:
            ocr_text = _pdf_ocr_text(pdf_bytes)
            if ocr_text.strip():
                extracted_text = ocr_text

    elif payload.source_type == "epub":
        if not payload.source_uri:
            raise HTTPException(status_code=400, detail="source_uri is required for source_type=epub")
        epub_bytes = _download_bytes(payload.source_uri)
        minio_put_bytes(minio_prefix, "original.epub", epub_bytes, "application/epub+zip")
        extracted_text = _epub_extract_text(epub_bytes)

    else:
        raise HTTPException(status_code=400, detail="Unsupported source_type")

    minio_put_text(minio_prefix, "extracted.txt", extracted_text)
    minio_put_json(
        minio_prefix,
        "metadata.json",
        {
            "id": source_id,
            "source_type": payload.source_type,
            "title": payload.title,
            "source_uri": payload.source_uri,
            "tags": payload.tags,
        },
    )

    conn = db()
    try:
        conn.execute(
            """
            INSERT INTO sources (id, source_type, title, source_uri, status, tags_json, minio_prefix, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                payload.source_type,
                payload.title,
                payload.source_uri,
                "awaiting_approval",
                json.dumps(payload.tags),
                minio_prefix,
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        assert row is not None
        return row_to_source(row)
    finally:
        conn.close()


@app.get("/admin/sources", response_model=List[SourceOut], dependencies=[Depends(require_api_key)])
def list_sources() -> List[SourceOut]:
    conn = db()
    try:
        rows = conn.execute("SELECT * FROM sources ORDER BY created_at DESC").fetchall()
        return [row_to_source(r) for r in rows]
    finally:
        conn.close()


@app.post("/admin/sources/{source_id}/approve", response_model=SourceOut, dependencies=[Depends(require_api_key)])
def approve_source(source_id: str) -> SourceOut:
    now = int(time.time())
    conn = db()
    try:
        row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        conn.execute(
            "UPDATE sources SET status = ?, approved_at = ?, updated_at = ? WHERE id = ?",
            ("approved", now, now, source_id),
        )
        conn.commit()
        row2 = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        assert row2 is not None
        return row_to_source(row2)
    finally:
        conn.close()


@app.post("/admin/sources/{source_id}/reject", response_model=SourceOut, dependencies=[Depends(require_api_key)])
def reject_source(source_id: str) -> SourceOut:
    now = int(time.time())
    conn = db()
    try:
        row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        conn.execute(
            "UPDATE sources SET status = ?, updated_at = ? WHERE id = ?",
            ("rejected", now, source_id),
        )
        conn.commit()
        row2 = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        assert row2 is not None
        return row_to_source(row2)
    finally:
        conn.close()


@app.post("/admin/sources/{source_id}/index", response_model=SourceOut, dependencies=[Depends(require_api_key)])
def index_source(source_id: str) -> SourceOut:
    conn = db()
    try:
        row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        if row["status"] != "approved":
            raise HTTPException(status_code=400, detail="Source must be approved before indexing")

        ingest_runner.invoke(
            {
                "source_id": source_id,
                "minio_prefix": row["minio_prefix"],
                "title": row["title"],
            }
        )

        now = int(time.time())
        conn.execute("UPDATE sources SET status = ?, updated_at = ? WHERE id = ?", ("indexed", now, source_id))
        conn.commit()
        row2 = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        assert row2 is not None
        return row_to_source(row2)
    finally:
        conn.close()


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = 0.2


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": CHAT_MODEL, "object": "model"},
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
def chat_completions(req: OpenAIChatRequest) -> Dict[str, Any]:
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message is required")
    query = user_messages[-1].content

    qvec = ollama_embed(query)

    client = weaviate_client()
    try:
        coll = client.collections.get("Chunk")
        res = coll.query.near_vector(near_vector=qvec, limit=5, return_properties=["source_id", "title", "chunk_index", "text"])
        chunks: List[Dict[str, Any]] = []
        for obj in res.objects:
            chunks.append(obj.properties)
    finally:
        client.close()

    context_lines: List[str] = []
    for c in chunks:
        context_lines.append(
            f"[source_id={c.get('source_id')} chunk={c.get('chunk_index')} title={c.get('title')}]\n{c.get('text')}"
        )
    context = "\n\n".join(context_lines)

    system = (
        "You are a religious assistant. Answer using the provided context when possible. "
        "If the context does not contain the answer, say you don't know. "
        "Cite sources by including the source_id and chunk index."
    )

    messages = [
        {"role": "system", "content": system + "\n\nCONTEXT:\n" + context},
    ]
    for m in req.messages:
        if m.role in {"system", "user", "assistant"}:
            messages.append({"role": m.role, "content": m.content})

    answer = ollama_chat(messages)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": CHAT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
    }
