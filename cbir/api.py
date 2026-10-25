"""
FastAPI backend for the CBIR Image Search Engine.

Start with:
    uvicorn cbir.api:app --reload --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health
GET  /api/datasets
GET  /api/datasets/{dataset_id}/stats
POST /api/index
GET  /api/index/status
POST /api/search/text
POST /api/search/image
POST /api/search/combined
POST /api/search/upload
GET  /api/images/{image_id}
GET  /api/descriptors/types
GET  /api/metrics
GET  /api/categories/{dataset_id}
"""

import os
import time
import logging
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cbir.config import settings
from cbir.database import DatabaseManager
from cbir.descriptors import DESCRIPTOR_TYPES
from cbir.distances.metrics import get_available_metrics
from cbir.indexing.indexer import ImageIndexer
from cbir.search import SearchEngine

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_START_TIME = time.time()

# ---------------------------------------------------------------------------
# App factory with lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared services on startup, clean up on shutdown."""
    logger.info("Starting CBIR services …")

    # Ensure data directories exist
    os.makedirs(settings.FEATURES_CACHE_DIR, exist_ok=True)
    os.makedirs(settings.INDEX_DIR, exist_ok=True)
    os.makedirs(settings.STATIC_DIR, exist_ok=True)

    db = DatabaseManager(settings.DB_PATH)
    indexer = ImageIndexer(db, cache_dir=settings.FEATURES_CACHE_DIR)
    search_engine = SearchEngine(
        db, indexer, static_base_url=f"http://localhost:8000/static"
    )

    app.state.db = db
    app.state.indexer = indexer
    app.state.search_engine = search_engine

    logger.info("CBIR services ready.")
    yield
    logger.info("Shutting down CBIR services.")
    db.close()


app = FastAPI(
    title="CBIR Image Search API",
    version="1.0.0",
    description="Content-Based Image Retrieval engine with MPEG-7, colour, LBP and HOG descriptors.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the entire data directory as static files so images are accessible
app.mount(
    "/static",
    StaticFiles(directory=settings.STATIC_DIR, check_dir=False),
    name="static",
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


class DatasetInfo(BaseModel):
    name: str
    total_images: int
    categories_count: int
    categories_list: List[str]
    descriptor_coverage: Dict[str, int]


class SearchResult(BaseModel):
    image_id: str
    file_path: str
    url: str
    category: str
    distance: Optional[float] = None
    score: Optional[float] = None
    rank: int = 0
    match_type: Optional[str] = None


class TextSearchRequest(BaseModel):
    query: str
    dataset: Optional[str] = None
    max_results: int = Field(default=20, ge=1, le=200)
    fuzzy_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class TextSearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    time_ms: float
    fuzzy: bool = False


class ImageSearchRequest(BaseModel):
    query_image_id: Optional[str] = None
    descriptor_type: str = "color_rgb"
    distance_metric: str = "l2"
    max_results: int = Field(default=20, ge=1, le=200)
    dataset: Optional[str] = None
    use_index: bool = True
    descriptor_weights: Optional[Dict[str, float]] = None


class ImageSearchResponse(BaseModel):
    query_image_id: Optional[str] = None
    descriptor_type: str
    distance_metric: str
    results: List[SearchResult]
    total: int
    time_ms: float
    search_type: str
    comparisons: int


class CombinedSearchRequest(BaseModel):
    text_query: str
    query_image_id: str
    descriptor_type: str = "color_rgb"
    distance_metric: str = "l2"
    text_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    max_results: int = Field(default=20, ge=1, le=200)


class IndexRequest(BaseModel):
    dataset: str
    descriptor_types: List[str]
    force_reindex: bool = False


class IndexStatusResponse(BaseModel):
    dataset: str
    status: str
    total_images: int
    processed_images: int
    errors: int
    progress_percent: float


class ImageDetail(BaseModel):
    image_id: str
    file_path: str
    url: str
    category: str
    dataset: str
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    available_descriptors: List[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@app.get("/api/datasets", response_model=List[DatasetInfo])
def get_datasets(request: Request):
    db: DatabaseManager = request.app.state.db
    results = []
    for ds_name in ["objects", "textures"]:
        stats = db.get_dataset_stats(ds_name)
        if stats.get("total_images", 0) == 0:
            continue
        cats = db.get_all_categories(ds_name)
        results.append(DatasetInfo(
            name=ds_name,
            total_images=stats["total_images"],
            categories_count=len(cats),
            categories_list=sorted(cats),
            descriptor_coverage=stats.get("descriptors", {}),
        ))
    return results


@app.get("/api/datasets/{dataset_id}/stats", response_model=DatasetInfo)
def get_dataset_stats(dataset_id: str, request: Request):
    db: DatabaseManager = request.app.state.db
    stats = db.get_dataset_stats(dataset_id)
    if stats.get("total_images", 0) == 0:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found or empty.")
    cats = db.get_all_categories(dataset_id)
    return DatasetInfo(
        name=dataset_id,
        total_images=stats["total_images"],
        categories_count=len(cats),
        categories_list=sorted(cats),
        descriptor_coverage=stats.get("descriptors", {}),
    )


@app.post("/api/index")
def trigger_index(req: IndexRequest, request: Request):
    """Start (synchronous) indexing. For large datasets, run in background via CLI."""
    indexer: ImageIndexer = request.app.state.indexer
    db: DatabaseManager = request.app.state.db

    dataset_paths = {
        "objects": settings.DATASET_B_OBJECTS_PATH,
        "textures": settings.DATASET_B_TEXTURES_PATH,
    }
    image_dir = dataset_paths.get(req.dataset)
    if not image_dir or not os.path.isdir(image_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{req.dataset}' path not found: {image_dir}"
        )

    # Validate descriptor types
    unknown = [d for d in req.descriptor_types if d not in DESCRIPTOR_TYPES]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown descriptor types: {unknown}. Valid: {DESCRIPTOR_TYPES}"
        )

    stats = indexer.index_dataset(
        req.dataset, image_dir, req.descriptor_types, req.force_reindex
    )
    return {"status": "completed", "stats": stats}


@app.get("/api/index/status", response_model=List[IndexStatusResponse])
def index_status(request: Request):
    db: DatabaseManager = request.app.state.db
    all_status = db.get_indexing_status()
    results = []
    for ds, row in all_status.items():
        total = row.get("total_images", 0) or 0
        processed = row.get("processed_images", 0) or 0
        pct = (processed / total * 100.0) if total > 0 else 0.0
        results.append(IndexStatusResponse(
            dataset=ds,
            status=row.get("status", "unknown"),
            total_images=total,
            processed_images=processed,
            errors=row.get("errors", 0) or 0,
            progress_percent=round(pct, 1),
        ))
    return results


@app.post("/api/search/text", response_model=TextSearchResponse)
def search_text(req: TextSearchRequest, request: Request):
    engine: SearchEngine = request.app.state.search_engine
    res = engine.text_search(req.query, req.dataset, req.max_results,
                             fuzzy_threshold=req.fuzzy_threshold)
    return TextSearchResponse(
        query=res["query"],
        results=[SearchResult(**r) for r in res["results"]],
        total=res["total"],
        time_ms=res["time_ms"],
        fuzzy=res.get("fuzzy", False),
    )


@app.post("/api/search/image", response_model=ImageSearchResponse)
def search_image(req: ImageSearchRequest, request: Request):
    engine: SearchEngine = request.app.state.search_engine
    if not req.query_image_id:
        raise HTTPException(status_code=400, detail="query_image_id is required.")
    try:
        res = engine.visual_search(
            query_image_id=req.query_image_id,
            descriptor_type=req.descriptor_type,
            distance_metric=req.distance_metric,
            max_results=req.max_results,
            dataset=req.dataset,
            use_index=req.use_index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ImageSearchResponse(
        query_image_id=req.query_image_id,
        descriptor_type=req.descriptor_type,
        distance_metric=req.distance_metric,
        results=[SearchResult(**r) for r in res["results"]],
        total=res["total"],
        time_ms=res["time_ms"],
        search_type=res["search_type"],
        comparisons=res["comparisons"],
    )


@app.post("/api/search/combined")
def search_combined(req: CombinedSearchRequest, request: Request):
    engine: SearchEngine = request.app.state.search_engine
    try:
        res = engine.combined_search(
            text_query=req.text_query,
            query_image_id=req.query_image_id,
            descriptor_type=req.descriptor_type,
            distance_metric=req.distance_metric,
            text_weight=req.text_weight,
            max_results=req.max_results,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return res


@app.post("/api/search/upload", response_model=ImageSearchResponse)
async def search_upload(
    request: Request,
    file: UploadFile = File(...),
    descriptor_type: str = Query("color_rgb"),
    distance_metric: str = Query("l2"),
    max_results: int = Query(20),
):
    """Search by uploading an external image (not in the database)."""
    engine: SearchEngine = request.app.state.search_engine

    suffix = os.path.splitext(file.filename or ".jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        res = engine.visual_search(
            query_image_path=tmp_path,
            descriptor_type=descriptor_type,
            distance_metric=distance_metric,
            max_results=max_results,
            use_index=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return ImageSearchResponse(
        query_image_id=None,
        descriptor_type=descriptor_type,
        distance_metric=distance_metric,
        results=[SearchResult(**r) for r in res["results"]],
        total=res["total"],
        time_ms=res["time_ms"],
        search_type=res["search_type"],
        comparisons=res["comparisons"],
    )


@app.get("/api/images/{image_id}", response_model=ImageDetail)
def get_image(image_id: str, request: Request):
    db: DatabaseManager = request.app.state.db
    img = db.get_image(image_id)
    if not img:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found.")

    # Fetch which descriptors are available for this image
    available = []
    for desc in DESCRIPTOR_TYPES:
        if db.descriptor_exists(image_id, desc):
            available.append(desc)

    file_path = img.get("file_path", "")

    # Compute URL relative to STATIC_DIR
    abs_path = os.path.normpath(os.path.abspath(file_path))
    static_root = os.path.normpath(os.path.abspath(settings.STATIC_DIR))
    try:
        rel = os.path.relpath(abs_path, static_root).replace("\\", "/")
    except ValueError:
        rel = os.path.basename(file_path)
    url = f"http://localhost:8000/static/{rel}"

    return ImageDetail(
        image_id=img["image_id"],
        file_path=file_path,
        url=url,
        category=img.get("category") or "",
        dataset=img.get("dataset") or "",
        width=img.get("width"),
        height=img.get("height"),
        file_size=img.get("file_size"),
        available_descriptors=available,
    )


@app.get("/api/descriptors/types")
def get_descriptor_types():
    return DESCRIPTOR_TYPES


@app.get("/api/metrics")
def get_metrics():
    return get_available_metrics()


@app.get("/api/categories/{dataset_id}")
def get_categories(dataset_id: str, request: Request):
    db: DatabaseManager = request.app.state.db
    return sorted(db.get_all_categories(dataset_id))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("cbir.api:app", host="0.0.0.0", port=8000, reload=True)
