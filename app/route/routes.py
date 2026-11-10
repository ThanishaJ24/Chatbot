from fastapi import APIRouter, HTTPException, Query

from app.services.document_processing import (
    list_source_files,
    preview_nodes,
    process_documents,
)

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/")
def api_home() -> dict[str, str]:
    return {"message": "Portfolio bot API is running"}


@router.get("/documents/metadata")
def document_metadata_api(recursive: bool = False) -> dict:
    files = list_source_files(recursive=recursive)
    return {
        "file_count": len(files),
        "files": files,
    }


@router.get("/documents/process")
def process_document_api(
    chunk_size: int = Query(default=1024, ge=128, le=8192),
    chunk_overlap: int = Query(default=128, ge=0, le=2048),
    preview_limit: int = Query(default=5, ge=0, le=20),
    num_files_limit: int | None = Query(default=None, ge=1, le=100),
    recursive: bool = False,
) -> dict:
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be smaller than chunk_size",
        )

    processed = process_documents(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        recursive=recursive,
        num_files_limit=num_files_limit,
    )
    return {
        **processed.summary(),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_files_limit": num_files_limit,
        "preview": preview_nodes(processed.nodes, limit=preview_limit),
    }
