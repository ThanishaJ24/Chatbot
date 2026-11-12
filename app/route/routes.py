from fastapi import APIRouter, HTTPException, Query

from app.services.document_processing import (
    list_source_files,
    parse_source_documents,
    preview_documents,
    preview_nodes,
    process_documents,
)
from app.services.retrieval import answer_question_from_index
from app.services.vector_indexing import index_documents_in_pinecone
from app.schemas import QueryRequest

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/documents/metadata")
def document_metadata_api(recursive: bool = False) -> dict:
    files = list_source_files(recursive=recursive)
    return {
        "file_count": len(files),
        "files": files,
    }


@router.get("/documents/parse")
def parse_documents_api(
    preview_limit: int = Query(default=3, ge=0, le=20),
    text_limit: int = Query(default=700, ge=100, le=3000),
    num_files_limit: int | None = Query(default=None, ge=1, le=100),
    file_offset: int = Query(default=0, ge=0),
    recursive: bool = False,
) -> dict:
    parsed = parse_source_documents(
        recursive=recursive,
        num_files_limit=num_files_limit,
        file_offset=file_offset,
    )
    return {
        **parsed.summary(),
        "num_files_limit": num_files_limit,
        "file_offset": file_offset,
        "preview": preview_documents(
            parsed.documents,
            limit=preview_limit,
            text_limit=text_limit,
        ),
    }


@router.get("/documents/process")
def process_document_api(
    chunk_size: int = Query(default=1024, ge=128, le=8192),
    chunk_overlap: int = Query(default=128, ge=0, le=2048),
    preview_limit: int = Query(default=5, ge=0, le=20),
    num_files_limit: int | None = Query(default=None, ge=1, le=100),
    file_offset: int = Query(default=0, ge=0),
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
        file_offset=file_offset,
    )
    return {
        **processed.summary(),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_files_limit": num_files_limit,
        "file_offset": file_offset,
        "preview": preview_nodes(processed.nodes, limit=preview_limit),
    }


@router.post("/documents/index")
def index_documents_api(
    chunk_size: int = Query(default=1024, ge=128, le=8192),
    chunk_overlap: int = Query(default=128, ge=0, le=2048),
    num_files_limit: int | None = Query(default=None, ge=1, le=100),
    file_offset: int = Query(default=0, ge=0),
    namespace: str | None = Query(default=None, min_length=1, max_length=80),
    recursive: bool = False,
) -> dict:
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be smaller than chunk_size",
        )

    try:
        indexed = index_documents_in_pinecone(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            recursive=recursive,
            num_files_limit=num_files_limit,
            file_offset=file_offset,
            namespace=namespace,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        **indexed.summary(),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_files_limit": num_files_limit,
        "file_offset": file_offset,
    }


@router.post("/documents/query")
def query_documents_api(request: QueryRequest) -> dict:
    try:
        answer = answer_question_from_index(
            question=request.question,
            top_k=request.top_k,
            namespace=request.namespace,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return answer.as_dict()
