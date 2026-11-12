from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.services.document_processing import process_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_INDEX_NAME = "portfolio-bot-rag"
DEFAULT_NAMESPACE = "documents"


@dataclass(slots=True)
class PineconeIndexConfig:
    api_key: str
    index_name: str = DEFAULT_INDEX_NAME
    namespace: str = DEFAULT_NAMESPACE
    cloud: str = "aws"
    region: str = "us-east-1"
    metric: str = "cosine"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION


@dataclass(slots=True)
class IndexedDocuments:
    document_count: int
    node_count: int
    files: list[str]
    index_name: str
    namespace: str
    embedding_model: str
    embedding_dimension: int

    def summary(self) -> dict[str, Any]:
        return {
            "document_count": self.document_count,
            "node_count": self.node_count,
            "files": self.files,
            "index_name": self.index_name,
            "namespace": self.namespace,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
        }


def index_documents_in_pinecone(
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    recursive: bool = False,
    num_files_limit: int | None = None,
    file_offset: int = 0,
    namespace: str | None = None,
) -> IndexedDocuments:
    config = _pinecone_config(namespace=namespace)
    processed = process_documents(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        recursive=recursive,
        num_files_limit=num_files_limit,
        file_offset=file_offset,
    )

    pinecone_index = _get_or_create_pinecone_index(config)
    vector_store = _pinecone_vector_store(
        pinecone_index=pinecone_index,
        namespace=config.namespace,
    )
    storage_context = _storage_context(vector_store)
    embed_model = _openai_embedding_model(config)

    vector_index_class = _vector_store_index_class()
    vector_index_class(
        nodes=processed.nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    summary = processed.summary()
    return IndexedDocuments(
        document_count=summary["document_count"],
        node_count=summary["node_count"],
        files=summary["files"],
        index_name=config.index_name,
        namespace=config.namespace,
        embedding_model=config.embedding_model,
        embedding_dimension=config.embedding_dimension,
    )


def _pinecone_config(namespace: str | None = None) -> PineconeIndexConfig:
    _load_dotenv_if_available()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment")

    return PineconeIndexConfig(
        api_key=api_key,
        index_name=os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME),
        namespace=namespace or os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
        metric=os.getenv("PINECONE_METRIC", "cosine"),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_dimension=int(
            os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION))
        ),
    )


def _get_or_create_pinecone_index(
    config: PineconeIndexConfig,
    create_if_missing: bool = True,
) -> Any:
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError as exc:
        raise RuntimeError(
            "Missing Pinecone dependency. Run: pip install -r requirements.txt"
        ) from exc

    client = Pinecone(api_key=config.api_key)
    index_names = client.list_indexes().names()

    if config.index_name not in index_names:
        if not create_if_missing:
            raise RuntimeError(f"Pinecone index not found: {config.index_name}")

        client.create_index(
            name=config.index_name,
            dimension=config.embedding_dimension,
            metric=config.metric,
            spec=ServerlessSpec(cloud=config.cloud, region=config.region),
        )

    return client.Index(config.index_name)


def _pinecone_vector_store(pinecone_index: Any, namespace: str) -> Any:
    try:
        from llama_index.vector_stores.pinecone import PineconeVectorStore
    except ImportError as exc:
        raise RuntimeError(
            "Missing LlamaIndex Pinecone integration. "
            "Run: pip install -r requirements.txt"
        ) from exc

    return PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace,
    )


def _storage_context(vector_store: Any) -> Any:
    from llama_index.core import StorageContext

    return StorageContext.from_defaults(vector_store=vector_store)


def _openai_embedding_model(config: PineconeIndexConfig) -> Any:
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "Missing LlamaIndex OpenAI embedding integration. "
            "Run: pip install -r requirements.txt"
        ) from exc

    return OpenAIEmbedding(model=config.embedding_model)


def _vector_store_index_class() -> Any:
    from llama_index.core import VectorStoreIndex

    return VectorStoreIndex


def _load_dotenv_if_available() -> None:
    load_dotenv(dotenv_path=ENV_PATH)
