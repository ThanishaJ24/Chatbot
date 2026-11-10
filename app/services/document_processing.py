from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Iterable

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")


@dataclass(slots=True)
class ProcessedDocuments:
    documents: list[Document]
    nodes: list[Any]

    def summary(self) -> dict[str, Any]:
        file_names = sorted(
            {
                document.metadata.get("file_name", "unknown")
                for document in self.documents
            }
        )
        return {
            "document_count": len(self.documents),
            "node_count": len(self.nodes),
            "files": file_names,
        }


def extract_file_metadata(file_path: str) -> dict[str, Any]:
    path = Path(file_path)
    stat = path.stat()
    file_type, _ = guess_type(path.name)
    source_key = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"

    try:
        relative_path = str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        relative_path = str(path.resolve())

    return {
        "source": "local",
        "source_id": sha256(source_key.encode("utf-8")).hexdigest(),
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "relative_path": relative_path,
        "file_extension": path.suffix.lower(),
        "file_type": file_type or "application/octet-stream",
        "file_size_bytes": stat.st_size,
        "created_at": _to_utc_iso(stat.st_ctime),
        "modified_at": _to_utc_iso(stat.st_mtime),
    }


def load_documents(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    required_exts: Iterable[str] = SUPPORTED_EXTENSIONS,
    recursive: bool = False,
    num_files_limit: int | None = None,
) -> list[Document]:
    input_dir = Path(data_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {input_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(input_dir),
        recursive=recursive,
        required_exts=list(required_exts),
        num_files_limit=num_files_limit,
        file_metadata=extract_file_metadata,
    )
    return reader.load_data()


def list_source_files(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    required_exts: Iterable[str] = SUPPORTED_EXTENSIONS,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    input_dir = Path(data_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {input_dir}")

    allowed_exts = {extension.lower() for extension in required_exts}
    files = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return [
        extract_file_metadata(str(path))
        for path in sorted(files)
        if path.is_file() and path.suffix.lower() in allowed_exts
    ]


def parse_documents(
    documents: list[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> list[Any]:
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    chunk_counts_by_source: dict[str, int] = {}

    for node in nodes:
        source_id = str(node.metadata.get("source_id", "unknown"))
        chunk_index = chunk_counts_by_source.get(source_id, 0)
        node.metadata["chunk_index"] = chunk_index
        node.metadata["chunk_size"] = len(node.get_content())
        chunk_counts_by_source[source_id] = chunk_index + 1

    return nodes


def process_documents(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    recursive: bool = False,
    num_files_limit: int | None = None,
) -> ProcessedDocuments:
    documents = load_documents(
        data_dir=data_dir,
        recursive=recursive,
        num_files_limit=num_files_limit,
    )
    nodes = parse_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return ProcessedDocuments(documents=documents, nodes=nodes)


def preview_nodes(nodes: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "text": node.get_content()[:500],
            "metadata": node.metadata,
        }
        for node in nodes[:limit]
    ]


def _to_utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
