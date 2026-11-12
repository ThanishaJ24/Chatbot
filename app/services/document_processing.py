from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Iterable

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


@dataclass(slots=True)
class ParsedDocuments:
    documents: list[Document]

    def summary(self) -> dict[str, Any]:
        file_names = sorted(
            {
                document.metadata.get("file_name", "unknown")
                for document in self.documents
            }
        )
        total_characters = sum(
            int(document.metadata.get("character_count", 0))
            for document in self.documents
        )
        return {
            "document_count": len(self.documents),
            "total_characters": total_characters,
            "files": file_names,
        }


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
    file_offset: int = 0,
) -> list[Document]:
    input_dir = Path(data_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {input_dir}")

    allowed_exts = {extension.lower() for extension in required_exts}
    paths = _iter_source_paths(input_dir, allowed_exts, recursive)
    paths = paths[file_offset:]
    if num_files_limit is not None:
        paths = paths[:num_files_limit]

    return [_load_document(path) for path in paths]


def list_source_files(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    required_exts: Iterable[str] = SUPPORTED_EXTENSIONS,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    input_dir = Path(data_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {input_dir}")

    allowed_exts = {extension.lower() for extension in required_exts}
    return [
        extract_file_metadata(str(path))
        for path in _iter_source_paths(input_dir, allowed_exts, recursive)
    ]


def parse_source_documents(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    recursive: bool = False,
    num_files_limit: int | None = None,
    file_offset: int = 0,
) -> ParsedDocuments:
    documents = load_documents(
        data_dir=data_dir,
        recursive=recursive,
        num_files_limit=num_files_limit,
        file_offset=file_offset,
    )
    return ParsedDocuments(documents=documents)


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
        content = node.get_content()
        node.id_ = sha256(
            f"{source_id}:{chunk_index}:{content}".encode("utf-8")
        ).hexdigest()
        node.metadata["chunk_index"] = chunk_index
        node.metadata["chunk_size"] = len(content)
        chunk_counts_by_source[source_id] = chunk_index + 1

    return nodes


def process_documents(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    recursive: bool = False,
    num_files_limit: int | None = None,
    file_offset: int = 0,
) -> ProcessedDocuments:
    documents = load_documents(
        data_dir=data_dir,
        recursive=recursive,
        num_files_limit=num_files_limit,
        file_offset=file_offset,
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


def preview_documents(
    documents: list[Document],
    limit: int = 5,
    text_limit: int = 700,
) -> list[dict[str, Any]]:
    return [
        {
            "text": document.text[:text_limit],
            "metadata": document.metadata,
        }
        for document in documents[:limit]
    ]


def _to_utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def _iter_source_paths(
    input_dir: Path,
    allowed_exts: set[str],
    recursive: bool,
) -> list[Path]:
    files = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in files
        if path.is_file() and path.suffix.lower() in allowed_exts
    )


def _load_document(path: Path) -> Document:
    metadata = extract_file_metadata(str(path))

    if path.suffix.lower() == ".pdf":
        text, page_count = _read_pdf_text(path)
        metadata["page_count"] = page_count
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    metadata["character_count"] = len(text)
    return Document(text=text, metadata=metadata)


def _read_pdf_text(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"Page {page_number}\n{page_text}")

    return "\n\n".join(pages), len(reader.pages)
