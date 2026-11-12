from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.services.vector_indexing import (
    _get_or_create_pinecone_index,
    _openai_embedding_model,
    _pinecone_config,
    _pinecone_vector_store,
)

DEFAULT_LLM_MODEL = "gpt-4o-mini"


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    score: float | None
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class QueryAnswer:
    answer: str
    chunks: list[RetrievedChunk]
    top_k: int
    llm_model: str
    index_name: str
    namespace: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "top_k": self.top_k,
            "llm_model": self.llm_model,
            "index_name": self.index_name,
            "namespace": self.namespace,
            "retrieved_chunks": [chunk.as_dict() for chunk in self.chunks],
        }


def answer_question_from_index(
    question: str,
    top_k: int = 5,
    namespace: str | None = None,
) -> QueryAnswer:
    config = _pinecone_config(namespace=namespace)
    pinecone_index = _get_or_create_pinecone_index(config, create_if_missing=False)
    vector_store = _pinecone_vector_store(
        pinecone_index=pinecone_index,
        namespace=config.namespace,
    )
    index = _vector_index_from_store(
        vector_store=vector_store,
        embed_model=_openai_embedding_model(config),
    )
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)
    chunks = [
        RetrievedChunk(
            text=node.node.get_content(),
            score=node.score,
            metadata=node.node.metadata,
        )
        for node in nodes
    ]

    llm_model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    answer = _generate_answer_with_langchain(
        question=question,
        chunks=chunks,
        llm_model=llm_model,
    )
    return QueryAnswer(
        answer=answer,
        chunks=chunks,
        top_k=top_k,
        llm_model=llm_model,
        index_name=config.index_name,
        namespace=config.namespace,
    )


def _vector_index_from_store(vector_store: Any, embed_model: Any) -> Any:
    from llama_index.core import VectorStoreIndex

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


def _generate_answer_with_langchain(
    question: str,
    chunks: list[RetrievedChunk],
    llm_model: str,
) -> str:
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing LangChain dependencies. Run: pip install -r requirements.txt"
        ) from exc

    context = _format_context(chunks)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise RAG assistant. Answer only from the provided "
                "context. If the context does not contain the answer, say you do "
                "not know. Keep the answer concise and cite source file names "
                "when useful.",
            ),
            (
                "human",
                "Question:\n{question}\n\nRetrieved context:\n{context}",
            ),
        ]
    )
    llm = ChatOpenAI(model=llm_model, temperature=0)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant context was retrieved."

    formatted_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        file_name = chunk.metadata.get("file_name", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", "unknown")
        formatted_chunks.append(
            f"[Chunk {index} | file={file_name} | chunk_index={chunk_index}]\n"
            f"{chunk.text}"
        )

    return "\n\n".join(formatted_chunks)
