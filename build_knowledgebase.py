from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".html"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Elasticsearch knowledge base with LangChain.")
    parser.add_argument("--data-path", required=True, help="Directory containing source documents.")
    parser.add_argument("--elasticsearch-url", default="http://localhost:9200", help="Elasticsearch URL.")
    parser.add_argument("--index-name", default="knowledge-base", help="Elasticsearch index name.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size for text splitting.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap for text splitting.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used for embeddings.",
    )
    return parser.parse_args()


def read_documents(data_path: str) -> list[Document]:
    from langchain_core.documents import Document

    path = Path(data_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Data path does not exist or is not a directory: {data_path}")

    documents = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = file_path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                documents.append(Document(page_content=text, metadata={"source": str(file_path)}))

    if not documents:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"No supported documents found in {data_path}. Expected files: {supported}")

    return documents


def build_knowledgebase(
    data_path: str,
    elasticsearch_url: str,
    index_name: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> tuple[int, int]:
    from langchain_elasticsearch import ElasticsearchStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = read_documents(data_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    try:
        store = ElasticsearchStore(es_url=elasticsearch_url, index_name=index_name, embedding=embeddings)
        store.add_documents(chunks)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to index documents into Elasticsearch at {elasticsearch_url}. "
            "Ensure Elasticsearch is running and reachable."
        ) from exc
    return len(documents), len(chunks)


def main() -> None:
    args = parse_args()
    document_count, chunk_count = build_knowledgebase(
        data_path=args.data_path,
        elasticsearch_url=args.elasticsearch_url,
        index_name=args.index_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )
    print(
        f"Knowledge base created successfully. Indexed {document_count} documents "
        f"as {chunk_count} chunks into '{args.index_name}'."
    )


if __name__ == "__main__":
    main()
