# Langchain-rag-elastic-search-

Minimal knowledge base builder using **LangChain** and **Elasticsearch**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build a knowledge base

Put `.txt`, `.md`, `.rst`, or `.html` files under a data directory, then run:

```bash
python build_knowledgebase.py \
  --data-path ./data \
  --elasticsearch-url http://localhost:9200 \
  --index-name knowledge-base
```

Optional flags:

- `--chunk-size` (default: `800`)
- `--chunk-overlap` (default: `120`)
- `--embedding-model` (default: `sentence-transformers/all-MiniLM-L6-v2`)
