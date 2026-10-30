* This project has been created as part of the 42 curriculum by haryandr.*

# RAG against the machine

Will you answer my questions?

## Description

The project is an implementation of a Retrieval Augmented Generation (RAG) system that answers questions about a corpus codebase (the vLLM repository).

Its purpose is to teach the basics of a RAG pipeline.

The pipeline has five stages:
- **Chunking**: split source files into smaller, meaningful pieces.
- **Indexing**: build a BM25 index from the chunks and persist it under 'data/processed/':
	- **vocabulary**: all unique tokens extracted from the chunks;
	- **term frequency (TF)**: how many times each token appears inside a chunk;
	- **inverse document frequency (IDF)**: how rare a token is across the whole corpus;
	- **length of documents**: length of each chunk.
- **Retrieval**: given a question, return the top-k most relevant chunks.
- **Evaluation**: measure retrieval quality with recall@k (correct sources among top-k results)
- **Answering**: feed a small local model (Qwen/Qwen3-0.6B) with a context built with the retrieved sources to generate an answer.

## Instructions

### Requirements

- Python 3.10+
- uv
- Ollama

### Installation

Installation fails without enough space. It is safer to redirect caches to `goinfre`:

```bash
export OLLAMA_MODELS=$HOME/goinfre/ollama/.models
export UV_CACHE_DIR=$HOME/goinfre/uv_cache
```

Install everything:

```bash
make install
```

or manually:

```bash
uv sync
ollama pull qwen3:0.6b
```

This creates a virtual environment (`.venv`) and installs all dependencies.

Make sure the Ollama server is running before generating answers:

```bash
make serve
```

### Compilation

No compilation is required - the project is in Python.

### Execution

Run the pipeline steps in order:

1. Index the corpus

```bash
uv run python -m src index --max_chunk_size 2000
```

2. Search a single question

```bash
uv run python -m src search "How to configure the OpenAI server" --k 5
```

3. Search a whole dataset

```bash
uv run python -m src search_dataset \
--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
--k 10 \
--save_directory data/output/search_results/UnansweredQuestions
```

4. Answer a single question

```bash
uv run python -m src answer "How to configure the openAI server" --k 3
```

5. Answer a whole dataset

```bash
uv run python -m src answer_dataset \
--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
--save_directory data/output/search_results_and_answer/UnansweredQuestions
```

6. Evaluate retrieval quality

```bash
uv run python -m src evaluate \
--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

## Resources

- [Abstract Syntax Trees (python `ast`)](https://docs.python.org/3/library/ast.html)
- [What is BM25?](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/)
- [Okapi BM25 (wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25)
- [bm25s (github)](https://github.com/xhluca/bm25s)
- [Ollama Python library](https://github.com/ollama/ollama-python) - `ollama.chat`
- [langchain text splitters (documentation)](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) - `RecursiveCharacterTextSplitter`
- [langchain-text-splitters (github)](https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters)
- [Python Fire](https://github.com/google/python-fire)

### AI Usage

AI tools were used as assistants for:

- Explaining BM25 scoring and how to use it
- 

## System architecture

## Chunking strategy

## Retrieval method

## Performance analysis

## Design decisions

## Challenges faced

## Example usage