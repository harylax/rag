*This project has been created as part of the 42 curriculum by haryandr.*

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

- [Abstract Syntax Trees](https://docs.python.org/3/library/ast.html) - `ast`
- [What is BM25?](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/)
- [Okapi BM25 (wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25)
- [bm25s (github)](https://github.com/xhluca/bm25s)
- [Ollama Python library](https://github.com/ollama/ollama-python) - `ollama.chat`
- [langchain text splitters (documentation)](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) - `RecursiveCharacterTextSplitter`
- [langchain-text-splitters (github)](https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters)
- [Python Fire](https://github.com/google/python-fire)
- [unicode symbol](https://www.unicodelibrary.com/category)

### AI Usage

AI tools were used as assistants for:

- Explaining BM25 scoring and how to use it
- 

## System architecture

```
┌───────────┐
│   Corpus  │ data/raw/vllm-0.10.1 (.py, .md, .txt)
└───────────┘
      ⇩
┌───────────┐
│  Chunker  │ TextChunker - MarkdownChunker - PythonCodeChunker
└───────────┘   ➥ list[Chunk]
      ⇩                   ➥ file_path, first/last char index, titles, text
┌───────────┐
│  Indexer  │ bm25s tokenizer + BM25 index
└───────────┘   ➥ data/processed/ (index, vocab, sources.json, corpus_checker.json)
      ⇩
┌───────────┐
│ Retriever │ BM25 retrieve - question ➙ index ➙ top-k MinimalSource
└───────────┘   ➣ search_dataset ➙ evaluate - Recall@k
      ⇩
┌───────────┐
│   Answer  │ Context built from sources ➙ Ollama (qwen3:0.6b)
└───────────┘   ➥ short answer
```
| Module            | Role                      |
|-------------------|---------------------------|
| `src/chunker.py`  | Splits each file into chunks, with distinct strategy for `.py`, and `.md` and `.txt`  |
| `src/indexer.py`  | Builds and saves the BM25 index; skips the work if the corpus or chunk size are unchanged |
| `src/retrieve.py` | Loads the index and returns the top-k sources for a question or a dataset |
| `src/evaluate.py` | Computes recall@k against the answered dataset |
| `src/answer.py`   | Builds the prompt from the retrieved sources and queries the local model |
| `src/models.py`   | Pydantic models shared by all stages (`Chunk`, `MinimalSource`, ...) |
| `src/__main__.py` | Command-line interface (Python Fire) |


## Chunking strategy

- **Python (`.py`)**: 
  1. the file is parsed with `ast` to get the top-level statements
  2. a statement that fits in the chunk size is kept as it is; a larger node is split recursively into its children until it fits or has no children left
  3. small consecutive statements are grouped while they fit, as long as they are not less indented that the first statement of the group
  4. each group is turned into a block of text (including all the text between statements); big blocks are splitted with `RecursiveCharacterTextSplitter` without chunk overlap and keeping separator at the end of the block
  5. tiny blocks are merged with their neighbor until the result is larger than chunk size
  6. the parsed file is walked through again to get definitions (`class ...`, `def ...`) as `titles`
  7. if the file has a python syntax error, it falls back to the langchain recursive splitter
  8. the chunks are built with the final blocks of text and the titles corresponding (ovelapping chunk lines) obtained previously
- **Markdown (`.md`) / text (`.txt`)**:
  1. markdown files are split in chunks at each heading (`#` to `######`), ignoring lines inside code fences; the heading hierarchy is stored as `titles`
  2. heading lines are removed from the chunk text 
  3. larger chunks are split with `RecursiveCharacterTextSplitter` with the separators `\n\n`, `\n`, `. `, space
  4. text files are split directly with `RecursiveCharacterTextSplitter` with the same separators

Markdown and Text are split with a 10% overlap to keep a whole sentence.

Python chunker doesn't need overlap, since its chunks already follow the code structure.

## Retrieval method

The `bm25s` library and its main class **BM25** were used for this project for the indexing and retrieval part. I did not reimplement the BM25 algorithm.

- `tokenize()`: transforms the chunk text, titles and file name (or the question) into tokens.
- `index()`: computes the term frequency in each chunk, the document frequency of each term in the corpus, the IDF (rareness) of each token, and the length of each chunk. 
- `save()`: save the index in `data/processed`. The list of all `MinimalSource` (file_path, first/last character index) and a corpus checker (each file size and modification date) are also saved in there.
- `load()`: loads the saved index into a `BM25` instance. The list of `MinimalSource` is loaded in a `Retriever` instance.
- `retrieve()`: scores every chunk against the question and returns the top-k sources, ranked from the best to the lesser good.

How **BM25** works:
- A token counts more when it is rare in the corpus (IDF): usual tokens like `the` or `to` get very low score, while rare tokens such as `OpenAI` get high score.
- A chunk gets a higher score when it contains the tokens of the question.
- A chunk with repeated rare tokens gets an increase of its score.
- The score is normalized by the chunk length, so a long chunk score is reduced in favor of a short one. 

## Performance analysis

## Design decisions

## Challenges faced

## Example usage
