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

- 42 RAG against the machine subject
- vllm-0.10.1 codebase
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

- Explaining BM25 scoring and how to use the `bm25s` library.
- Explaining the `langchain-text-splitters` library and how to use it.

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
  3. small consecutive statements are grouped while they fit, as long as they are not less indented than the first statement of the group
  4. each group is turned into a block of text (including all the text between statements); big blocks are split with `RecursiveCharacterTextSplitter` without chunk overlap and keeping separator at the end of the block
  5. tiny blocks are merged with their neighbor until the result is larger than chunk size
  6. the parsed file is walked through again to get definitions (`class ...`, `def ...`) as `titles`
  7. if the file has a python syntax error, it falls back to the langchain recursive splitter
  8. the chunks are built with the final blocks of text and the titles corresponding (overlapping chunk lines) obtained previously
- **Markdown (`.md`) / text (`.txt`)**:
  1. markdown files are split in chunks at each heading (`#` to `######`), ignoring lines inside code fences; the heading hierarchy is stored as `titles`
  2. heading lines are removed from the chunk text 
  3. larger chunks are split with `RecursiveCharacterTextSplitter` with the separators `\n\n`, `\n`, `. `, space
  4. text files are split directly with `RecursiveCharacterTextSplitter` with the same separators

Markdown and Text are split with a 10% overlap to keep a whole sentence.

Python chunker doesn't need overlap, since its chunks already follow the code structure.

## Retrieval method

The `bm25s` library and its main class **BM25** were used for this project for the indexing and retrieval part. I did not reimplement the BM25 algorithm.

- `tokenize()`: transforms the chunk text, titles and file name (or the question) into tokens. Common tokens (stopword), such as `the` or `to` are removed in advance by `tokenize()`.
- `index()`: computes the term frequency in each chunk, the document frequency of each term in the corpus, the IDF (rareness) of each token, and the length of each chunk. 
- `save()`: save the index in `data/processed`. The list of all `MinimalSource` (file_path, first/last character index) and a corpus checker (each file size and modification date) are also saved in there.
- `load()`: loads the saved index into a `BM25` instance. The list of `MinimalSource` is loaded in a `Retriever` instance.
- `retrieve()`: scores every chunk against the question and returns the top-k sources, ranked from the best to the lesser good.

How **BM25** works:
- A token counts more when it is rare in the corpus (IDF): usual tokens like `import` or `def` get very low score, while rare tokens such as `OpenAI` get high score.
- A chunk gets a higher score when it contains the tokens of the question.
- A chunk with repeated rare tokens gets an increase of its score.
- The score is normalized by the chunk length, so a long chunk score is reduced in favor of a short one. 

## Performance analysis

- **Recall@5**:

      `Recall@k` measures the accuracy of the k sources retrieved for a dataset of about 100 questions.  A retrieved chunk matches an expected source when they are in the same file and their character ranges overlap enough (IoU > 0.05).

| max_chunk_size | Docs recall@5 | Code recall@5 |
|----------------|---------------|---------------|
|     200        | 80.0%         | 62.6%         |
|     300        | 84.0%         | 61.6%         |
|     400        | 83.0%         | 64.6%         |
|     500        | 80.0%         | 68.7%         |
|     600        | 83.0%         | 67.7%         |
|     700        | 84.0%         | 69.7%         |
|     800        | 83.0%         | 65.7%         |
|     900        | 84.0%         | 66.7%         |
|     1000       | 86.0%         | 65.7%         |
|     1100       | 88.0%         | 66.7%         |
|     1200       | 86.0%         | 68.7%         |
|     1300       | 87.0%         | 66.7%         |
|     1400       | 87.0%         | 66.7%         |
|     1500       | 85.0%         | 67.7%         |
|     1600       | 84.0%         | 64.6%         |
|     1700       | 87.0%         | 66.7%         |
|     1800       | 87.0%         | 67.7%         |
|     1900       | 88.0%         | 66.7%         |
|     2000       | 87.0%         | 66.7%         |

Below 200 max_chunk_size, docs recall@5 drops sharply (71% at 150, 56% at 100, 27% at 50). In fact short chunk doesn't give enough information, so several chunks can match the expected source.

- **Indexing time**: chunking and indexing the whole corpus is quick, particularly for larger `max_chunk_size`.  It takes only a few seconds, far below the 5 minutes limit.
- **Retrieval throughput**: searching for 200 questions is almost instantaneous, thanks to the `bm25s` optimization.

## Design decisions

- **Plug a ready-to-use BM25 to my pipeline instead of reimplementing it**: fast to adapt to a RAG pipeline (pure python) and optimized (uses numpy).
- **Chunks that follow structure, with titles (heading / definition) in the index**: chunks follow the structure of the code and documents (functions, classes, headings). A chunk is more likely to be a meaningful unit instead of a block of `max_chunk_size` characters. Moreover, the function names, the headings and the file name are indexed with the chunk, which helps on the retrieval process.
- **Ollama**: the model is only used as an answering machine based on the retrieved context, so a small `qwen3:0.6b` served by Ollama is enough. What's more it is much lighter to install and run (CPU only).

## Challenges faced

- **Find ways to improve recall@5**: for lower chunk size especially for docs, recall@5 was under the performance limit.
    - **Fix**: the file name and the titles (headings / definitions) are tokenized together with the chunk text. This raised recall@5 by 8 to 13 points on docs and 9 to 22 points on code, depending on the chunk size.
- **Reduce text loss between two python code chunks**: splitting in nodes with AST leaves the comments and decorators that sit between two nodes outside them.
    - **Fix**: each block now starts where the previous one ends, so the text between goes into the next chunk.
- **Avoid reindexing the corpus at each run**: indexing takes time, and it is useless if nothing changed. An `Indexer` call always re-index the corpus.
    - **Fix**: the size and modification date of each file and the chunk size are written in `corpus_checker.json`. The index is rebuilt only if the `max_chunk_size` has changed or any change in a corpus file has occurred.

## Example usage

All the commands are described in the [Execution](#execution) section.

For instance, to search the sources of a question:
```bash
uv run python -m src search "How to configure the OpenAI server" --k 3
```
Output:
```bash
data/raw/vllm-0.10.1/benchmarks/auto_tune/README.md [3394:3471]
data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md [1940:3168]
data/raw/vllm-0.10.1/examples/online_serving/openai_chat_completion_client_with_tools_required.py [0:565]
```

To answer a question:
```bash
uv run python -m src answer "How to configure the openAI server" --k 3
```

Output:
```bash
>>>Query: 'How to configure the openAI server'
Answer: To configure the OpenAI server, you can set the API key and base URL using the provided code snippet.
```
