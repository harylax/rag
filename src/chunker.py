from langchain_text_splitters import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter
)  # type: ignore
from .models import MinimalSource
from pathlib import Path


class Chunks:
    def __init__(self) -> None:
        self.processed_md_chunks: list[MinimalSource] = []
        self.processed_py_chunks: list[MinimalSource] = []
        self.processed_txt_chunks: list[MinimalSource] = []


def return_read_file(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError as err:
        print(f"{err.__class__.__name__}: {err}")
        exit(1)


if __name__ == "__main__":
    print(return_read_file('data/raw/vllm-0.10.1/READY.md'))
