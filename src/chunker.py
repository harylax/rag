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


def path_content_in_folder(
        folder_path: str,
        file_extension: str
        ) -> dict[str, str]:
    res: dict[str, str] = {}
    for file in Path(folder_path).rglob('*' + file_extension):
        try:
            res[str(file)] = file.read_text()
        except OSError:
            continue
    if not res:
        print(f"\033[31mNo '{file_extension}' file found.\033[0m")
    return res


def return_read_file(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError as err:
        print(f"\033[31m{err.__class__.__name__}: {err}\033[0m")
        exit(1)


if __name__ == "__main__":
    md_files: dict[str, str] = path_content_in_folder('data/raw', '.md')
    for path, content in md_files.items():
        print("\n============================")
        print(f"\033[32mpath: {path}")
        print("------------------------------")
        print(f"content:\033[0m\n{content}")
        print("============================\n")
