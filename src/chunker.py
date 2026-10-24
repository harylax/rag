from langchain_text_splitters import (
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter
)
from .models import Chunk
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
import ast
from dataclasses import dataclass


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size: int = max_chunk_size
        if (
            self.max_chunk_size <= 0
            or self.max_chunk_size > 2000
        ):
            print(
                "\033[31mValueError: max_chunk_size should be at least "
                "1 and at most 2000\033[0m"
            )
            exit(1)

    @abstractmethod
    def split_in_chunks(self, text: str, file_path: str) -> list[Chunk]:
        ...


class TextChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)

    def split_in_chunks(self, text: str, file_path: str) -> list[Chunk]:
        res: list[Chunk] = []
        splitter = RecursiveCharacterTextSplitter(
            ['\n\n', '\n', '. ',  ' ', ''],
            chunk_size=self.max_chunk_size,
            chunk_overlap=int(self.max_chunk_size * 0.1),
            add_start_index=True
        )
        documents = splitter.create_documents([text])
        for doc in documents:
            start: int = doc.metadata['start_index']
            end: int = start + len(doc.page_content)
            res.append(Chunk(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end - 1,
                titles=[],
                text=doc.page_content
            ))
        return res


class MarkdownChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)

    def split_by_title(self, text: str, file_path: str) -> list[Chunk]:
        res: list[Chunk] = []

        lines: list[str] = text.splitlines(keepends=True)
        titles: list[str] = []

        in_bloc: bool = False
        start_char: int = 0
        start_lineno: int = 0
        for lineno, line in enumerate(lines):
            if line.startswith('```'):
                in_bloc = not in_bloc
            if line.startswith((
                '# ', '## ', '### ',
                '#### ', '##### ', '###### '
            )):
                if in_bloc:
                    continue
                chunk_text: str = ''.join(lines[start_lineno:lineno])
                end_char: int = start_char + len(chunk_text)
                res.append(Chunk(
                    file_path=file_path,
                    first_character_index=start_char,
                    last_character_index=end_char - 1,
                    titles=titles,
                    text=chunk_text
                ))
                start_char = end_char
                start_lineno = lineno

                if titles:
                    count_current_hash: int = line.split()[0].count('#')
                    while count_current_hash <= titles[-1].split()[0].count('#'):
                        titles.pop()
                        if not titles:
                            break
                titles.append(line)

        if start_char < len(text):
            res.append(Chunk(
                file_path=file_path,
                first_character_index=start_char,
                last_character_index=len(text) - 1,
                titles=titles,
                text=text[start_char:]
            ))

        return self.clean_titles(res)

    def clean_titles(self, chunks: list[Chunk]) -> list[Chunk]:
        res: list[Chunk] = []
        for chunk in chunks:
            lines: list[str] = chunk.text.splitlines(keepends=True)
            if all(
                line.startswith((
                    '# ', '## ', '### ',
                    '#### ', '##### ', '###### '
                    ))
                or not line.strip()
                for line in lines
                    ):
                continue
            start: int = chunk.first_character_index
            end: int = chunk.last_character_index
            chunk_text: str = chunk.text
            if lines and lines[0].startswith((
                '# ', '## ', '### ', '#### ', '##### ', '###### '
            )):
                start = start + len(lines[0])
                chunk_text = ''.join(lines[1:])
            res.append(Chunk(
                file_path=chunk.file_path,
                first_character_index=start,
                last_character_index=end,
                titles=chunk.titles,
                text=chunk_text
            ))
        return res

    def split_in_chunks(self, text: str, file_path: str) -> list[Chunk]:
        res: list[Chunk] = []
        chunks: list[Chunk] = self.split_by_title(text, file_path)
        for chunk in chunks:
            if len(chunk.text) > self.max_chunk_size:
                splitter = RecursiveCharacterTextSplitter(
                    ['\n\n', '\n', '. ', ' ', ''],
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=int(self.max_chunk_size * 0.1),
                    add_start_index=True
                    )
                documents = splitter.create_documents([chunk.text])
                for doc in documents:
                    start: int = chunk.first_character_index \
                        + doc.metadata['start_index']
                    end: int = start + len(doc.page_content)
                    res.append(Chunk(
                        file_path=file_path,
                        first_character_index=start,
                        last_character_index=end - 1,
                        titles=chunk.titles,
                        text=doc.page_content
                    ))
            else:
                res.append(chunk)
        return res


@dataclass
class Definition:
    lineno: int
    end_lineno: int
    title: str


class PythonCodeChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)

    def get_definitions(self, text: str) -> list[Definition]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        res: list[Definition] = []
        for node in ast.walk(tree):
            if not isinstance(node, (
                ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef
            )):
                continue
            if isinstance(node, ast.ClassDef):
                title = f"class {node.name}"
            elif isinstance(node, ast.FunctionDef):
                title = f"def {node.name}"
            elif isinstance(node, ast.AsyncFunctionDef):
                title = f"async def {node.name}"
            else:
                continue
            lineno = (
                node.decorator_list[0].lineno
                if node.decorator_list
                else node.lineno
            )
            end_lineno = node.end_lineno or node.lineno
            res.append(Definition(
                lineno=lineno,
                end_lineno=end_lineno,
                title=title
            ))
        return sorted(res, key=lambda x: x.lineno)


def folder_files_content(
        folder_path: str = 'data/raw/vllm-0.10.1'
        ) -> dict[str, str]:
    res: dict[str, str] = {}
    for file in Path(folder_path).rglob('*'):
        if file.is_file() and file.suffix in ['.txt', '.md', '.py']:
            res[str(file)] = file.read_text()
    if not res:
        print("\033[31mFolderErro: no markdown / python file found.\033[0m")
        exit(1)
    return res


def chunk_folder_files(
        folder_path: str = 'data/raw/vllm-0.10.1',
        max_chunk_size: int = 2000
) -> list[Chunk]:
    res: list[Chunk] = []
    files_content: dict[str, str] = folder_files_content(folder_path)
    for path, content in tqdm(
        files_content.items(),
        desc='Chunking',
        unit=' file'
            ):
        if Path(path).suffix == '.txt':
            res.extend(
                TextChunker(max_chunk_size).split_in_chunks(content, path)
                )
        if Path(path).suffix == '.md':
            res.extend(
                MarkdownChunker(max_chunk_size).split_in_chunks(content, path)
                )
        if Path(path).suffix == '.py':
            res.extend(
                PythonCodeChunker(max_chunk_size).split_in_chunks(content, path)
                )
    return res


def read_file(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError as err:
        print(f"\033[31m{err.__class__.__name__}: {err}\033[0m")
        exit(1)


def test_md(
        max_chunk_size: int = 2000,
        file_path: str = 'data/raw/vllm-0.10.1/README.md'
        ) -> None:
    content = read_file(file_path)
    chunks = MarkdownChunker(max_chunk_size).split_in_chunks(content, file_path)
    for chunk in chunks:
        # print(chunk.model_dump_json(indent=2))
        print('========================================')
        print(chunk.text)
        print('----------------------------------------')
        print(content[chunk.first_character_index:chunk.last_character_index + 1])
        print('========================================')
        assert chunk.text == content[chunk.first_character_index:chunk.last_character_index + 1]


def test_py(
        max_chunk_size: int = 2000,
        file_path: str = 'data/raw/vllm-0.10.1/setup.py'
        ) -> None:
    content = read_file(file_path)
    chunks = PythonCodeChunker(max_chunk_size).split_in_chunks(content, file_path)
    for chunk in chunks:
        # print(chunk.model_dump_json(indent=2))
        print('========================================')
        print(chunk.text)
        print('----------------------------------------')
        print(content[chunk.first_character_index:chunk.last_character_index + 1])
        print('========================================')
        assert chunk.text == content[chunk.first_character_index:chunk.last_character_index + 1]


def test_txt(
        max_chunk_size: int = 2000,
        file_path: str = 'data/raw/vllm-0.10.1/requirements/cpu.txt'
        ) -> None:
    content = read_file(file_path)
    chunks = TextChunker(max_chunk_size).split_in_chunks(content, file_path)
    for chunk in chunks:
        # print(chunk.model_dump_json(indent=2))
        print('========================================')
        print(chunk.text)
        print('----------------------------------------')
        print(content[chunk.first_character_index:chunk.last_character_index + 1])
        print('========================================')
        assert chunk.text == content[chunk.first_character_index:chunk.last_character_index + 1]


def test_process(max_chunk_size: int = 2000) -> None:
    # processed_txt = chunk_folder_files(
    #     max_chunk_size=max_chunk_size, file_extension=['.txt']
    #     )
    # processed_md = chunk_folder_files(
    #     max_chunk_size=max_chunk_size, file_extension=['.txt', '.md']
    #     )
    # processed_py = chunk_folder_files(
    #     max_chunk_size=max_chunk_size, file_extension=['.py']
    #     )
    processed_all = chunk_folder_files(max_chunk_size=max_chunk_size)
    # for chunk in processed_txt:
    #     print(chunk.model_dump_json(indent=2))
    # for chunk in processed_md:
    #     print(chunk.model_dump_json(indent=2))
    # for chunk in processed_py:
    #     print(chunk.model_dump_json(indent=2))
    for chunk in processed_all:
        print(chunk.model_dump_json(indent=2))


if __name__ == "__main__":
    test_process(100)
