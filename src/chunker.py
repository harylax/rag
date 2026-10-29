from langchain_text_splitters import RecursiveCharacterTextSplitter
from .models import Chunk
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
import ast
from dataclasses import dataclass
import sys

MIN_CHUNK_SIZE: int = 200


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size: int = max_chunk_size
        if (
            self.max_chunk_size < 200
            or self.max_chunk_size > 2000
        ):
            print(
                "\033[31mValueError: max_chunk_size should be at least "
                f"{MIN_CHUNK_SIZE} and at most 2000\033[0m",
                file=sys.stderr
            )
            sys.exit(1)

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
                    while count_current_hash <= titles[-1].split()[0].count(
                        '#'
                    ):
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

    def node_length(self, lines: list[str], node: ast.stmt) -> int:
        start: int = node.lineno - 1
        end: int = node.end_lineno or node.lineno
        return len(''.join(lines[start:end]))

    def _get_children(self, node: ast.stmt) -> list[ast.stmt]:
        children: list[ast.stmt] = list(getattr(node, 'body', []))
        children.extend(getattr(node, 'orelse', []))
        children.extend(getattr(node, 'finalbody', []))
        for handler in getattr(node, 'handlers', []):
            children.extend(handler.body)
        for case in getattr(node, 'cases', []):
            children.extend(case.body)
        return sorted(children, key=lambda x: x.lineno)

    def smallest_nodes(
            self,
            lines: list[str],
            node: ast.stmt
            ) -> list[ast.stmt]:
        if self.node_length(lines, node) <= self.max_chunk_size:
            return [node]
        children: list[ast.stmt] = self._get_children(node)
        if not children:
            return [node]
        res: list[ast.stmt] = []
        for child in children:
            res.extend(self.smallest_nodes(lines, child))
        return res

    def group_nodes(
            self,
            lines: list[str],
            nodes: list[ast.stmt]
            ) -> list[list[ast.stmt]]:
        res: list[list[ast.stmt]] = []
        group: list[ast.stmt] = []
        group_len: int = 0
        for node in nodes:
            node_len: int = self.node_length(lines, node)
            if not group:
                group, group_len = [node], node_len
            elif (
                node_len + group_len > self.max_chunk_size
                or group[0].col_offset > node.col_offset
            ):
                res.append(group)
                group, group_len = [node], node_len
            else:
                group.append(node)
                group_len += node_len
        if group:
            res.append(group)
        return res

    def build_blocks(self, text: str, tree: ast.Module | None) -> list[str]:
        if tree is None:
            return [text]
        lines: list[str] = text.splitlines(keepends=True)
        nodes: list[ast.stmt] = []
        for node in tree.body:
            nodes.extend(self.smallest_nodes(lines, node))
        blocks: list[str] = []
        start: int = 0
        for group in self.group_nodes(lines, nodes):
            end: int = group[-1].end_lineno or group[-1].lineno
            blocks.append(''.join(lines[start:end]))
            start = end
        tail: str = ''.join(lines[start:])
        if tail:
            blocks.append(tail)
        return blocks

    def split_oversized(self, blocks: list[str]) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=0,
            keep_separator='end',
            strip_whitespace=False
        )
        res: list[str] = []
        for block in blocks:
            if len(block) <= self.max_chunk_size:
                res.append(block)
            else:
                res.extend(splitter.split_text(block))
        return res

    def merge_small(self, blocks: list[str]) -> list[str]:
        res: list[str] = []
        for block in blocks:
            if res and len(res[-1]) + len(block) <= self.max_chunk_size:
                res[-1] += block
            else:
                res.append(block)
        return res

    def get_definitions(self, tree: ast.Module | None) -> list[Definition]:
        res: list[Definition] = []
        if tree is None:
            return res
        for node in ast.walk(tree):
            if not isinstance(node, (
                ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef
            )):
                continue
            if isinstance(node, ast.ClassDef):
                prefix = 'class'
            elif isinstance(node, ast.AsyncFunctionDef):
                prefix = 'async def'
            else:
                prefix = 'def'
            lineno = (
                node.decorator_list[0].lineno
                if node.decorator_list
                else node.lineno
            )
            res.append(Definition(
                lineno=lineno,
                end_lineno=node.end_lineno or node.lineno,
                title=f"{prefix} {node.name}"
            ))
        return sorted(res, key=lambda x: x.lineno)

    def split_in_chunks(self, text: str, file_path: str) -> list[Chunk]:
        try:
            tree: ast.Module | None = ast.parse(text)
        except SyntaxError:
            tree = None
        blocks = self.build_blocks(text, tree)
        blocks = self.split_oversized(blocks)
        blocks = self.merge_small(blocks)
        definitions = self.get_definitions(tree)

        res: list[Chunk] = []
        start: int = 0
        for block in blocks:
            end: int = start + len(block)
            first_line: int = text.count('\n', 0, start) + 1
            last_line: int = text.count('\n', 0, end - 1) + 1
            res.append(Chunk(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end - 1,
                titles=[
                    d.title for d in definitions
                    if d.lineno <= last_line and d.end_lineno >= first_line
                ],
                text=block
            ))
            start = end
        return res


def folder_files_content(
        folder_path: str = 'data/raw/vllm-0.10.1'
        ) -> dict[str, str]:
    res: dict[str, str] = {}
    for file in Path(folder_path).rglob('*'):
        if file.is_file() and file.suffix in ['.txt', '.md', '.py']:
            res[str(file)] = file.read_text()
    if not res:
        print(
            "\033[31mFolderErro: no markdown / python file found.\033[0m",
            file=sys.stderr
            )
        sys.exit(1)
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
                PythonCodeChunker(max_chunk_size).split_in_chunks(
                    content, path
                    )
                )
    return res


# def read_file(path: str) -> str:
#     try:
#         return Path(path).read_text()
#     except OSError as err:
#         print(f"\033[31m{err.__class__.__name__}: {err}\033[0m")
#         exit(1)


# def test_md(
#         max_chunk_size: int = 2000,
#         file_path: str = 'data/raw/vllm-0.10.1/README.md'
#         ) -> None:
#     content = read_file(file_path)
#     chunks = MarkdownChunker(max_chunk_size).split_in_chunks(content, file_path)
#     for chunk in chunks:
#         # print(chunk.model_dump_json(indent=2))
#         print('========================================')
#         print(chunk.text)
#         print('----------------------------------------')
#         print(content[chunk.first_character_index:chunk.last_character_index + 1])
#         print('========================================')
#         assert chunk.text == content[chunk.first_character_index:chunk.last_character_index + 1]


# def test_py(
#         max_chunk_size: int = 2000,
#         file_path: str = 'data/raw/vllm-0.10.1/setup.py'
#         ) -> None:
#     content = read_file(file_path)
#     chunks = PythonCodeChunker(max_chunk_size).split_in_chunks(content, file_path)
#     for chunk in chunks:
#         if chunk.titles:
#             print(chunk.model_dump_json(indent=2))
#         # print('========================================')
#         # print(chunk.text)
#         # print('----------------------------------------')
#         # print(content[chunk.first_character_index:chunk.last_character_index + 1])
#         # print('========================================')
#         # assert chunk.text == content[chunk.first_character_index:chunk.last_character_index + 1]


# def test_txt(
#         max_chunk_size: int = 2000,
#         file_path: str = 'data/raw/vllm-0.10.1/requirements/cpu.txt'
#         ) -> None:
#     content = read_file(file_path)
#     chunks = TextChunker(max_chunk_size).split_in_chunks(content, file_path)
#     for chunk in chunks:
#         # print(chunk.model_dump_json(indent=2))
#         print('========================================')
#         print(chunk.text)
#         print('----------------------------------------')
#         print(content[chunk.first_character_index:chunk.last_character_index + 1])
#         print('========================================')
#         assert chunk.text == content[chunk.first_character_index:chunk.last_character_index + 1]


# def test_process(max_chunk_size: int = 2000) -> None:
#     # processed_txt = chunk_folder_files(
#     #     max_chunk_size=max_chunk_size, file_extension=['.txt']
#     #     )
#     # processed_md = chunk_folder_files(
#     #     max_chunk_size=max_chunk_size, file_extension=['.txt', '.md']
#     #     )
#     # processed_py = chunk_folder_files(
#     #     max_chunk_size=max_chunk_size, file_extension=['.py']
#     #     )
#     processed_all = chunk_folder_files(max_chunk_size=max_chunk_size)
#     # for chunk in processed_txt:
#     #     print(chunk.model_dump_json(indent=2))
#     # for chunk in processed_md:
#     #     print(chunk.model_dump_json(indent=2))
#     # for chunk in processed_py:
#     #     print(chunk.model_dump_json(indent=2))
#     for chunk in processed_all:
#         print(chunk.model_dump_json(indent=2))


# if __name__ == "__main__":
#     test_py(200)
