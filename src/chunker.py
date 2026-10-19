from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter
)  # type: ignore
from .models import Chunk
from pathlib import Path


class Chunker:
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size: int = max_chunk_size
        self.processed_md_chunks: list[Chunk] = []


class MarkdownChunker(Chunker):
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
                last_character_index=len(text),
                titles=titles,
                text=text[start_char:]
            ))

        # splitter = MarkdownHeaderTextSplitter(
        #     headers_to_split_on=[
        #         ('#', 'h1'),
        #         ('##', 'h2'),
        #         ('###', 'h3'),
        #         ('####', 'h4'),
        #         ('#####', 'h5'),
        #         ('######', 'h6'),
        #         ]
        #     )
        # documents = splitter.split_text(text)

        # default_start: int = 0

        # for doc in documents:
        #     titles: list[str] = []
        #     for key in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
        #         title: str | None = doc.metadata.get(key)
        #         if title:
        #             titles.append(title)

        #     chunk_text: str = doc.page_content
        #     lines: list[str] = chunk_text.splitlines()
        #     if not lines:
        #         continue

        #     start: int = text.find(chunk_text, default_start)
        #     if start < 0:
        #         start = text.find(lines[0].strip(), default_start)
        #     if start < 0:
        #         start = default_start

        #     end: int = -1
        #     if text.startswith(chunk_text, start):
        #         end = start + len(chunk_text)
        #     else:
        #         pos: int = text.find(lines[-1], start)
        #         end = max(pos + len(lines[-1]), start + len(chunk_text))

        #     res.append(Chunk(
        #         file_path=file_path,
        #         first_character_index=start,
        #         last_character_index=end - 1,
        #         titles=titles,
        #         text=chunk_text
        #     ))

        #     default_start = end

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
                file_path=file_path,
                first_character_index=start,
                last_character_index=end,
                titles=chunk.titles,
                text=chunk_text
            ))
        return res

    def recursive_split(self, text: str, file_path: str) -> list[Chunk]:
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
    file_path: str = 'data/raw/vllm-0.10.1/README.md'
    content: str = return_read_file(file_path)
    for chunk in MarkdownChunker(200).test(content, file_path):
        # print(chunk.model_dump_json(indent=2))
        # print('========================================')
        # print(chunk.text)
        # print('----------------------------------------')
        # print(content[chunk.first_character_index:chunk.last_character_index + 1])
        # print('========================================')
        assert chunk.text == content[chunk.first_character_index:chunk.last_character_index + 1]
