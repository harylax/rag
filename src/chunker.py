from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter
)  # type: ignore
from .models import Chunk
from pathlib import Path


class Chunker:
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size: int= max_chunk_size
        self.processed_md_chunks: list[Chunk] = []


class MarkdownChunker(Chunker):
    def __init__(self, max_chunk_size = 2000) -> None:
        super().__init__(max_chunk_size)

    def split_by_title(self, text: str, file_path: str) -> list[Chunk]:
        res: list[Chunk] = []

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'h1'),
                ('##', 'h2'),
                ('###', 'h3'),
                ('####', 'h4'),
                ('#####', 'h5'),
                ('######', 'h6'),
                ]
            )
        documents = splitter.split_text(text)

        default_start: int = 0

        for doc in documents:
            titles: list[str] = []
            for key in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                title: str | None = doc.metadata.get(key)
                if title:
                    titles.append(title)

            chunk_text: str = doc.page_content
            start: int = text.find(chunk_text.splitlines()[0].strip())
            if start < default_start:
                start = default_start
            end: int = start + len(chunk_text)

            res.append(Chunk(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end - 1,
                titles=titles,
                text=chunk_text
            ))

            default_start = end

        return res

    def test(self, text: str, file_path: str) -> list[Chunk]:
        res: list[Chunk] = []
        chunks: list[Chunk] = self.split_by_title(text, file_path)
        for chunk in chunks:
            if len(chunk.text) > self.max_chunk_size:
                splitter = RecursiveCharacterTextSplitter(
                    ['\n\n', '\n', '. ', ' ', ''],
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=self.max_chunk_size * 0.1)
                documents = splitter.split_text(chunk.text)
                start: int = chunk.first_character_index
                for doc in documents:
                    end: int = start + len(doc)
                    res.append(Chunk(
                        file_path=file_path,
                        first_character_index=start,
                        last_character_index=end - 1,
                        titles=chunk.titles,
                        text=doc
                    ))
                    start = end
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
        print(chunk.model_dump_json(indent=2))
        # print('========================================')
        # print(chunk.text)
        # print('----------------------------------------')
        # print(content[chunk.first_character_index:chunk.last_character_index + 1])
        # print('========================================')
