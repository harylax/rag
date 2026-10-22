from bm25s import BM25, tokenize
from .chunker import chunk_folder_files
from .models import Chunk, MinimalSource
from pathlib import Path
from typing import Any
import json


class Indexer:
    def __init__(
            self,
            folder_path: str = 'data/raw/vllm-0.10.1',
            max_chunk_size: int = 2000,
            file_extension: list[str] = ['.txt', '.md', '.py']
            ) -> None:
        self.bm25 = BM25()

        data_path = Path('data/processed/data.csc.index.npy')
        indices_path = Path('data/processed/indices.csc.index.npy')
        indptr_path = Path('data/processed/indptr.csc.index.npy')
        params_path = Path('data/processed/params.index.json')
        vocab_path = Path('data/processed/vocab.index.json')
        checker_path = Path('data/processed/corpus_checker.json')
        sources_path = Path('data/processed/sources.json')

        old_checker: dict[str, Any] | None = None
        if checker_path.exists():
            old_checker = json.loads(
                Path('data/processed/corpus_checker.json').read_text()
                )

        new_checker: dict[str, Any] = self.corpus_checker(
            folder_path,
            file_extension,
            max_chunk_size
            )

        if (
            all(path.exists() for path in (
                data_path,
                indices_path,
                indptr_path,
                params_path,
                vocab_path,
                checker_path,
                sources_path
            ))
            and old_checker == new_checker
                ):
            # self.bm25 = self.bm25.load('data/processed')
            # self.sources: list[MinimalSource] = [
            #     MinimalSource.model_validate(source)
            #     for source in json.loads(sources_path.read_text())
            # ]
            print(">>>>>LOAD<<<<<")
        else:
            chunks: list[Chunk] = chunk_folder_files(
                folder_path, max_chunk_size, file_extension
            )
            sources = [
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index
                ) for chunk in chunks
            ]
            self.tokens = self.get_tokens(chunks)
            self.bm25.index(self.tokens, leave_progress=True)
            self.bm25.save('data/processed/')
            Path('data/processed/corpus_checker.json').write_text(
                json.dumps(new_checker, indent=2)
                )
            self.save_sources(sources)
            print(">>>>>REINDEX<<<<<")

    def save_sources(self, sources: list[MinimalSource]) -> None:
        data: list[Any] = [source.model_dump() for source in sources]
        Path('data/processed/sources.json').write_text(
            json.dumps(data, indent=2)
        )

    def get_tokens(self, chunks: list[Chunk]) -> Any:
        res = []
        for chunk in chunks:
            titles: str = '\n'.join(chunk.titles)
            path: str = str(Path(chunk.file_path).stem)
            res.append(path + titles + chunk.text)
        return tokenize(res, leave=True)

    def corpus_checker(
            self,
            corpus_path: str,
            file_extension: list[str],
            max_chunk_size: int
            ) -> dict[str, Any]:
        files: dict[str, Any] = {}
        for file in Path(corpus_path).rglob('*'):
            if file.suffix in file_extension:
                stat = file.stat()
                files[str(file)] = [stat.st_mtime, stat.st_size]
        files['max_chunk_size'] = max_chunk_size
        return files


if __name__ == "__main__":
    indexer = Indexer(max_chunk_size=1000, file_extension=['.md', '.txt', '.py'])
