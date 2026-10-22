from bm25s import BM25, tokenize
from .chunker import chunk_folder_files
from .models import Chunk
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
        data_exists = Path('data/processed/data.csc.index.npy').exists()
        indices_exists = Path('data/processed/indices.csc.index.npy').exists()
        indptr_exists = Path('data/processed/indptr.csc.index.npy').exists()
        params_exists = Path('data/processed/params.index.json').exists()
        vocab_exists = Path('data/processed/vocab.index.json').exists()
        checker_exists = Path('data/processed/corpus_checker.json').exists()

        old: dict[str, Any] | None = None
        if checker_exists:
            old = json.loads(
                Path('data/processed/corpus_checker.json').read_text()
                )

        new = self.check_corpus(folder_path, file_extension, max_chunk_size)

        if (
            all(file for file in (
                data_exists,
                indices_exists,
                indptr_exists,
                params_exists,
                vocab_exists,
                checker_exists
            ))
            and old == new
                ):
            self.bm25.load('data/processed')
            print(">>>>>LOAD<<<<<")
        else:
            self.chunks: list[Chunk] = chunk_folder_files(
                folder_path, max_chunk_size, file_extension
            )
            self.tokens = self.get_tokens()
            self.bm25.index(self.tokens, leave_progress=True)
            self.bm25.save('data/processed/')
            Path('data/processed/corpus_checker.json').write_text(
                json.dumps(new)
                )
            print(">>>>>REINDEX<<<<<")

    def get_tokens(self) -> list[list[str]]:
        res = []
        for chunk in self.chunks:
            titles: str = '\n'.join(chunk.titles)
            res.append(titles + chunk.text)
        return tokenize(res, leave=True)

    # def get_tokens(self) -> list[list[str]]:
    #     res = []
    #     for chunk in self.chunks:
    #         titles: str = '\n'.join(chunk.titles)
    #         res.append(tokenize(titles + chunk.text))
    #     return res

    def check_corpus(
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
    indexer = Indexer(max_chunk_size=10, file_extension=['.md', '.txt', '.py'])
    import inspect
    print(inspect.signature(BM25.index))
    print(inspect.signature(tokenize))
