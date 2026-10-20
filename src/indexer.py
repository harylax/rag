from bm25s import BM25, tokenize
from .chunker import process_needed_chunks
from .models import Chunk
from pathlib import Path


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

        if all(
            file for file in (
                data_exists,
                indices_exists,
                indptr_exists,
                params_exists,
                vocab_exists)
                ):
            self.bm25.load('data/processed')
            print(">>>>>LOAD<<<<<")
        else:
            self.chunks: list[Chunk] = process_needed_chunks(
            folder_path, max_chunk_size, file_extension
        )
            self.tokens = self.get_tokens()
            self.bm25.index(self.tokens)
            self.bm25.save('data/processed/')
            print(">>>>>REINDEX<<<<<")

    def get_tokens(self) -> list[list[str]]:
        res = []
        for chunk in self.chunks:
            titles: str = '\n'.join(chunk.titles)
            res.append(titles + chunk.text)
        return tokenize(res)

    # def get_tokens(self) -> list[list[str]]:
    #     res = []
    #     for chunk in self.chunks:
    #         titles: str = '\n'.join(chunk.titles)
    #         res.append(tokenize(titles + chunk.text))
    #     return res


if __name__ == "__main__":
    indexer = Indexer(file_extension=['.md', '.txt'])
