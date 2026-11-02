"""Build and save the BM25 index of the corpus chunks."""

from bm25s import BM25
from .chunker import chunk_folder_files
from .models import Chunk, MinimalSource
from pathlib import Path
from typing import Any
import json
from tqdm import tqdm
from bm25s import tokenize


class Indexer:
    """BM25 index of the corpus, rebuilt only if it changed."""

    def __init__(
            self,
            folder_path: str = 'data/raw/vllm-0.10.1',
            max_chunk_size: int = 2000
            ) -> None:
        """Index the corpus unless the saved index is up to date.

        Args:
            folder_path: Folder of the corpus.
            max_chunk_size: Maximum chunk size in characters.
        """
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
            folder_path, max_chunk_size
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
            print("Index up to date under data/processed")
            return

        chunks: list[Chunk] = chunk_folder_files(
            folder_path, max_chunk_size
        )
        sources = [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index
            ) for chunk in chunks
        ]
        self.bm25 = BM25()
        self.tokens = self.get_tokens(chunks)
        self.bm25.index(self.tokens)
        self.bm25.save('data/processed/')
        Path('data/processed/corpus_checker.json').write_text(
            json.dumps(new_checker, indent=2)
            )
        self.save_sources(sources)
        print(
            f"Ingestion completed! Indexed {len(sources)} "
            f"chunks under data/processed"
            )

    def save_sources(self, sources: list[MinimalSource]) -> None:
        """Save the file path and character range of each chunk.

        Args:
            sources: One source per chunk, in index order.
        """
        data: list[Any] = [source.model_dump() for source in sources]
        Path('data/processed/sources.json').write_text(
            json.dumps(data, indent=2)
        )

    def get_tokens(self, chunks: list[Chunk]) -> Any:
        """Tokenize each chunk with its file name and titles.

        Args:
            chunks: Chunks of the corpus.

        Returns:
            The tokenized chunks, ready to be indexed.
        """
        res = []
        for chunk in tqdm(chunks, desc='Tokenizing', unit=' chunk'):
            titles: str = '\n'.join(chunk.titles)
            path: str = str(Path(chunk.file_path).stem)
            res.append(path + '\n' + titles + '\n' + chunk.text)
        return tokenize(res)

    def corpus_checker(
            self,
            corpus_path: str,
            max_chunk_size: int
            ) -> dict[str, Any]:
        """Describe the state of the corpus.

        Args:
            corpus_path: Folder of the corpus.
            max_chunk_size: Maximum chunk size in characters.

        Returns:
            Modification date and size of each file, and the chunk size.
        """
        files: dict[str, Any] = {}
        for file in Path(corpus_path).rglob('*'):
            if file.suffix in ['.txt', '.md', '.py']:
                stat = file.stat()
                files[str(file)] = [stat.st_mtime, stat.st_size]
        files['max_chunk_size'] = max_chunk_size
        return files
