"""Retrieve the most relevant chunks of a question with BM25."""

from bm25s import BM25, tokenize
from .models import (
    MinimalSource, MinimalSearchResults, RagDataset, StudentSearchResults
    )
import uuid
from pathlib import Path
from tqdm import tqdm
import json
from pydantic import ValidationError
import sys


class Retriever:
    """Search the BM25 index."""

    def __init__(self) -> None:
        """Load the saved index and the sources."""
        try:
            self.bm25 = BM25().load('data/processed')
        except OSError as err:
            print(
                "\033[31m"
                f"{err.__class__.__name__}: {err}\n"
                "Index missing in data/processed.\nUsage: "
                "uv run python -m src index --max_chunk_size <int>"
                "\033[0m",
                file=sys.stderr
                )
            sys.exit(1)
        sources_path = Path('data/processed/sources.json')
        self.sources: list[MinimalSource] = [
                MinimalSource.model_validate(source)
                for source in json.loads(sources_path.read_text())
            ]

    def get_question_corresponding_sources(
            self,
            question: str,
            k: int = 3,
            question_id: str = str(uuid.uuid4())
            ) -> MinimalSearchResults:
        """Find the top-k sources of a question with BM25.

        Args:
            question: The question.
            k: Number of sources to keep.
            question_id: Identifier of the question.

        Returns:
            The question with its retrieved sources.
        """
        if k == 0:
            print(
                "\033[31mValueError: k must be greater than 0.\033[0m",
                file=sys.stderr
                )
            sys.exit(1)
        try:
            indices, scores = self.bm25.retrieve(tokenize(question), k=k)
        except ValueError as err:
            print(f"\033[31mValueError: {err}\033[0m", file=sys.stderr)
            sys.exit(1)
        return MinimalSearchResults(
            question_id=question_id,
            question=question,
            retrieved_sources=[
                self.sources[i]
                for i, score in zip(indices[0], scores[0])
                if score > 0
                ]
        )

    def search(self, question: str, k: int = 3) -> MinimalSearchResults:
        """Find the top-k sources of a question.

        Args:
            question: The question.
            k: Number of sources to keep.

        Returns:
            The question with its retrieved sources.
        """
        return self.get_question_corresponding_sources(question, k)

    def search_dataset(
            self,
            dataset_path: str,
            k: int,
            save_directory: str
            ) -> None:
        """Search each question of a dataset and save the results.

        Args:
            dataset_path: JSON file of the questions.
            k: Number of sources to keep.
            save_directory: Folder of the results file.
        """
        try:
            rag_dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text()
                )
        except OSError as err:
            print(
                f"\033[31m{err.__class__.__name__}: {err}\033[0m",
                file=sys.stderr
                )
            sys.exit(1)
        except ValidationError as err:
            for error in err.errors():
                print(
                    f"\033[31mValidationError: {error['msg']}\033[0m",
                    file=sys.stderr
                    )
            sys.exit(1)

        search_results: list[MinimalSearchResults] = []
        for unanswered_question in tqdm(
            rag_dataset.rag_questions, desc='Searching', unit=' question'
                ):
            search_results.append(self.get_question_corresponding_sources(
                unanswered_question.question,
                k,
                unanswered_question.question_id
            ))
        save_path = Path(save_directory) / Path(dataset_path).name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        student_search_results = StudentSearchResults(
            search_results=search_results,
            k=k
        )
        save_path.write_text(student_search_results.model_dump_json(indent=2))
        print(f"Saved student_search_results to {save_path}")
