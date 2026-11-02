"""Command-line interface of the RAG pipeline."""

from .indexer import Indexer
from .retrieve import Retriever
from fire import Fire
from .answer import Answer
from .evaluate import evaluate


class CLI:
    """Commands of the RAG pipeline."""

    def index(
            self,
            max_chunk_size: int = 2000,
            corpus_path: str = 'data/raw/vllm-0.10.1'
    ) -> None:
        """Index the corpus.

        Args:
            max_chunk_size: Maximum chunk size in characters.
            corpus_path: Folder of the corpus to index.
        """
        Indexer(corpus_path, max_chunk_size)

    def search(
            self,
            question: str,
            k: int = 3
    ) -> None:
        """Print the top-k sources of a question.

        Args:
            question: The question.
            k: Number of sources to keep.
        """
        search_results = Retriever().search(question, k)
        for source in search_results.retrieved_sources:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
                )

    def search_dataset(
            self,
            dataset_path: str = (
                'data/datasets/UnansweredQuestions/dataset_docs_public.json'
                ),
            k: int = 3,
            save_directory: str = (
                'data/output/search_results/UnansweredQuestions'
                )
    ) -> None:
        """Save the top-k sources of each question of a dataset.

        Args:
            dataset_path: JSON file of the questions.
            k: Number of sources to keep.
            save_directory: Folder of the results file.
        """
        Retriever().search_dataset(
            dataset_path, k, save_directory
            )

    def answer(
            self,
            question: str,
            k: int = 3
    ) -> None:
        """Answer a question from its top-k sources.

        Args:
            question: The question.
            k: Number of sources used as context.
        """
        Answer().answer(question, k)

    def answer_dataset(
            self,
            student_search_results_path: str = (
                'data/output/search_results/UnansweredQuestions'
                '/dataset_docs_public.json'
            ),
            save_directory: str = (
                'data/output/search_results_and_answer/UnansweredQuestions'
            )
    ) -> None:
        """Answer each question of a search results file.

        Args:
            student_search_results_path: Search results file.
            save_directory: Folder of the answers file.
        """
        Answer().answer_dataset(student_search_results_path, save_directory)

    def evaluate(
            self,
            student_search_results_path: str = (
                'data/output/search_results/UnansweredQuestions/'
                'dataset_docs_public.json'
            ),
            dataset_path: str = (
                'data/datasets/AnsweredQuestions/dataset_docs_public.json'
            )
    ) -> None:
        """Print the recall@k of the search results.

        Args:
            student_search_results_path: Search results file.
            dataset_path: JSON file of the expected sources.
        """
        evaluate(student_search_results_path, dataset_path)


if __name__ == "__main__":
    Fire(CLI)
