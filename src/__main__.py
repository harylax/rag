from .indexer import Indexer
from .retrieve import Retriever
from fire import Fire
from .answer import Answer
from .evaluate import evaluate


class CLI:
    def index(
            self,
            max_chunk_size: int = 2000,
            corpus_path: str = 'data/raw/vllm-0.10.1'
    ) -> None:
        Indexer(corpus_path, max_chunk_size)

    def search(
            self,
            question: str,
            k: int = 3
    ) -> None:
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
        Retriever().search_dataset(
            dataset_path, k, save_directory
            )

    def answer(
            self,
            question: str,
            k: int = 3
    ) -> None:
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
        evaluate(student_search_results_path, dataset_path)


if __name__ == "__main__":
    Fire(CLI)
