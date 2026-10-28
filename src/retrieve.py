from bm25s import BM25, tokenize
from .models import (
    MinimalSource, MinimalSearchResults, RagDataset, StudentSearchResults
    )
import uuid
from pathlib import Path
from tqdm import tqdm
import json
from pydantic import ValidationError


class Retriever:
    def __init__(self) -> None:
        try:
            self.bm25 = BM25().load('data/processed')
        except OSError as err:
            print(
                "\033[31m"
                f"{err.__class__.__name__}: {err}\n"
                "Index missing in data/processed.\nUsage: "
                "uv run python -m src index --max_chunk_size <int>"
                "\033[0m"
                )
            exit(1)
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
        indices, _ = self.bm25.retrieve(tokenize(question), k=k)
        return MinimalSearchResults(
            question_id=question_id,
            question=question,
            retrieved_sources=[self.sources[i] for i in indices[0]]
        )

    def search(self, question: str, k: int = 3) -> MinimalSearchResults:
        return self.get_question_corresponding_sources(question, k)

    def search_dataset(
            self,
            dataset_path: str = 'data/datasets/UnansweredQuestions/dataset_docs_public.json',
            k: int = 3,
            save_directory: str = 'data/output/search_results/UnansweredQuestions'
            ) -> None:

        try:
            rag_dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text()
                )
        except OSError as err:
            print(f"\033[31m{err.__class__.__name__}: {err}\033[0m")
            exit(1)
        except ValidationError as err:
            for error in err.errors():
                print(f"\033[31mValidationError: {error['msg']}\033[0m")
            exit(1)

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


def read_source(source: MinimalSource) -> str:
    file_path = Path(source.file_path)
    try:
        file_content: str = file_path.read_text()
        start = source.first_character_index
        end = source.last_character_index
        return file_content[start:end + 1]
    except OSError as err:
        print(f"\033[31m{err.__class__.__name__}: {err}\033[0m")
        exit(1)


if __name__ == "__main__":
    retriever = Retriever()
    sources = retriever.search("How can I configure OpenAI?", 10)
    for source in sources.retrieved_sources:
        print("=====================================")
        print(f"path: {source.file_path!r}")
        print("-------------------------------------")
        print(">>>SOURCE<<<")
        print(read_source(source))
        print("=====================================")
    # import json
    # print(json.dumps([source.model_dump() for source in sources], indent=2))
