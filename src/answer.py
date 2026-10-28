import ollama
from .models import (
    MinimalSource, MinimalSearchResults,
    StudentSearchResultsAndAnswer, StudentSearchResults,
    MinimalAnswer
    )
from .retrieve import Retriever
from typing import Any
from pathlib import Path
from tqdm import tqdm


MAX_CONTEXT_CHARS: int = 7000

class Answer:
    def __init__(self, model: str = 'qwen3:0.6b') -> None:
        self.model: str = model

    @staticmethod
    def read_source_text(source: MinimalSource) -> str:
        file_content: str = Path(source.file_path).read_text()
        start: int = source.first_character_index
        end: int = source.last_character_index + 1
        return file_content[start:end]

    @staticmethod
    def build_context(
        retrieved_sources: list[MinimalSource],
        max_chars: int = MAX_CONTEXT_CHARS
            ) -> str:
        res: list[str] = []
        remaining: int = max_chars
        for i, source in enumerate(retrieved_sources, start=1):
            part: str = (
                f"[{i}] {source.file_path!r}\n"
                f"{Answer.read_source_text(source).strip()}"
            )
            if len(part) > remaining:
                if remaining > 200:
                    res.append(part[:remaining])
                break
            res.append(part)
            remaining -= len(part) + 2
        return '\n\n'.join(res)

    @staticmethod
    def build_prompt(
        question: str,
        search_results: MinimalSearchResults
            ) -> str:
        res: str = 'Read the following chunks of documents:\n'
        res += Answer.build_context(search_results.retrieved_sources)
        res += (
            '\n\nOnly answer the following question based on the '
            'previous provided documents. If you cannot find the '
            'answer in the documents, just say "I don\'t know"\n'
        )
        res += f'The question :"{question}"\n'
        return res

    def answer(
            self,
            question: str,
            k: int = 3
    ) -> str:
        search_results: MinimalSearchResults = Retriever().search(question, k)
        response: Any = ollama.chat(
            model=self.model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a RAG assistant. '
                        'Only answer in 1 to 3 sentences.'
                    )
                },
                {
                    'role': 'user',
                    'content': self.build_prompt(question, search_results)
                }
            ],
            think=False,
            stream=True,
            options={'temperature': 0, 'num_ctx': 4096, 'num_predict': 200}
        )
        print(f"\033[32m>>>Query: {question!r}\033[0m")
        print("\033[33mAnswer: \033[0m", end='')
        generated: str = ''
        for token in response:
            content: str = token['message']['content']
            generated += content
            print(f"\033[33m{content}\033[0m", end='', flush=True)
        print()
        return generated

    def _answer(self, minimal_search_results: MinimalSearchResults) -> MinimalAnswer:
        response: Any = ollama.chat(
            model=self.model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a RAG assistant. '
                        'Only answer in 1 to 3 sentences.'
                    )
                },
                {
                    'role': 'user',
                    'content': self.build_prompt(
                        minimal_search_results.question,
                        minimal_search_results
                        )
                }
            ],
            think=False,
            options={'temperature': 0, 'num_ctx': 4096, 'num_predict': 200}
        )
        return MinimalAnswer(
            question_id=minimal_search_results.question_id,
            question=minimal_search_results.question,
            retrieved_sources=minimal_search_results.retrieved_sources,
            answer=response['message']['content']
        )

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
    ) -> StudentSearchResultsAndAnswer:
        search_path = Path(student_search_results_path)

        if not search_path.exists():
            dataset_path = Path('data/datasets/UnansweredQuestions') / search_path.name
            print(
                "\033[31m"
                f"student_search_result_path: '{student_search_results_path}' "
                "not found.\nUsage: uv run python -m src search_dataset \\"
                f"\n--dataset_path {dataset_path} --k <int> \\"
                f"\n--save_directory data/output/search_results/UnansweredQuestions"
                "\033[0m"
            )
            exit(1)

        student_search_results = StudentSearchResults.model_validate_json(
            search_path.read_text()
            )
        k: int = student_search_results.k
        answers: list[MinimalAnswer] = []
        for search_res in tqdm(
            student_search_results.search_results,
            desc='Answering',
            unit=' question'
        ):
            answers.append(self._answer(search_res))

        student_search_results_and_answer = StudentSearchResultsAndAnswer(
            search_results=answers, k=k
        )

        save_path: Path = Path(save_directory) / search_path.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            student_search_results_and_answer.model_dump_json(indent=2)
        )

        return student_search_results_and_answer


if __name__ == "__main__":
    from .indexer import Indexer
    Indexer(max_chunk_size=2000)
    question = 'How can I configure the vLLM Base Code?'
    results = Retriever().search(question, 10)
    prompt = Answer.build_prompt(question, results)
    print(prompt)
    print(f"\nPrompt length: {len(prompt)} characters")
