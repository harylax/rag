import ollama
from .models import (
    MinimalSource, MinimalSearchResults,
    StudentSearchResultsAndAnswer, StudentSearchResults,
    MinimalAnswer
    )
from .utils import read_source_text
from .retrieve import Retriever
from typing import Any
from pathlib import Path
from tqdm import tqdm


class Answer:
    def __init__(self, model: str = 'qwen3:0.6b') -> None:
        self.model: str = model

    @staticmethod
    def build_context(retrieved_sources: list[MinimalSource]) -> str:
        res: list[str] = []
        for i, source in enumerate(retrieved_sources, start=1):
            res.append(f"[{i}] {read_source_text(source).strip()}")
        return '\n'.join(res)

    def build_prompt(self, question: str, k: int = 3) -> str:
        search_results: MinimalSearchResults = \
            Retriever().search(question, k)
        res: str = 'Read the following chunks of documents:\n'
        res += self.build_context(search_results.retrieved_sources)
        res += (
            '\nOnly answer the following question based on the '
            'previous provided documents. If you cannot find the '
            'answer in the documents, just say "I do not know"\n'
        )
        res += f'The question :"{question}"\n'
        return res

    def answer(
            self,
            question: str,
            k: int = 3,
            stream: bool = True
    ) -> str:
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
                    'content': self.build_prompt(question, k)
                }
            ],
            think=False,
            stream=stream,
            options={'temperature': 0, 'num_ctx': 4096, 'num_predict': 200}
        )
        if not stream:
            return response['message']['content']
        print(f"\033[32m>>>Query: {question!r}\033[0m")
        print("\033[33mAnswer: \033[0m", end='')
        generated: str = ''
        for token in response:
            content: str = token['message']['content']
            generated += content
            print(f"\033[33m{content}\033[0m", end='', flush=True)
        print()
        return generated

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
            answers.append(MinimalAnswer(
                question_id=search_res.question_id,
                question=search_res.question,
                retrieved_sources=search_res.retrieved_sources,
                answer=self.answer(search_res.question, k, False)
            ))

        student_search_results_and_answer = StudentSearchResultsAndAnswer(
            search_results=answers, k=k
        )

        save_path: Path = Path(save_directory) / search_path.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            student_search_results_and_answer.model_dump_json(indent=2)
        )

        return student_search_results_and_answer
