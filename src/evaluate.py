from .models import MinimalSource, StudentSearchResults, RagDataset
from pathlib import Path
import sys
from pydantic import ValidationError

IOU_THRESHOLD: float = 0.05
STUDENT_SEARCH_RESULTS_PATH: str = (
    'data/output/search_results/UnansweredQuestions/dataset_docs_public.json'
)
DATASET_PATH: str = 'data/datasets/AnsweredQuestions/dataset_docs_public.json'


def iou(found: MinimalSource, expected: MinimalSource) -> float:
    if found.file_path != expected.file_path:
        return 0.0
    found_i = {
        i for i in range(
            found.first_character_index,
            found.last_character_index + 1
        )
    }
    expected_i = {
        i for i in range(
            expected.first_character_index,
            expected.last_character_index + 1
        )
    }
    inter = len(found_i.intersection(expected_i))
    union = len(found_i.union(expected_i))
    return inter / union if union else 0.0


def score_at_k(
        retrieved: list[MinimalSource],
        expected: list[MinimalSource],
        k: int
        ) -> float:
    top_k: list[MinimalSource] = retrieved[:k]
    found: int = sum(
        any(iou(f, e) > IOU_THRESHOLD for f in top_k)
        for e in expected
        )
    return found / len(expected)


def recall_at_k(
        retrieved: dict[str, list[MinimalSource]],
        expected: dict[str, list[MinimalSource]],
        k: int
        ) -> float:
    scores: list[float] = [
        score_at_k(retrieved.get(question_id, []), sources, k)
        for question_id, sources in expected.items()
    ]
    return sum(scores) / len(scores)


def evaluate(
        student_search_results_path: str = STUDENT_SEARCH_RESULTS_PATH,
        dataset_path: str = DATASET_PATH
        ) -> None:
    try:
        students_search_results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text()
            )
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

    expected: dict[str, list[MinimalSource]] = {
        q.question_id: q.sources
        for q in rag_dataset.rag_questions
    }
    retrieved: dict[str, list[MinimalSource]] = {
        r.question_id: r.retrieved_sources
        for r in students_search_results.search_results
    }

    n_expected = len(expected)
    n_retrieved = len(retrieved)
    print(f"Student data is valid: {n_expected == n_retrieved}")
    print(f"Total number of questions: {n_expected}")
    print(f"Total number of questions with sources: {n_expected}")
    print(f"Total number of questions with student sources: {n_retrieved}")
    print("🎯 Evaluation Results")
    print("========================================")
    print(f"📊 Questions evaluated: {len(retrieved)}")
    data: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        if k > students_search_results.k:
            print(f"📈 Recall@{k}: N/A")
            continue
        recall = recall_at_k(retrieved, expected, k)
        print(f"📈 Recall@{k}: {recall:.3f} ({(recall * 100):.1f}%)")
        data[f'recall@{k}'] = recall
    print(data)
