import re
from bm25s.stopwords import STOPWORDS_EN, STOPWORDS_EN_PLUS
from .models import MinimalSource
from pathlib import Path


STOPWORDS = frozenset(STOPWORDS_EN + STOPWORDS_EN_PLUS)


def tokenize_code(text: str) -> list[str]:
    res: list[str] = re.findall(r"[A-Za-z0-9_]+", text)
    extra: list[str] = []
    for s in res:
        parts: list[
            str
            ] = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", s)
        if len(parts) > 1:
            extra.extend(parts)
    res.extend(extra)
    tokens = [s.lower() for s in res if len(s) > 1]
    return [t for t in tokens if t not in STOPWORDS]


def tokenize_docs(text: str) -> list[str]:
    res: list[str] = re.findall(r"(?u)\b\w\w+\b", text.lower())
    return [t for t in res if t not in STOPWORDS]


def read_source_text(source: MinimalSource) -> str:
    file_content: str = Path(source.file_path).read_text()
    start: int = source.first_character_index
    end: int = source.last_character_index + 1
    return file_content[start:end]
