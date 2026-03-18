from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_file(path: Path, content: str) -> None:
    if path.exists():
        return
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    dirs = [
        root / "apps" / "cli",
        root / "apps" / "web",
        root / "frontend" / "src" / "app",
        root / "frontend" / "src" / "components",
        root / "frontend" / "src" / "lib",
        root / "rag-service" / "rag_service" / "modules",
        root / "rag-service" / "data",
        root / "src" / "hotel_review_ai",
        root / "src" / "hotel_review_ai" / "data",
        root / "src" / "hotel_review_ai" / "rag",
        root / "src" / "hotel_review_ai" / "llm",
        root / "data" / "raw",
        root / "data" / "cleaned",
        root / "data" / "rag",
        root / "artifacts" / "faiss",
        root / "artifacts" / "runs",
        root / "models" / "base",
        root / "models" / "adapters",
        root / "tests",
    ]

    for d in dirs:
        ensure_dir(d)

    ensure_file(
        root / "pyproject.toml",
        "\n".join(
            [
                "[build-system]",
                'requires = ["setuptools>=68", "wheel"]',
                'build-backend = "setuptools.build_meta"',
                "",
                "[project]",
                'name = "hotel-review-ai"',
                'version = "0.0.0"',
                'requires-python = ">=3.10"',
                "",
                "[tool.setuptools]",
                'package-dir = {"" = "src"}',
                "",
                "[tool.setuptools.packages.find]",
                'where = ["src"]',
                "",
            ]
        )
        + "\n",
    )

    ensure_file(root / "src" / "hotel_review_ai" / "__init__.py", 'APP_NAME = "hotel-review-ai"\n')

    ensure_file(
        root / "src" / "hotel_review_ai" / "config.py",
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from dataclasses import dataclass",
                "from pathlib import Path",
                "",
                "",
                "@dataclass(frozen=True)",
                "class ProjectPaths:",
                "    root: Path",
                "    data_dir: Path",
                "    raw_dir: Path",
                "    cleaned_dir: Path",
                "    rag_dir: Path",
                "    artifacts_dir: Path",
                "    faiss_dir: Path",
                "    models_dir: Path",
                "    base_models_dir: Path",
                "    adapters_dir: Path",
                "",
                "",
                "def get_paths() -> ProjectPaths:",
                "    root = Path(__file__).resolve().parents[2]",
                "    data_dir = root / 'data'",
                "    raw_dir = data_dir / 'raw'",
                "    cleaned_dir = data_dir / 'cleaned'",
                "    rag_dir = data_dir / 'rag'",
                "    artifacts_dir = root / 'artifacts'",
                "    faiss_dir = artifacts_dir / 'faiss'",
                "    models_dir = root / 'models'",
                "    base_models_dir = models_dir / 'base'",
                "    adapters_dir = models_dir / 'adapters'",
                "    return ProjectPaths(",
                "        root=root,",
                "        data_dir=data_dir,",
                "        raw_dir=raw_dir,",
                "        cleaned_dir=cleaned_dir,",
                "        rag_dir=rag_dir,",
                "        artifacts_dir=artifacts_dir,",
                "        faiss_dir=faiss_dir,",
                "        models_dir=models_dir,",
                "        base_models_dir=base_models_dir,",
                "        adapters_dir=adapters_dir,",
                "    )",
                "",
            ]
        ),
    )

    ensure_file(root / "src" / "hotel_review_ai" / "data" / "__init__.py", "")
    ensure_file(
        root / "src" / "hotel_review_ai" / "data" / "clean.py",
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from typing import Any",
                "",
                "",
                "def normalize_review_text(title: str | None, text: str | None) -> str:",
                "    parts: list[str] = []",
                "    if title:",
                "        parts.append(title.strip())",
                "    if text:",
                "        parts.append(text.strip())",
                "    return \"\\n\".join([p for p in parts if p])",
                "",
                "",
                "def clean_record(record: dict[str, Any]) -> dict[str, Any]:",
                "    title = record.get('title')",
                "    text = record.get('text')",
                "    record = dict(record)",
                "    record['review'] = normalize_review_text(title, text)",
                "    return record",
                "",
            ]
        ),
    )

    ensure_file(root / "src" / "hotel_review_ai" / "rag" / "__init__.py", "")
    ensure_file(
        root / "src" / "hotel_review_ai" / "rag" / "build_index.py",
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def build_index() -> None:",
                "    raise RuntimeError('RAG index builder not implemented yet')",
                "",
            ]
        ),
    )
    ensure_file(
        root / "src" / "hotel_review_ai" / "rag" / "qa.py",
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def answer_question() -> None:",
                "    raise RuntimeError('RAG QA not implemented yet')",
                "",
            ]
        ),
    )

    ensure_file(root / "src" / "hotel_review_ai" / "llm" / "__init__.py", "")
    ensure_file(
        root / "src" / "hotel_review_ai" / "llm" / "hf.py",
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def load_llm() -> None:",
                "    raise RuntimeError('LLM loader not implemented yet')",
                "",
            ]
        ),
    )

    ensure_file(
        root / "apps" / "cli" / "main.py",
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def main() -> None:",
                "    raise RuntimeError('CLI entrypoint not implemented yet')",
                "",
                "",
                "if __name__ == '__main__':",
                "    main()",
                "",
            ]
        ),
    )

    ensure_file(
        root / "apps" / "web" / "gradio_app.py",
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def create_app() -> None:",
                "    raise RuntimeError('Web app not implemented yet')",
                "",
            ]
        ),
    )

    ensure_file(root / "tests" / "__init__.py", "")


if __name__ == '__main__':
    main()
