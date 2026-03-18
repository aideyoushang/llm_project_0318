from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    raw_dir: Path
    cleaned_dir: Path
    rag_dir: Path
    artifacts_dir: Path
    faiss_dir: Path
    models_dir: Path
    base_models_dir: Path
    adapters_dir: Path


def get_paths() -> ProjectPaths:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / 'data'
    raw_dir = data_dir / 'raw'
    cleaned_dir = data_dir / 'cleaned'
    rag_dir = data_dir / 'rag'
    artifacts_dir = root / 'artifacts'
    faiss_dir = artifacts_dir / 'faiss'
    models_dir = root / 'models'
    base_models_dir = models_dir / 'base'
    adapters_dir = models_dir / 'adapters'
    return ProjectPaths(
        root=root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        cleaned_dir=cleaned_dir,
        rag_dir=rag_dir,
        artifacts_dir=artifacts_dir,
        faiss_dir=faiss_dir,
        models_dir=models_dir,
        base_models_dir=base_models_dir,
        adapters_dir=adapters_dir,
    )
