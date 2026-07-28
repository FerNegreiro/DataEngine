from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = (
    "data/raw",
    "data/processed",
    "data/samples",
    "docker",
    "docs",
    "pipelines/ingestion",
    "pipelines/processing",
    "pipelines/loading",
    "src/extraction",
    "src/transformation",
    "src/validation",
    "src/utils",
    "tests",
)

REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "pyproject.toml",
    ".gitignore",
)


def test_project_structure() -> None:
    missing_directories = [
        path for path in REQUIRED_DIRECTORIES if not (PROJECT_ROOT / path).is_dir()
    ]
    missing_files = [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).is_file()]

    assert not missing_directories, f"Pastas ausentes: {missing_directories}"
    assert not missing_files, f"Arquivos ausentes: {missing_files}"
