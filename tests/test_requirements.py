"""Every package the code imports must be in a requirements file, with a minimum version.

This runs in CI, so a pull request that adds an import without the matching
requirements line turns tests-pass red and can't merge. The failure message
says which file imports what, and the line to add.

    requirements.txt      core: data pulls, features, rankers, tests (CI installs this)
    requirements-ml.txt   GPU and language-model packages (CI doesn't install these)
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ_FILES = ["requirements.txt", "requirements-ml.txt"]
SKIP_DIRS = {".git", ".venv", "venv", "env", "data", "__pycache__", ".github", "node_modules"}

# Import name -> pip name, where they differ. Add to this if the test misreads a package.
IMPORT_TO_PIP = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "dotenv": "python-dotenv",
    "huggingface_hub": "huggingface-hub",
    "google": "google-genai",  # `from google import genai`
}
ML_PACKAGES = {"torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets", "vllm",
               "trl", "safetensors", "sentencepiece", "tokenizers", "huggingface-hub"}


def _norm(name: str) -> str:
    """pip treats exchange_calendars, exchange-calendars and Exchange.Calendars as one name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _python_files() -> list[Path]:
    return [p for p in ROOT.rglob("*.py") if not SKIP_DIRS & set(p.relative_to(ROOT).parts)]


def _local_modules() -> set[str]:
    """This repo's own modules: root .py files and folders with an __init__.py."""
    files = {p.stem for p in ROOT.glob("*.py")}
    packages = {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    return files | packages | {"tests", "conftest", "__future__"}


def third_party_imports() -> dict[str, set[str]]:
    """pip name -> the files that import it."""
    local = _local_modules()
    found: dict[str, set[str]] = {}
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top in sys.stdlib_module_names or top in local:
                    continue
                pip_name = _norm(IMPORT_TO_PIP.get(top, top))
                found.setdefault(pip_name, set()).add(path.relative_to(ROOT).as_posix())
    return found


def requirements() -> dict[str, tuple[str, str]]:
    """Normalized package name -> (requirements file, the line as written)."""
    out: dict[str, tuple[str, str]] = {}
    for fname in REQ_FILES:
        path = ROOT / fname
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#")[0].strip()
            if not line or line.startswith("-"):
                continue
            name = re.split(r"[\s<>=!~;\[]", line, maxsplit=1)[0]
            out[_norm(name)] = (fname, line)
    return out


def test_every_import_is_in_a_requirements_file():
    listed = requirements()
    missing = {pkg: files for pkg, files in third_party_imports().items() if pkg not in listed}
    lines = [
        f"  {', '.join(sorted(files))} imports {pkg}: add '{pkg}>=X.Y' to "
        f"{'requirements-ml.txt' if pkg in ML_PACKAGES else 'requirements.txt'} "
        f"(X.Y = your version, from: pip show {pkg})"
        for pkg, files in sorted(missing.items())
    ]
    assert not missing, "Packages imported but not in a requirements file:\n" + "\n".join(lines)


def test_every_requirement_has_a_minimum_version():
    bad = [f"  {fname}: '{line}'" for fname, line in requirements().values()
           if not re.search(r"(>=|==|~=)\s*\d", line)]
    assert not bad, "Add a minimum version (package>=X.Y, from pip show) to:\n" + "\n".join(bad)
