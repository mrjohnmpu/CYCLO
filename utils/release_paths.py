"""Repository-local defaults and explicit external dependency locations."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def external_root(name: str, env_var: str) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT / "third_party" / name


def add_moses_path() -> Path:
    root = external_root("moses", "MOSES_ROOT")
    # Accept either a checkout root or the directory containing __init__.py.
    import_root = root.parent if (root / "__init__.py").is_file() else root
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    return import_root


def peptiverse_root() -> Path:
    return external_root("PeptiVerse", "PEPTIVERSE_ROOT")
