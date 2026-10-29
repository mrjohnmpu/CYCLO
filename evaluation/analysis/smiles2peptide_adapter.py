#!/usr/bin/env python3
"""Helpers to load the SMILES2PEPTIDE analyzer without its web dependencies."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_PROJECT_ROOT = Path(os.environ.get(
    "SMILES2PEPTIDE_ROOT", Path(__file__).resolve().parents[2] / "third_party" / "SMILES2PEPTIDE"
)).expanduser().resolve()


def _install_gradio_client_shim() -> None:
    if 'gradio_client' in sys.modules and 'gradio_client.utils' in sys.modules:
        return

    client = types.ModuleType('gradio_client')
    utils = types.ModuleType('gradio_client.utils')

    def _json_schema_to_python_type(schema: Any, defs: Any = None) -> str:
        return 'Any' if isinstance(schema, bool) else 'object'

    utils._json_schema_to_python_type = _json_schema_to_python_type
    utils.json_schema_to_python_type = _json_schema_to_python_type
    client.utils = utils
    sys.modules['gradio_client'] = client
    sys.modules['gradio_client.utils'] = utils


def _install_gradio_shim() -> None:
    if 'gradio' in sys.modules and 'gradio.blocks' in sys.modules:
        return

    class _Dummy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> '_Dummy':
            return self

        def __getattr__(self, name: str) -> '_Dummy':
            return self

        def __enter__(self) -> '_Dummy':
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def click(self, *args: Any, **kwargs: Any) -> '_Dummy':
            return self

        def launch(self, *args: Any, **kwargs: Any) -> '_Dummy':
            return self

        def queue(self, *args: Any, **kwargs: Any) -> '_Dummy':
            return self

        def submit(self, *args: Any, **kwargs: Any) -> '_Dummy':
            return self

    gradio = types.ModuleType('gradio')
    gradio.__path__ = []  # type: ignore[attr-defined]
    for name in [
        'Blocks',
        'Interface',
        'Textbox',
        'File',
        'Checkbox',
        'Image',
        'Markdown',
        'Row',
        'Column',
        'Tab',
        'Tabs',
        'Button',
        'Examples',
        'Accordion',
        'Group',
        'State',
        'Slider',
        'Radio',
        'Dropdown',
    ]:
        setattr(gradio, name, _Dummy)

    blocks = types.ModuleType('gradio.blocks')
    blocks.Block = _Dummy
    components = types.ModuleType('gradio.components')

    gradio.blocks = blocks  # type: ignore[attr-defined]
    gradio.components = components  # type: ignore[attr-defined]
    sys.modules['gradio'] = gradio
    sys.modules['gradio.blocks'] = blocks
    sys.modules['gradio.components'] = components


@lru_cache(maxsize=1)
def load_smiles2peptide_app(project_root: str | Path = DEFAULT_PROJECT_ROOT) -> Any:
    """Import the app module after installing tiny import shims."""

    project_root = Path(project_root)
    if not project_root.exists():
        raise FileNotFoundError(f'SMILES2PEPTIDE project root not found: {project_root}')

    _install_gradio_client_shim()
    _install_gradio_shim()

    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    app_path = project_root / 'app.py'
    spec = importlib.util.spec_from_file_location('smiles2peptide_app', app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load SMILES2PEPTIDE app from {app_path}')

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_peptide_analyzer(project_root: str | Path = DEFAULT_PROJECT_ROOT) -> Any:
    return load_smiles2peptide_app(project_root).PeptideAnalyzer()


def load_peptide_encoder(project_root: str | Path = DEFAULT_PROJECT_ROOT) -> Any:
    return load_smiles2peptide_app(project_root).PeptideEncoder()
