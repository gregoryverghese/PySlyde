"""
Execute the Python code blocks of the documentation, so the documented API
cannot drift away from the real one again (JOSS review, issue #52).

Every ``.. code-block:: python`` (reST) or fenced ``python`` block (Markdown,
for the README) in a page is run, in order, in one shared
namespace and in a temporary working directory. The pages use the example
dataset (:mod:`pyslyde.datasets`); the Quick Start also downloads the
torchvision ResNet-18 weights on first use.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
PAGES = ["docs/quickstart.rst", "docs/index.rst", "docs/example_data.rst", "README.md"]

_RST_BLOCK = re.compile(
    r"^\.\. code-block:: python\s*\n((?:\n|[ \t]+.*\n)+)", re.MULTILINE
)
_MD_BLOCK = re.compile(r"^```python\s*\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _python_blocks(page: Path) -> list[str]:
    text = page.read_text(encoding="utf-8")
    pattern = _MD_BLOCK if page.suffix == ".md" else _RST_BLOCK
    return [textwrap.dedent(m.group(1)).strip("\n") for m in pattern.finditer(text)]


def test_pages_have_python_blocks():
    for page in PAGES:
        assert _python_blocks(REPO / page), f"no python code blocks found in {page}"


@pytest.mark.parametrize("page", PAGES)
def test_doc_code_blocks_run(page, integration_data_dir, tmp_path, monkeypatch):
    """Run all Python blocks of ``page`` sequentially in a fresh namespace."""
    monkeypatch.chdir(tmp_path)
    # Make example_data_dir() return the integration data selected by conftest.
    monkeypatch.setenv("PYSLYDE_DATA_DIR", str(integration_data_dir))

    namespace: dict = {"__name__": "__docs__"}
    for i, block in enumerate(_python_blocks(REPO / page), start=1):
        try:
            exec(compile(block, f"{page}[block {i}]", "exec"), namespace)
        except Exception as e:  # pragma: no cover - failure path
            pytest.fail(
                f"{page}: code block {i} failed with {type(e).__name__}: {e}\n\n{block}",
                pytrace=False,
            )
