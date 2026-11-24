"""Shared pytest configuration for public and internal test runs.

Tests that need local ``data/`` fixtures are marked ``requires_data``.
A clean public clone does not ship PDFs, Gold files or experiment
artifacts, so those tests are skipped automatically. Self-contained
tests always run.

Internal full run (with local fixtures)::

    PYTHONPATH=backend python -m pytest -q

Public run (no ``data/``)::

    PYTHONPATH=backend python -m pytest -q

See ``docs/DATA_LAYOUT.md`` for the fixture layout.
"""

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Sentinels that only exist in the internal, fixture-enabled checkout.
# Do not use ``data/`` itself: test runs may create an empty data directory.
DATA_SENTINELS = (
    PROJECT_ROOT / "data/samples/TPS7A02.pdf",
    PROJECT_ROOT / "data/outputs/TPS7A02.raw_structure.v1.json",
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5b/gold/gold.frozen.json",
)


def data_fixtures_available() -> bool:
    """Return whether local (non-published) data fixtures are present."""

    return all(path.is_file() for path in DATA_SENTINELS)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_data: test needs local data/ fixtures that are not "
        "published in the public repository",
    )


def pytest_collection_modifyitems(config, items):
    if data_fixtures_available():
        return

    skip = pytest.mark.skip(
        reason=(
            "缺少本地 data/ fixtures；公开仓库只运行自包含测试，"
            "fixture 说明见 docs/DATA_LAYOUT.md"
        )
    )
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT
