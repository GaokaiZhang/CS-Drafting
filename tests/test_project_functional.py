import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.functional


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, "main_fixed_window.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--small_window" in result.stdout
    assert "--middle_window" in result.stdout
    assert "--mode {baseline,hierarchical,compare}" in result.stdout


def test_ui_assets_exist_and_linked():
    index_path = REPO_ROOT / "ui" / "index.html"
    app_path = REPO_ROOT / "ui" / "app.js"
    demo_path = REPO_ROOT / "ui" / "demo_data.js"
    css_path = REPO_ROOT / "ui" / "styles.css"

    index_html = index_path.read_text()
    app_js = app_path.read_text()
    demo_js = demo_path.read_text()

    assert 'id="overview"' in index_html
    assert 'id="hierarchical"' in index_html
    assert 'id="baseline"' in index_html
    assert "./app.js" in index_html
    assert "./demo_data.js" in index_html
    assert "./styles.css" in index_html
    assert "window.DEMO_COMPARISON" in demo_js
    assert "loadData(window.DEMO_COMPARISON" in app_js
    assert css_path.exists()
