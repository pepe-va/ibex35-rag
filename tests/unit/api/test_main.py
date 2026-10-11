"""Unit tests for the application entry point."""

import subprocess
import sys


def test_importing_the_app_does_not_import_langchain():
    """LangSmith env vars are set in main.py; LangChain must not be imported before that."""
    code = (
        "import sys, src.api.main; "
        "loaded = sorted(m for m in sys.modules if m.startswith(('langchain', 'langgraph'))); "
        "sys.exit(f'LangChain imported at startup: {loaded}' if loaded else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
