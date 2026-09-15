"""conftest.py — shared pytest fixtures."""
import sys
from pathlib import Path

# Ensure src/ is on sys.path when running tests from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
