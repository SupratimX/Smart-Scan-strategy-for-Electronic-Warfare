"""conftest.py — add src to path at project root level too."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
