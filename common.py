import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
os.environ['PROJECT_ROOT'] = str(PROJECT_ROOT)
