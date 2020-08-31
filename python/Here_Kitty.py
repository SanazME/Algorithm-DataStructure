#!/usr/bin/env python3
import sys
from pathlib import Path

pathFile = sys.argv[1]
file_open = Path(pathFile)
print(file_open.read_text())
