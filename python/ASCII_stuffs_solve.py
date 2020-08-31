#!/usr/bin/env python3
import sys
from pathlib import Path

pathFile = sys.argv[1]
file_open = Path(pathFile)
ff = file_open.read_text()
print(bytes.fromhex(ff).decode('utf-8'))
