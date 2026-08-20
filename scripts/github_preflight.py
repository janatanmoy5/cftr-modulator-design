#!/usr/bin/env python3
"""Read-only GitHub publication checks for the CFTR project."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", ".git", "data", "models", "results", "__pycache__"}
SECRET = re.compile(r"(?i)(api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*['\"][^'\"]{8,}")
ABS_PATH = re.compile("/" + r"Users/[^/\s]+/")
errors, warnings = [], []

for path in sorted(ROOT.rglob("*")):
    rel = path.relative_to(ROOT)
    if any(part in SKIP for part in rel.parts):
        continue
    if path.is_file() and path.stat().st_size > 20 * 1024 * 1024:
        errors.append(f"oversized publication file: {rel} ({path.stat().st_size} bytes)")
    if path.suffix == ".py":
        try: ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        except SyntaxError as exc: errors.append(f"Python syntax: {rel}:{exc.lineno}: {exc.msg}")
    if path.is_file() and path.suffix.lower() in {".py", ".sh", ".md", ".yaml", ".yml", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        if SECRET.search(text): errors.append(f"possible embedded secret: {rel}")
        if ABS_PATH.search(text): warnings.append(f"user-specific absolute path: {rel}")

required = ["README.md", "requirements.txt", "environment.yml", ".gitignore",
            "app.py", "run_webapp.sh", "fullpipeline.sh", "config/config.yaml"]
for name in required:
    if not (ROOT / name).exists(): errors.append(f"missing required file: {name}")

for item in warnings: print(f"WARNING: {item}")
for item in errors: print(f"ERROR: {item}")
print(f"GitHub preflight: {len(errors)} error(s), {len(warnings)} warning(s)")
raise SystemExit(1 if errors else 0)
