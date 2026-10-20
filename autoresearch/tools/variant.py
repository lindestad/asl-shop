"""variant.py NAME 'old=>new' ... : write autoresearch/exp_NAME.py = train.py with exact replacements.
A replacement given as @file reads 'old=>new' pairs separated by lines of '====' from file."""
import sys
from pathlib import Path
here = Path(__file__).resolve().parents[1]
name, specs = sys.argv[1], sys.argv[2:]
src = (here / "train.py").read_text()
pairs = []
for spec in specs:
    text = Path(spec[1:]).read_text() if spec.startswith("@") else spec
    for chunk in text.split("\n====\n"):
        if chunk.strip():
            old, new = chunk.split("=>", 1)
            pairs.append((old.strip("\n"), new.strip("\n")))
for old, new in pairs:
    if src.count(old) != 1:
        sys.exit(f"pattern found {src.count(old)} times: {old[:80]!r}")
    src = src.replace(old, new)
(here / f"exp_{name}.py").write_text(src)
print(f"wrote exp_{name}.py ({len(pairs)} edits)")
