from pathlib import Path

ROOTS = ["engine", "apps"]

for root in ROOTS:
    root_path = Path(root)

    if not root_path.exists():
        continue

    for file in root_path.rglob("*.py"):
        print(f"\n{'='*80}")
        print(file)
        print(f"{'='*80}\n")

        print(file.read_text(encoding="utf-8"))