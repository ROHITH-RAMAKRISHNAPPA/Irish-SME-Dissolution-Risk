import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "notebooks" / "03_eda.ipynb"

nb = json.loads(NB.read_text(encoding="utf-8"))

TERMS = ("test_audit", "triage_gains", "model_predictions", "curve_reconciliation")

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if not any(t in src for t in TERMS):
        continue
    print("=" * 78)
    print(f"CELL INDEX {i}")
    print("=" * 78)
    print(src)
    print()
