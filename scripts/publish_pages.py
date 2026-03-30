from pathlib import Path
import json
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS_OUTPUTS = DOCS / "outputs"
DOCS_OUTPUTS.mkdir(parents=True, exist_ok=True)

# Candidate locations for generated outputs
candidate_dirs = [
    ROOT / "outputs",
    ROOT / "docs" / "outputs",
]

required = [
    "fig_binder_crossing.png",
    "fig_order_collapse.png",
    "fig_susceptibility_collapse.png",
    "fig_observables_grid.png",
]

copied = []
missing = []

for name in required:
    src = None
    for base in candidate_dirs:
        p = base / name
        if p.exists():
            src = p
            break
    if src is None:
        missing.append(name)
        continue
    dst = DOCS_OUTPUTS / name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    copied.append(name)

# Copy optional result JSON files if present
for optional in ["summary_results.json", "raw_results.json", "criticality_advanced_2d_results.json"]:
    for base in candidate_dirs + [ROOT]:
        p = base / optional
        if p.exists():
            shutil.copy2(p, DOCS_OUTPUTS / optional)
            break

manifest = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "copied_figures": copied,
    "missing_figures": missing,
}

(DOCS_OUTPUTS / "site_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if missing:
    raise SystemExit(f"Missing required figures: {missing}")
else:
    print("Pages bundle prepared successfully.")
