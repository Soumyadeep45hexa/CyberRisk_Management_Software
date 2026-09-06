import pandas as pd
import json
from pathlib import Path

csv_path = Path("export-assets-2026-09-03.csv")
output_path = Path("data/assets.json")

df = pd.read_csv(csv_path)
tech_keywords = ["Apache", "PostgreSQL", "Nginx", "Linux", "MySQL"]
assets = []

for idx, row in df.iterrows():
    raw_tag = str(row.get('Asset Tag', '')).split('.')[0]
    asset_id = f"AST-{raw_tag}" if raw_tag else f"AST-{idx+1:03d}"
    
    name = row.get('Asset Name')
    asset_name = str(name).strip() if pd.notna(name) and str(name).strip() else f"{row.get('Model', 'Asset')} ({raw_tag})"
    
    assets.append({
        "asset_id": asset_id,
        "asset_name": asset_name,
        "criticality": (idx % 5) + 5,
        "tech_keyword": tech_keywords[idx % len(tech_keywords)],
        "mfa_enabled": 1 if idx % 2 == 0 else 0,
        "patched": 1 if idx % 3 == 0 else 0,
        "firewall": 1,
        "network_segmented": 1 if idx % 2 == 1 else 0
    })

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(assets, f, indent=2)

print(f"Saved {len(assets)} assets to {output_path}")