import sqlite3
import json
from pathlib import Path

# Set paths relative to the project root directory
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "risk_platform.db"
SCHEMA_PATH = BASE_DIR / "db" / "schema.sql"
ASSETS_JSON = BASE_DIR / "data" / "assets.json"

def init_db():
    # Step 1: Connect to SQLite (creates risk_platform.db if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Step 2: Read and execute the SQL schema to create empty tables
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
        cursor.executescript(schema_sql)
    print("Tables created successfully.")

    # Migrate databases created before the EAL terminology change.
    risk_columns = {row[1] for row in cursor.execute("PRAGMA table_info(risk_results)")}
    if "ale" in risk_columns and "eal" not in risk_columns:
        cursor.execute("ALTER TABLE risk_results RENAME COLUMN ale TO eal")

    optimization_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(optimization_results)")
    }
    if "mitigated_ale" in optimization_columns and "mitigated_eal" not in optimization_columns:
        cursor.execute(
            "ALTER TABLE optimization_results RENAME COLUMN mitigated_ale TO mitigated_eal"
        )

    # Step 3: Populate the 'assets' table with data from assets.json (if present)
    if ASSETS_JSON.exists():
        with open(ASSETS_JSON, 'r') as f:
            assets = json.load(f)
            
            for asset in assets:
                cursor.execute("""
                    INSERT OR REPLACE INTO assets 
                    (asset_id, asset_name, criticality, tech_keyword, mfa_enabled, patched, firewall, network_segmented)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    asset.get("asset_id"),
                    asset.get("asset_name"),
                    asset.get("criticality"),
                    asset.get("tech_keyword"),
                    int(asset.get("mfa_enabled", 0)),
                    int(asset.get("patched", 0)),
                    int(asset.get("firewall", 0)),
                    int(asset.get("network_segmented", 0))
                ))
        print(f"Loaded {len(assets)} initial assets into the 'assets' table.")
    else:
        print("assets.json not found in data/ — skipped asset seeding.")

    # Step 4: Commit changes and close connection
    conn.commit()
    conn.close()
    print("Database initialization complete: db/risk_platform.db")

if __name__ == "__main__":
    init_db()