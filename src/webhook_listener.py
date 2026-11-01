import json
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

# Load environment variables (.env) from project root or current folder
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

# Reuse project's existing pipeline scripts
from fetch_nvd import DB_PATH as FETCH_DB_PATH, fetch_and_store_cves
from risk_engine import run_risk_engine

DB_PATH = FETCH_DB_PATH
CVSS_FLAG_THRESHOLD = 7.0
SCAN_INTERVAL_MINUTES = 15

CONFIG_PATH = Path(__file__).resolve().parent.parent / "monitoring_config.json"


def get_monitoring_mode():
    """Returns 'webhook', 'scheduled', or 'both'."""
    if not CONFIG_PATH.exists():
        return "both"
    try:
        return json.loads(CONFIG_PATH.read_text()).get("mode", "both")
    except Exception:
        return "both"


# ------------------------------------------------------------------------------
# 1. Background Scheduler Task
# ------------------------------------------------------------------------------
def scheduled_recheck():
    mode = get_monitoring_mode()
    if mode == "webhook":
        print("[scheduler] Skipped — monitoring mode is 'webhook' only.")
        return

    print(f"[scheduler] Running scheduled vulnerability re-scan at {datetime.now().isoformat()}")
    try:
        fetch_and_store_cves()
        run_risk_engine()
        matches = get_high_severity_matches()
        log_flag(
            repo_name="SCHEDULED_SCAN",
            changed_files=["(no push — periodic re-scan)"],
            commit_message=f"Automatic re-scan every {SCAN_INTERVAL_MINUTES} min",
            matches=matches,
        )
        print(f"[scheduler] Re-scan complete. {len(matches)} high-severity matches found.")
    except Exception as e:
        print(f"[scheduler] Re-scan failed: {e}")


# ------------------------------------------------------------------------------
# 2. Modern Lifespan Handler
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scheduled_recheck,
        "interval",
        minutes=SCAN_INTERVAL_MINUTES,
        id="scheduled_recheck",
        next_run_time=datetime.now(),  # Runs once immediately on startup
    )
    scheduler.start()
    print(f"[scheduler] Started — will re-scan every {SCAN_INTERVAL_MINUTES} minutes.")

    yield

    print("[scheduler] Shutting down...")
    scheduler.shutdown(wait=False)


app = FastAPI(title="Cyber Risk Webhook Listener", lifespan=lifespan)

# ------------------------------------------------------------------------------
# 3. File Extension / Tech Stack Detection
# ------------------------------------------------------------------------------
EXTENSION_TO_KEYWORD = {
    ".py": "python",
    ".js": "nodejs",
    ".ts": "nodejs",
    ".java": "java",
    ".go": "golang",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "dotnet",
    ".sql": "mysql",
    ".yml": "kubernetes",
    ".yaml": "kubernetes",
    ".tf": "terraform",
}

FILENAME_TO_KEYWORD = {
    "requirements.txt": "python",
    "pipfile": "python",
    "package.json": "nodejs",
    "pom.xml": "java",
    "build.gradle": "java",
    "go.mod": "golang",
    "gemfile": "ruby",
    "dockerfile": "docker",
    "docker-compose.yml": "docker",
}


def detect_tech_keywords(changed_files):
    keywords = set()
    for f in changed_files:
        base = f.split("/")[-1].lower()
        if base in FILENAME_TO_KEYWORD:
            keywords.add(FILENAME_TO_KEYWORD[base])
            continue
        ext = "." + base.split(".")[-1] if "." in base else ""
        if ext in EXTENSION_TO_KEYWORD:
            keywords.add(EXTENSION_TO_KEYWORD[ext])
    return keywords


def fetch_cve_for_keyword(keyword, cursor, results_per_page=5):
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={keyword}&resultsPerPage={results_per_page}"
    headers = {"User-Agent": "CyberRiskListener/2.0"}
    nvd_key = os.getenv("NVD_API_KEY")
    if nvd_key:
        headers["apiKey"] = nvd_key

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        vulnerabilities = data.get("vulnerabilities", [])

        for item in vulnerabilities:
            cve = item.get("cve", {})
            cve_id = cve.get("id")
            descriptions = cve.get("descriptions", [])
            desc = next((d.get("value") for d in descriptions if d.get("lang") == "en"), "")
            metrics = cve.get("metrics", {})
            cvss_data = metrics.get("cvssMetricV31", metrics.get("cvssMetricV30", []))
            severity = cvss_data[0].get("cvssData", {}).get("baseScore", 0.0) if cvss_data else 0.0

            cursor.execute("""
                INSERT OR REPLACE INTO cves (cve_id, keyword, severity_score, description)
                VALUES (?, ?, ?, ?)
            """, (cve_id, keyword, severity, desc))

        return len(vulnerabilities)
    except Exception as e:
        print(f"  -> Error fetching CVEs for keyword '{keyword}': {e}")
        return 0


# ------------------------------------------------------------------------------
# 4. Database Helper Functions
# ------------------------------------------------------------------------------
def get_conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vulnerability_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            repo_name TEXT,
            changed_files TEXT,
            commit_message TEXT,
            high_severity_count INTEGER,
            top_cves TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            eal REAL,
            var_95 REAL,
            risk_level TEXT,
            timestamp TEXT,
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
        )
    """)
    conn.commit()
    return conn


def get_high_severity_matches(threshold=CVSS_FLAG_THRESHOLD, limit=5):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.asset_id, a.asset_name, c.cve_id, c.severity_score
        FROM assets a
        JOIN cves c ON a.tech_keyword = c.keyword
        WHERE c.severity_score >= ?
        ORDER BY c.severity_score DESC
        LIMIT ?
    """, (threshold, limit))
    rows = cursor.fetchall()
    conn.close()
    return rows


def log_flag(repo_name, changed_files, commit_message, matches):
    conn = get_conn()
    top_cves = ", ".join(f"{r[2]} ({r[1]}, {r[3]})" for r in matches) if matches else "none"
    conn.execute(
        """INSERT INTO vulnerability_flags
           (timestamp, repo_name, changed_files, commit_message, high_severity_count, top_cves)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), repo_name, ", ".join(changed_files), commit_message, len(matches), top_cves),
    )
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# 5. Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "Webhook listener is active and running", "mode": get_monitoring_mode()}


@app.get("/monitoring/status")
def monitoring_status():
    return {
        "status": "running",
        "mode": get_monitoring_mode(),
        "scan_interval_minutes": SCAN_INTERVAL_MINUTES,
        "timestamp": datetime.now().isoformat(),
    }


class NewAsset(BaseModel):
    asset_id: str
    asset_name: str
    tech_keyword: str
    criticality: int = 3
    mfa_enabled: int = 0
    patched: int = 0
    firewall: int = 0
    network_segmented: int = 0


@app.post("/assets/add")
def add_asset(asset: NewAsset):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO assets
        (asset_id, asset_name, tech_keyword, criticality, mfa_enabled, patched, firewall, network_segmented)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        asset.asset_id, asset.asset_name, asset.tech_keyword, asset.criticality,
        asset.mfa_enabled, asset.patched, asset.firewall, asset.network_segmented,
    ))
    conn.commit()
    conn.close()

    fetch_and_store_cves()
    run_risk_engine()

    return {"status": "asset added and risk-evaluated", "asset_id": asset.asset_id}


@app.post("/webhooks/github")
async def github_webhook(request: Request):
    payload = await request.json()
    event_type = request.headers.get("X-GitHub-Event", "unknown")

    if event_type == "ping":
        return {"status": "pong"}

    if event_type != "push":
        return {"status": "ignored", "reason": f"event type '{event_type}' not handled"}

    mode = get_monitoring_mode()
    if mode == "scheduled":
        return {"status": "ignored", "reason": "monitoring mode is 'scheduled' only"}

    repo_name = payload.get("repository", {}).get("name", "unknown-repo")
    commits = payload.get("commits", [])

    changed_files = []
    commit_message = ""
    for commit in commits:
        changed_files += commit.get("added", []) + commit.get("modified", [])
        commit_message = commit.get("message", commit_message)

    detected_keywords = detect_tech_keywords(changed_files)
    if not detected_keywords:
        return {"status": "ignored", "reason": "no recognizable tech stack in changed files", "changed_files": changed_files}

    conn = get_conn()
    cursor = conn.cursor()
    cve_counts = {}
    for keyword in detected_keywords:
        cve_counts[keyword] = fetch_cve_for_keyword(keyword, cursor)
    conn.commit()
    conn.close()

    # Recalculate risk & log history
    run_risk_engine()

    matches = get_high_severity_matches()
    log_flag(repo_name, changed_files, commit_message, matches)

    return {
        "status": "processed",
        "repo": repo_name,
        "changed_files": changed_files,
        "detected_tech_keywords": list(detected_keywords),
        "new_cves_found_per_keyword": cve_counts,
        "high_severity_matches_found": len(matches),
    }


if __name__ == "__main__":
    uvicorn.run("webhook_listener:app", host="0.0.0.0", port=8001, reload=True)FastAPI + Uvicorn for webhook/API-style backend monitoring
    
    