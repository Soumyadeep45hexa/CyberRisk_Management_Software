"""
GitHub Webhook Listener — Continuous Vulnerability Re-Scan
-------------------------------------------------------------
Listens for GitHub 'push' events. When a push modifies a dependency
file (requirements.txt, package.json, etc.), it re-triggers the
project's existing fetch_nvd.py logic (fetch_and_store_cves), which
already knows how to pull CVEs for every tech_keyword tracked in the
'assets' table. This avoids re-implementing keyword matching here.

After the refresh, it logs a summary audit entry to 'vulnerability_flags'
so the dashboard can show "what changed and why" — separate from the
'cves' master table, which fetch_nvd.py already maintains.

Project layout (matches the real SIH project structure):
    SIH/
    ├── dashboard.py
    ├── src/
    │   ├── fetch_nvd.py      <- existing script, reused here
    │   ├── risk_engine.py    <- existing script, reused here
    │   ├── optimizer.py
    │   └── webhook_listener.py   <- THIS FILE GOES HERE, in src/
    └── db/
        ├── risk_platform.db
        └── schema.sql

Run from inside src/:
    cd src
    uvicorn webhook_listener:app --port 8001 --reload

Then expose it publicly with ngrok:
    ngrok http 8001

Register the resulting https://xxxx.ngrok.io/webhooks/github URL
as a webhook on your GitHub repo (Settings -> Webhooks -> Add webhook,
Content type: application/json, event: "Just the push event").
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request

# Reuse the project's existing scripts instead of duplicating their logic.
# fetch_nvd.py and risk_engine.py must live in the same src/ folder as this file.
from fetch_nvd import fetch_and_store_cves, DB_PATH as FETCH_DB_PATH
from risk_engine import run_risk_engine

app = FastAPI(title="Cyber Risk Webhook Listener")

# Use the exact same DB path fetch_nvd.py already uses, so there is only
# ever one source of truth for the database location.
DB_PATH = FETCH_DB_PATH
CVSS_FLAG_THRESHOLD = 7.0

# ------------------------------------------------------------------------------
# Shared monitoring mode config — the dashboard writes to this file when the
# customer picks a mode ("webhook", "scheduled", or "both"); this listener
# reads it before acting on either mechanism, so the customer's choice
# actually controls what runs, without needing to restart this process.
# Lives at the project root so both dashboard.py (root) and this file
# (src/) agree on the same path.
# ------------------------------------------------------------------------------
import json

CONFIG_PATH = Path(__file__).resolve().parent.parent / "monitoring_config.json"


def get_monitoring_mode():
    """Returns 'webhook', 'scheduled', or 'both'. Defaults to 'both' if no
    config file exists yet (e.g. before the customer has made a choice)."""
    if not CONFIG_PATH.exists():
        return "both"
    try:
        return json.loads(CONFIG_PATH.read_text()).get("mode", "both")
    except Exception:
        return "both"

# ------------------------------------------------------------------------------
# Scheduled re-scan — catches NEW vulnerabilities on EXISTING, unchanged
# assets. The GitHub webhook below only fires when code is pushed; this
# runs independently on a timer, because new CVEs get published for the
# same tech stack even when nothing in the company's environment changes.
# Together these two mechanisms are what make the platform "continuous":
#   - webhook  -> reacts to code/asset changes (event-driven)
#   - scheduler -> reacts to the passage of time / new CVE publications
# ------------------------------------------------------------------------------
from apscheduler.schedulers.background import BackgroundScheduler

# For a live demo, a short interval (minutes) proves it works without
# waiting hours. In a real deployment this would be much longer (e.g.
# once every 24 hours) to respect NVD's public rate limits.
SCAN_INTERVAL_MINUTES = 15


def scheduled_recheck():
    """Runs on a timer, independent of any push. Re-checks every
    tech_keyword already known to the assets table against NVD, then
    recalculates risk for everything -- exactly like a manual re-run of
    fetch_nvd.py + risk_engine.py, just automatic. Skips itself entirely
    if the customer has selected webhook-only mode."""
    mode = get_monitoring_mode()
    if mode == "webhook":
        print(f"[scheduler] Skipped — monitoring mode is 'webhook' only.")
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


@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scheduled_recheck,
        "interval",
        minutes=SCAN_INTERVAL_MINUTES,
        id="scheduled_recheck",
        next_run_time=datetime.now(),  # also run once immediately on startup
    )
    scheduler.start()
    print(f"[scheduler] Started — will re-scan every {SCAN_INTERVAL_MINUTES} minutes.")

import re
import requests

# ------------------------------------------------------------------------------
# Tech-stack detection from ANY changed file, not just dependency manifests.
# Maps a file's extension/name to a keyword that's meaningful to search on
# NVD. This is intentionally a simple, editable lookup table — extend it
# with whatever tech stacks your assets actually use.
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
    """Looks at every changed file (any type) in the push and returns the
    set of tech keywords implied by them — e.g. a new .py file implies
    'python', a Dockerfile implies 'docker'. This replaces the old
    filename-only dependency check, so ANY relevant file change triggers
    a check, not just requirements.txt/package.json edits."""
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
    """Targeted NVD lookup for exactly one keyword (not the whole assets
    table like fetch_and_store_cves does). Stores any results into the
    same 'cves' table so risk_engine.py picks them up on its next run via
    the existing assets.tech_keyword = cves.keyword join."""
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={keyword}&resultsPerPage={results_per_page}"
    try:
        response = requests.get(url, timeout=15)
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
# DB helpers — purely additive tables, nothing here touches assets/cves/
# risk_results/optimization_results, which schema.sql already owns.
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
    """Joins assets to cves on tech_keyword = keyword (same join your FAIR
    pipeline uses) to find current high-severity matches, for the audit log."""
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
        (datetime.now().isoformat(), repo_name, ", ".join(changed_files),
         commit_message, len(matches), top_cves),
    )
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Webhook endpoint
# ------------------------------------------------------------------------------
@app.get("/")
def health_check():
    """Visit http://localhost:8001 in a browser to confirm this is running."""
    return {"status": "webhook listener is running", "monitoring_mode": get_monitoring_mode()}


@app.get("/monitoring/status")
def monitoring_status():
    """Lets the dashboard display the currently active monitoring mode."""
    return {
        "mode": get_monitoring_mode(),
        "scan_interval_minutes": SCAN_INTERVAL_MINUTES,
    }


# ------------------------------------------------------------------------------
# New asset registration — real-time vulnerability check on a NEW tech stack
# ------------------------------------------------------------------------------
from pydantic import BaseModel


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
    """
    Call this whenever a new asset/tech stack is added to the organization
    (e.g. someone provisions a new service). This is a DIFFERENT trigger
    from the GitHub webhook above: that one reacts to code changes in an
    existing repo, this one reacts to a brand-new asset being registered
    that may use a tech stack never seen before (new tech_keyword).

    Without this, a new asset would sit with zero CVE data until someone
    manually re-ran fetch_nvd.py — this makes that happen immediately.
    """
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

    # Immediately check this (and every other) tech_keyword against NVD —
    # fetch_and_store_cves() already queries DISTINCT tech_keyword from
    # assets, so the new one is picked up automatically without needing a
    # separate single-keyword code path.
    fetch_and_store_cves()

    # Recalculate risk for all assets, including the new one, and log it
    # to risk_history so it immediately shows up in the trend view.
    run_risk_engine()

    return {
        "status": "asset added and risk-evaluated",
        "asset_id": asset.asset_id,
        "tech_keyword": asset.tech_keyword,
    }


@app.post("/webhooks/github")
async def github_webhook(request: Request):
    payload = await request.json()
    event_type = request.headers.get("X-GitHub-Event", "unknown")

    if event_type == "ping":
        # GitHub sends this once, automatically, when the webhook is first registered.
        return {"status": "pong"}

    if event_type != "push":
        return {"status": "ignored", "reason": f"event type '{event_type}' not handled"}

    mode = get_monitoring_mode()
    if mode == "scheduled":
        return {"status": "ignored", "reason": "monitoring mode is 'scheduled' only — webhook processing disabled by customer setting"}

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

    # Targeted check: query NVD specifically for the tech stacks implied by
    # THIS push, instead of blindly re-checking every keyword already in
    # the assets table. New CVEs get written into the same 'cves' table.
    conn = get_conn()
    cursor = conn.cursor()
    cve_counts = {}
    for keyword in detected_keywords:
        cve_counts[keyword] = fetch_cve_for_keyword(keyword, cursor)
    conn.commit()
    conn.close()

    # Recalculate risk for all assets so any matching assets (tech_keyword
    # in detected_keywords) immediately reflect the refreshed CVE data,
    # and log it to risk_history for the trend view.
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
        "matches": [
            {"asset_id": m[0], "asset_name": m[1], "cve_id": m[2], "cvss": m[3]}
            for m in matches
        ],
    }