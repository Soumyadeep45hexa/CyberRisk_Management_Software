import os
import sqlite3
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

NVD_API_KEY = os.getenv("NVD_API_KEY")
DB_PATH = Path(__file__).parent.parent / "db" / "risk_platform.db"

def fetch_and_store_cves():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT tech_keyword FROM assets WHERE tech_keyword IS NOT NULL")
    keywords = [row[0] for row in cursor.fetchall()]

    print(f"Found keywords to search: {keywords}")

    headers = {}
    if NVD_API_KEY:
        headers["apiKey"] = NVD_API_KEY
        print("Using NVD API Key from .env")
    else:
        print("No NVD_API_KEY found in .env, proceeding with public rate limits...")

    total_cves_saved = 0

    for keyword in keywords:
        print(f"Fetching vulnerabilities for: {keyword}...")
        url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={keyword}&resultsPerPage=10"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            vulnerabilities = data.get("vulnerabilities", [])
            print(f"  -> Found {len(vulnerabilities)} CVEs for {keyword}")

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
                
                total_cves_saved += 1

        except Exception as e:
            print(f"  -> Error fetching data for {keyword}: {e}")

    conn.commit()
    conn.close()
    print(f"\nFinished! Successfully saved {total_cves_saved} CVE records to database.")

if __name__ == "__main__":
    fetch_and_store_cves()