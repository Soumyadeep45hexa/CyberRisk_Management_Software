import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

# Define database path relative to this file
DB_PATH = Path(__file__).parent.parent / "db" / "risk_platform.db"

def calculate_asset_risk(asset, cve_scores):
    """
    FAIR (Factor Analysis of Information Risk) / Monte Carlo simulation model.
    Calculates Expected Annual Loss (EAL) and 95th Percentile Value at Risk (VaR).
    """
    criticality = asset['criticality']
    mfa = asset['mfa_enabled']
    patched = asset['patched']
    firewall = asset['firewall']
    segmented = asset['network_segmented']
    
    # Calculate Vulnerability Factor based on active security controls
    # Each control reduces exposure/probability of successful breach
    vuln_factor = 1.0
    if mfa: vuln_factor *= 0.7
    if patched: vuln_factor *= 0.5
    if firewall: vuln_factor *= 0.8
    if segmented: vuln_factor *= 0.85
    
    # Base Threat Event Frequency (TEF) from average CVE severity for asset tech stack
    avg_cve_severity = np.mean(cve_scores) if len(cve_scores) > 0 else 5.0
    base_tef = (avg_cve_severity / 10.0) * 12  # Estimated threat attempts per year
    
    # Run Monte Carlo Simulation (1,000 iterations)
    simulations = 1000
    simulated_losses = []
    
    for _ in range(simulations):
        # Sample Threat Event Frequency via Poisson distribution
        threat_attempts = np.random.poisson(lam=base_tef)
        # Sample successful breaches based on vulnerability factor
        successful_breaches = np.random.binomial(n=threat_attempts, p=vuln_factor)
        
        if successful_breaches > 0:
            # Loss Magnitude (LM) scales exponentially with asset criticality ($1k to $1M range)
            base_loss = (10 ** (criticality / 2.5)) * 100
            # Sample loss magnitude from log-normal distribution to reflect real-world tail risk
            single_loss = np.random.lognormal(mean=np.log(base_loss), sigma=0.4)
            total_annual_loss = successful_breaches * single_loss
        else:
            total_annual_loss = 0.0
            
        simulated_losses.append(total_annual_loss)
        
    eal = float(np.mean(simulated_losses))
    var_95 = float(np.percentile(simulated_losses, 95))  # Value at Risk (95th percentile)
    
    return eal, var_95

def run_risk_engine():
    if not DB_PATH.exists():
        print(f"Error: Database file not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Fetch all normalized assets from the database
    cursor.execute("SELECT * FROM assets")
    assets = cursor.fetchall()
    
    if not assets:
        print("No assets found in database. Please run db/db_setup.py first.")
        conn.close()
        return

    total_processed = 0
    
    for asset in assets:
        # Fetch matching CVE severity scores for asset tech_keyword
        cursor.execute("SELECT severity_score FROM cves WHERE keyword = ?", (asset['tech_keyword'],))
        cve_rows = cursor.fetchall()
        cve_scores = [r['severity_score'] for r in cve_rows if r['severity_score'] is not None]
        
        # Calculate quantitative FAIR / Monte Carlo risk metrics
        eal, var_95 = calculate_asset_risk(asset, cve_scores)
        
        # Determine overall Risk Level badge based on Expected Annual Loss
        if eal > 50000:
            risk_level = "CRITICAL"
        elif eal > 15000:
            risk_level = "HIGH"
        elif eal > 5000:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        # Store calculated results in risk_results table
        cursor.execute("""
            INSERT OR REPLACE INTO risk_results (asset_id, eal, var_95, risk_level)
            VALUES (?, ?, ?, ?)
        """, (asset['asset_id'], eal, var_95, risk_level))
        
        total_processed += 1
        
    conn.commit()
    conn.close()
    print(f"Risk calculation complete! Successfully processed {total_processed} assets into 'risk_results' table.")

if __name__ == "__main__":
    run_risk_engine()