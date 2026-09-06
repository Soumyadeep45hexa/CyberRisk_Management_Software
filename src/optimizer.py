import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

# Define database path relative to this file
DB_PATH = Path(__file__).parent.parent / "db" / "risk_platform.db"

def calculate_mitigated_risk(asset, cve_scores, controls_override=None):
    """
    Recalculates EAL after applying target security controls to evaluate risk reduction.
    """
    criticality = asset['criticality']
    
    # Use override controls if provided, otherwise use baseline asset controls
    mfa = controls_override.get('mfa_enabled', asset['mfa_enabled']) if controls_override else asset['mfa_enabled']
    patched = controls_override.get('patched', asset['patched']) if controls_override else asset['patched']
    firewall = controls_override.get('firewall', asset['firewall']) if controls_override else asset['firewall']
    segmented = controls_override.get('network_segmented', asset['network_segmented']) if controls_override else asset['network_segmented']
    
    # Calculate Vulnerability Factor based on active security controls
    vuln_factor = 1.0
    if mfa: vuln_factor *= 0.7
    if patched: vuln_factor *= 0.5
    if firewall: vuln_factor *= 0.8
    if segmented: vuln_factor *= 0.85
    
    # Base Threat Event Frequency (TEF) from average CVE severity
    avg_cve_severity = np.mean(cve_scores) if len(cve_scores) > 0 else 5.0
    base_tef = (avg_cve_severity / 10.0) * 12
    
    # Run Monte Carlo Simulation (1,000 iterations)
    simulations = 1000
    simulated_losses = []
    
    for _ in range(simulations):
        threat_attempts = np.random.poisson(lam=base_tef)
        successful_breaches = np.random.binomial(n=threat_attempts, p=vuln_factor)
        
        if successful_breaches > 0:
            base_loss = (10 ** (criticality / 2.5)) * 100
            single_loss = np.random.lognormal(mean=np.log(base_loss), sigma=0.4)
            total_annual_loss = successful_breaches * single_loss
        else:
            total_annual_loss = 0.0
            
        simulated_losses.append(total_annual_loss)
        
    return float(np.mean(simulated_losses))

def run_optimization():
    if not DB_PATH.exists():
        print(f"Error: Database file not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Fetch assets and baseline risk results
    cursor.execute("""
        SELECT a.*, r.eal AS baseline_eal
        FROM assets a 
        LEFT JOIN risk_results r ON a.asset_id = r.asset_id
    """)
    assets = cursor.fetchall()
    
    if not assets:
        print("No assets found in database. Please run db/db_setup.py and src/risk_engine.py first.")
        conn.close()
        return

    # Typical estimated implementation costs per control per asset ($/year)
    CONTROL_COSTS = {
        'mfa_enabled': 1200.0,
        'patched': 2500.0,
        'firewall': 1800.0,
        'network_segmented': 3000.0
    }

    total_optimized = 0

    for asset in assets:
        cursor.execute("SELECT severity_score FROM cves WHERE keyword = ?", (asset['tech_keyword'],))
        cve_rows = cursor.fetchall()
        cve_scores = [r['severity_score'] for r in cve_rows if r['severity_score'] is not None]
        
        baseline_eal = asset['baseline_eal'] if asset['baseline_eal'] is not None else calculate_mitigated_risk(asset, cve_scores)
        
        # Simulate fully mitigated state (enabling all missing controls)
        mitigated_controls = {
            'mfa_enabled': 1,
            'patched': 1,
            'firewall': 1,
            'network_segmented': 1
        }
        
        # Calculate cost of adding missing controls
        cost_to_mitigate = 0.0
        if not asset['mfa_enabled']: cost_to_mitigate += CONTROL_COSTS['mfa_enabled']
        if not asset['patched']: cost_to_mitigate += CONTROL_COSTS['patched']
        if not asset['firewall']: cost_to_mitigate += CONTROL_COSTS['firewall']
        if not asset['network_segmented']: cost_to_mitigate += CONTROL_COSTS['network_segmented']
        
        mitigated_eal = calculate_mitigated_risk(asset, cve_scores, mitigated_controls)
        risk_reduction = max(0.0, baseline_eal - mitigated_eal)
        
        # Calculate Return on Security Investment (ROSI %)
        if cost_to_mitigate > 0:
            rosi = ((risk_reduction - cost_to_mitigate) / cost_to_mitigate) * 100
        else:
            rosi = 0.0  # Already fully protected
            
        # Select best recommended action
        if cost_to_mitigate == 0:
            action = "Maintain current control architecture"
        elif rosi > 50:
            action = "High Priority: Implement missing controls (Positive ROI)"
        elif rosi > 0:
            action = "Medium Priority: Implement missing controls (Moderate ROI)"
        else:
            action = "Low Priority: Risk reduction costs exceed expected loss"

        # Store in optimization_results table
        cursor.execute("""
            INSERT OR REPLACE INTO optimization_results 
            (asset_id, mitigated_eal, cost_to_mitigate, rosi, recommended_action)
            VALUES (?, ?, ?, ?, ?)
        """, (asset['asset_id'], mitigated_eal, cost_to_mitigate, rosi, action))
        
        total_optimized += 1

    conn.commit()
    conn.close()
    print(f"Optimization complete! Saved remediation recommendations for {total_optimized} assets into 'optimization_results'.")

if __name__ == "__main__":
    run_optimization()