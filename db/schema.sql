CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    asset_name TEXT NOT NULL,
    tech_keyword TEXT,
    criticality INTEGER DEFAULT 3,
    mfa_enabled INTEGER DEFAULT 0,
    patched INTEGER DEFAULT 0,
    firewall INTEGER DEFAULT 0,
    network_segmented INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cves (
    cve_id TEXT PRIMARY KEY,
    keyword TEXT,
    severity_score REAL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS risk_results (
    asset_id TEXT PRIMARY KEY,
    eal REAL,
    var_95 REAL,
    risk_level TEXT,
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

CREATE TABLE IF NOT EXISTS optimization_results (
    asset_id TEXT PRIMARY KEY,
    mitigated_eal REAL,
    cost_to_mitigate REAL,
    rosi REAL,
    recommended_action TEXT,
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);