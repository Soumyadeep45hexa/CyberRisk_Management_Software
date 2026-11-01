"""
Cyber Risk Quantification Dashboard
------------------------------------
FAIR-style quantitative risk analysis using a numpy-vectorized Monte Carlo
engine with real-time NVD API vulnerability score enrichment.

Run with:  streamlit run dashboard.py
"""

import io
import json
import os
from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Load the project credentials even when Streamlit is launched from another directory.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)


# ------------------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Cyber Risk Quantification Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    /* Keep Streamlit's native drag-resize behavior.  The main workspace uses
       the remaining width automatically as the analyst changes the sidebar. */
    [data-testid="stSidebar"] { min-width: 0; max-width: none; }
    [data-testid="stMain"] { min-width: 0; }
    [data-testid="stSidebar"] [data-testid="stChatMessage"] { padding: .35rem 0; }
    .workflow-card {
        border: 1px solid rgba(148, 163, 184, .35); border-radius: 12px;
        padding: .9rem 1rem; min-height: 92px; background: rgba(15, 23, 42, .03);
    }
    .workflow-card h4 { margin: 0 0 .35rem 0; }
    .workflow-card p { margin: 0; color: #64748b; font-size: .87rem; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Cyber Risk Command Center")
st.caption("FAIR-model Monte Carlo simulation • NVD vulnerability enrichment • decision-ready cyber risk management")

CURRENCY_OPTIONS = {
    "Dollar (USD)": ("$", 1.0),
    "Rupee (INR)": ("₹", 83.0),
}
selected_currency = st.sidebar.selectbox(
    "Display currency",
    options=list(CURRENCY_OPTIONS),
    key="display_currency",
)
currency_symbol, currency_rate = CURRENCY_OPTIONS[selected_currency]


def format_money(value, decimals=0):
    converted = float(value or 0) * currency_rate
    return f"{currency_symbol}{converted:,.{decimals}f}"


def currency_label(label):
    return f"{label} ({selected_currency.split(' ')[-1].strip('()')})"

# This placeholder is deliberately created before every other sidebar element.
# The LLM chat is populated after the simulation has calculated its context,
# but remains visually pinned at the top of the left sidebar.
assistant_slot = st.sidebar.empty()
st.sidebar.caption("🔧 Build: investment-optimizer-v3")

st.markdown("### Workspace")
st.caption("Select a workspace to view only its controls, analysis, and results.")
asset_tab, eal_tab, optimization_tab, monitoring_tab, integrations_tab = st.tabs([
    "📁 Asset Upload",
    "📊 EAL Calculations",
    "💰 Optimization",
    "🗄️ Live Monitoring",
    "🔗 Integrations",
])

SEVERITY_COLORS = {
    "Critical": "#b91c1c",
    "High": "#ef4444",
    "Medium": "#f97316",
    "Low": "#22c55e",
}

# ------------------------------------------------------------------------------
# 2. Groq client (optional — chat degrades gracefully if key is missing)
# ------------------------------------------------------------------------------
from groq import Groq

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = None
groq_error = None
if groq_api_key:
    try:
        groq_client = Groq(api_key=groq_api_key)
    except Exception as exc:
        groq_error = str(exc)
else:
    groq_error = "No GROQ_API_KEY found in your .env file"

# ------------------------------------------------------------------------------
# 3. Monte Carlo engine (numpy, fully vectorized)
# ------------------------------------------------------------------------------
def pert_sample(min_val, mode, max_val, size, rng, lam=4.0):
    min_val, mode, max_val = sorted([min_val, mode, max_val])
    rng_span = max_val - min_val
    if rng_span <= 0:
        return np.full(size, min_val, dtype=float)
    mu = (min_val + lam * mode + max_val) / (lam + 2)
    alpha = 1 + lam * (mu - min_val) / rng_span
    beta = 1 + lam * (max_val - mu) / rng_span
    return min_val + rng.beta(alpha, beta, size=size) * rng_span


def simulate_asset(row, n_sim, rng, max_events_cap=12, upstream_breached=None):
    tef = pert_sample(row["tef_min"], row["tef_mode"], row["tef_max"], n_sim, rng)
    vuln = np.clip(rng.normal(row["vuln_prob"], row["vuln_prob"] * 0.15 + 0.02, n_sim), 0.0, 1.0)
    lef = tef * vuln

    multiplier = row.get("exposure_multiplier", 1.0)
    multiplier = float(multiplier) if pd.notna(multiplier) else 1.0

    if upstream_breached is not None and multiplier != 1.0:
        # Trials where the upstream asset was breached get a boosted
        # effective frequency for THIS asset, in that same trial only.
        effective_lef = np.where(upstream_breached, lef * multiplier, lef)
    else:
        effective_lef = lef

    counts = rng.poisson(effective_lef)
    max_count = int(min(max(counts.max(), 1), max_events_cap))
    counts = np.minimum(counts, max_count)

    magnitudes = pert_sample(
        row["loss_min"], row["loss_mode"], row["loss_max"],
        size=(n_sim, max_count), rng=rng,
    )
    col_idx = np.arange(max_count)
    mask = col_idx[None, :] < counts[:, None]
    annual_loss = np.where(mask, magnitudes, 0.0).sum(axis=1)
    breached = counts > 0
    return annual_loss, effective_lef, breached


def _resolve_dependency_order(register_df: pd.DataFrame) -> list:
    """
    Returns asset names ordered so each asset's upstream dependency (if any)
    is simulated before it. Cycles are broken defensively — a dependency
    edge that would create a loop is simply ignored, never raised.
    """
    names = register_df["asset_name"].tolist()
    name_set = set(names)
    depends_on = dict(zip(
        register_df["asset_name"],
        register_df.get("depends_on", pd.Series([None] * len(names))),
    ))

    visited, order, visiting = set(), [], set()

    def visit(name):
        if name in visited or name in visiting:
            return
        visiting.add(name)
        upstream = depends_on.get(name)
        if isinstance(upstream, str) and upstream in name_set and upstream != name:
            visit(upstream)
        visiting.discard(name)
        visited.add(name)
        order.append(name)

    for n in names:
        visit(n)
    return order


@st.cache_data(show_spinner=False)
def run_portfolio_simulation(register_json: str, n_sim: int, seed: int):
    register = pd.read_json(io.StringIO(register_json), orient="records")
    if "depends_on" not in register.columns:
        register["depends_on"] = None
    if "exposure_multiplier" not in register.columns:
        register["exposure_multiplier"] = 1.0

    rng = np.random.default_rng(seed)
    asset_order = _resolve_dependency_order(register)
    register_indexed = register.set_index("asset_name")

    per_asset_losses = {}
    per_asset_lef = {}
    breach_flags = {}

    for name in asset_order:
        row = register_indexed.loc[name]
        upstream_name = row.get("depends_on")
        upstream_breached = breach_flags.get(upstream_name) if isinstance(upstream_name, str) else None

        losses, effective_lef, breached = simulate_asset(row, n_sim, rng, upstream_breached=upstream_breached)
        per_asset_losses[name] = losses
        per_asset_lef[name] = float(np.mean(effective_lef))
        breach_flags[name] = breached

    # Preserve original register row order for consistent downstream merges
    ordered_names = [n for n in register["asset_name"].tolist() if n in per_asset_losses]
    loss_matrix = np.column_stack([per_asset_losses[n] for n in ordered_names])
    portfolio_loss = loss_matrix.sum(axis=1)

    summary_rows = []
    for name in ordered_names:
        arr = per_asset_losses[name]
        summary_rows.append({
            "asset_name": name,
            "eal": float(np.mean(arr)),
            "var_90": float(np.percentile(arr, 90)),
            "var_95": float(np.percentile(arr, 95)),
            "var_99": float(np.percentile(arr, 99)),
            "std_dev": float(np.std(arr)),
            "mean_lef": per_asset_lef[name],
        })
    summary_df = pd.DataFrame(summary_rows)
    total_eal = summary_df["eal"].sum()
    summary_df["contribution_pct"] = np.where(
        total_eal > 0, 100 * summary_df["eal"] / total_eal, 0
    )

    portfolio_summary = {
        "eal": float(np.mean(portfolio_loss)),
        "var_90": float(np.percentile(portfolio_loss, 90)),
        "var_95": float(np.percentile(portfolio_loss, 95)),
        "var_99": float(np.percentile(portfolio_loss, 99)),
        "std_dev": float(np.std(portfolio_loss)),
        "min": float(portfolio_loss.min()),
        "max": float(portfolio_loss.max()),
    }
    return summary_df, portfolio_summary, portfolio_loss


def sensitivity_analysis(register_df, n_sim, seed, shock=0.20):
    n_sim_fast = min(n_sim, 4000)

    def portfolio_eal_with(modifier_col, factor):
        rng = np.random.default_rng(seed)
        total = np.zeros(n_sim_fast)
        asset_order = _resolve_dependency_order(register_df)
        indexed = register_df.set_index("asset_name")
        breach_flags = {}
        for name in asset_order:
            r = indexed.loc[name].copy()
            if modifier_col == "tef":
                r["tef_min"] *= factor
                r["tef_mode"] *= factor
                r["tef_max"] *= factor
            elif modifier_col == "vuln":
                r["vuln_prob"] = np.clip(r["vuln_prob"] * factor, 0, 1)
            elif modifier_col == "loss":
                r["loss_min"] *= factor
                r["loss_mode"] *= factor
                r["loss_max"] *= factor
            upstream_name = r.get("depends_on")
            upstream_breached = breach_flags.get(upstream_name) if isinstance(upstream_name, str) else None
            losses, _, breached = simulate_asset(r, n_sim_fast, rng, upstream_breached=upstream_breached)
            breach_flags[name] = breached
            total += losses
        return float(np.mean(total))

    base = portfolio_eal_with("none", 1.0)
    drivers = []
    for col, label in [("tef", "Threat Event Frequency"), ("vuln", "Vulnerability"), ("loss", "Loss Magnitude")]:
        low = portfolio_eal_with(col, 1 - shock)
        high = portfolio_eal_with(col, 1 + shock)
        drivers.append({"driver": label, "low": low, "base": base, "high": high, "swing": high - low})
    return pd.DataFrame(drivers).sort_values("swing", ascending=True)


# ------------------------------------------------------------------------------
# 3b. Investment optimizer — Δ-EAL per control + knapsack budget solver
# ------------------------------------------------------------------------------
CONTROL_TEMPLATE_COLUMNS = [
    "control_id",
    "asset_name",
    "control_name",
    "cost",
    "vuln_prob_reduction_pct",
    "tef_reduction_pct",
    "loss_reduction_pct",
]


def _simulate_portfolio_eal(register_df, n_sim, seed):
    """Dependency-aware portfolio EAL for an arbitrary (possibly modified)
    register — the same chaining logic as run_portfolio_simulation, but
    returning just the scalar mean EAL, fast enough to call once per
    candidate control."""
    rng = np.random.default_rng(seed)
    asset_order = _resolve_dependency_order(register_df)
    indexed = register_df.set_index("asset_name")
    breach_flags = {}
    total = np.zeros(n_sim)
    for name in asset_order:
        r = indexed.loc[name]
        upstream_name = r.get("depends_on")
        upstream_breached = breach_flags.get(upstream_name) if isinstance(upstream_name, str) else None
        losses, _, breached = simulate_asset(r, n_sim, rng, upstream_breached=upstream_breached)
        breach_flags[name] = breached
        total += losses
    return float(np.mean(total))


def apply_control_to_register(register_df, control):
    """Returns a copy of register_df with the named control's FAIR-field
    reductions applied to its target asset only. Unknown asset -> no-op."""
    modified = register_df.copy()
    # FAIR numeric columns can load as int64 (e.g. whole-dollar loss values).
    # Multiplying by a fractional discount then produces a non-integer float,
    # which pandas refuses to write back into an int64 cell — force float64
    # up front so every control's math always has somewhere valid to land.
    fair_numeric_cols = [
        "tef_min", "tef_mode", "tef_max",
        "vuln_prob",
        "loss_min", "loss_mode", "loss_max",
    ]
    for col in fair_numeric_cols:
        if col in modified.columns:
            modified[col] = modified[col].astype("float64")

    mask = modified["asset_name"] == control["asset_name"]
    if not mask.any():
        return modified
    idx = modified.index[mask][0]

    vuln_cut = float(np.clip(control.get("vuln_prob_reduction_pct", 0) or 0, 0, 0.9))
    tef_cut = float(np.clip(control.get("tef_reduction_pct", 0) or 0, 0, 0.9))
    loss_cut = float(np.clip(control.get("loss_reduction_pct", 0) or 0, 0, 0.9))

    if vuln_cut:
        modified.at[idx, "vuln_prob"] = float(np.clip(modified.at[idx, "vuln_prob"] * (1 - vuln_cut), 0, 1))
    if tef_cut:
        for col in ["tef_min", "tef_mode", "tef_max"]:
            modified.at[idx, col] = modified.at[idx, col] * (1 - tef_cut)
    if loss_cut:
        for col in ["loss_min", "loss_mode", "loss_max"]:
            modified.at[idx, col] = modified.at[idx, col] * (1 - loss_cut)
    return modified


def compute_delta_eal(register_df, control, baseline_eal, n_sim, seed):
    """Δ-EAL and $-ROI for one candidate control, holding everything else
    fixed. Reuses the dependency-aware simulator so a control on an
    upstream asset also credits the downstream exposure it prevents."""
    n_sim_fast = min(int(n_sim), 5000)
    modified = apply_control_to_register(register_df, control)
    controlled_eal = _simulate_portfolio_eal(modified, n_sim_fast, seed)
    delta_eal = max(0.0, baseline_eal - controlled_eal)
    cost = float(control.get("cost", 0) or 0)
    roi = (delta_eal / cost) if cost > 0 else 0.0
    return delta_eal, roi


def apply_controls_to_register(register_df, controls_list):
    """Applies every control in controls_list to register_df, one after
    another — reductions on the same asset compound multiplicatively."""
    modified = register_df.copy()
    for control in controls_list:
        modified = apply_control_to_register(modified, control)
    return modified


def knapsack_optimize(candidates, budget, bucket=1000):
    """0/1 knapsack (exact DP, cost discretized into $`bucket` units so the
    table stays small even for large dollar budgets). Returns
    (selected_within_budget, full_roi_ranked_list)."""
    roi_ranked = sorted(
        candidates,
        key=lambda c: -(c["delta_eal"] / c["cost"] if c["cost"] > 0 else 0),
    )
    if not candidates or budget <= 0:
        return [], roi_ranked

    n = len(candidates)
    budget_units = max(0, int(budget // bucket))
    costs_units = [max(0, int(c["cost"] // bucket)) for c in candidates]
    values = [c["delta_eal"] for c in candidates]

    dp = [0.0] * (budget_units + 1)
    keep = [[False] * (budget_units + 1) for _ in range(n)]
    for i in range(n):
        for b in range(budget_units, costs_units[i] - 1, -1):
            candidate_value = dp[b - costs_units[i]] + values[i]
            if candidate_value > dp[b]:
                dp[b] = candidate_value
                keep[i][b] = True

    selected = []
    b = budget_units
    for i in range(n - 1, -1, -1):
        if keep[i][b]:
            selected.append(candidates[i])
            b -= costs_units[i]
    selected.reverse()
    return selected, roi_ranked


# ------------------------------------------------------------------------------
# 4. NVD + EPSS API Helpers & Asset Register Ingestion
# ------------------------------------------------------------------------------
import io
import re
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

TEMPLATE_COLUMNS = [
    "asset_name",
    "threat_event",
    "vulnerability",
    "risk_level",
    "tef_min",
    "tef_mode",
    "tef_max",
    "vuln_prob",
    "loss_min",
    "loss_mode",
    "loss_max",
]

INDUSTRY_AVG_BREACH_COST = {
    "healthcare": 9_770_000,
    "financial": 6_080_000,
    "technology": 5_300_000,
    "energy": 5_290_000,
    "industrial": 5_560_000,
    "retail": 3_910_000,
    "public_sector": 2_600_000,
    "default": 4_880_000,
}

def fetch_nvd_details(vuln_text):
    cve_match = re.search(r"CVE-\d{4}-\d{4,7}", str(vuln_text), re.IGNORECASE)
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {}

    if cve_match:
        cve_id = cve_match.group(0).upper()
        params["cveId"] = cve_id
    else:
        cve_id = "N/A"
        params["keywordSearch"] = str(vuln_text)[:50]

    headers = {"User-Agent": "CyberRiskDashboard/2.0"}

    try:
        response = requests.get(
            base_url, params=params, headers=headers, timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            vulnerabilities = data.get("vulnerabilities", [])

            if vulnerabilities:
                cve_item = vulnerabilities[0].get("cve", {})
                if cve_id == "N/A":
                    cve_id = cve_item.get("id", "N/A")

                metrics = cve_item.get("metrics", {})
                metric_entry = None
                if "cvssMetricV31" in metrics:
                    metric_entry = metrics["cvssMetricV31"][0]
                elif "cvssMetricV30" in metrics:
                    metric_entry = metrics["cvssMetricV30"][0]
                elif "cvssMetricV2" in metrics:
                    metric_entry = metrics["cvssMetricV2"][0]

                if metric_entry:
                    cvss_data = metric_entry.get("cvssData", {})
                    base_score = float(cvss_data.get("baseScore", 5.0))
                    impact_score = float(
                        metric_entry.get("impactScore", base_score * 0.6)
                    )
                    exploitability_score = float(
                        metric_entry.get(
                            "exploitabilityScore", base_score * 0.4
                        )
                    )
                    return (
                        cve_id,
                        base_score,
                        impact_score,
                        exploitability_score,
                    )
    except Exception:
        pass

    return cve_id, 5.0, 5.0, 5.0


def fetch_epss_score(cve_id):
    if not cve_id or cve_id == "N/A":
        return None
    try:
        r = requests.get(
            "https://api.first.org/data/v1/epss",
            params={"cve": cve_id},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                return float(data[0]["epss"])
    except Exception:
        pass
    return None


def calculate_dynamic_loss(
    impact_score, exploitability_score, epss_score, sector="default"
):
    """Estimate the loss of one incident affecting one asset, in USD.

    Industry breach-cost figures describe an entire organization-wide breach,
    so they are deliberately not used as an input here.
    """
    asset_base_loss = {
        "healthcare": 30_000,
        "financial": 25_000,
        "technology": 20_000,
        "energy": 30_000,
        "industrial": 22_000,
        "retail": 18_000,
        "public_sector": 15_000,
        "default": 15_000,
    }
    base_loss = asset_base_loss.get(sector, asset_base_loss["default"])
    impact = float(np.clip(impact_score or 0.0, 0.0, 10.0))
    exploitability = float(np.clip(exploitability_score or 0.0, 0.0, 10.0))
    likelihood_factor = (
        float(np.clip(epss_score, 0.01, 0.95))
        if epss_score is not None
        else float(np.clip(exploitability / 10.0, 0.05, 0.80))
    )

    severity_multiplier = 0.25 + 1.75 * (impact / 10.0)
    exploit_multiplier = 0.90 + 0.20 * (exploitability / 10.0)
    loss_mode = round(base_loss * severity_multiplier * exploit_multiplier, -2)
    loss_min = round(max(1_000, loss_mode * (0.25 + 0.15 * (1 - likelihood_factor))), -2)
    loss_max = round(loss_mode * (1.75 + 1.25 * likelihood_factor), -2)

    return float(loss_min), float(loss_mode), float(loss_max), likelihood_factor


def infer_sector(asset_name):
    name_lower = str(asset_name).lower()
    if any(k in name_lower for k in ["patient", "ehr", "health", "clinical"]):
        return "healthcare"
    if any(k in name_lower for k in ["payment", "bank", "ledger", "transaction"]):
        return "financial"
    if any(k in name_lower for k in ["technology", "cloud", "application", "api"]):
        return "technology"
    if any(k in name_lower for k in ["scada", "ics", "plc", "grid"]):
        return "energy"
    if any(k in name_lower for k in ["factory", "plant", "manufactur"]):
        return "industrial"
    if any(k in name_lower for k in ["pos", "checkout", "storefront", "retail"]):
        return "retail"
    if any(k in name_lower for k in ["gov", "municipal", "public"]):
        return "public_sector"
    return "default"


def enrich_with_nvd_dynamic_loss(df_raw):
    """Fetches live NVD + EPSS data and computes FAIR values with live progress tracking."""
    enriched_rows = []
    total_rows = len(df_raw)

    if total_rows == 0:
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)

    # UI Progress Elements
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    for idx, (row_idx, row) in enumerate(df_raw.iterrows(), start=1):
        asset_name = str(row.get("asset_name", "Unnamed Asset")).strip()
        vuln_input = str(row.get("vulnerability", "Generic Vulnerability")).strip()
        threat_evt = str(row.get("threat_event", "Unauthorized Network Access")).strip()

        # Update progress status: "Fetching... 1/25 completed (Asset Name)"
        status_text.markdown(
            f"**Fetching... {idx}/{total_rows} completed**\n\n`{asset_name}`"
        )
        progress_bar.progress(idx / total_rows)

        # Live NVD & EPSS API fetches
        cve_id, cvss_score, impact_score, exploitability_score = fetch_nvd_details(vuln_input)
        epss_score = fetch_epss_score(cve_id)

        sector = infer_sector(asset_name)
        loss_min, loss_mode, loss_max, likelihood_factor = calculate_dynamic_loss(
            impact_score, exploitability_score, epss_score, sector
        )

        vuln_prob = float(np.clip(likelihood_factor, 0.05, 0.95))
        tef_mode = round(max(0.2, likelihood_factor * 1.5), 2)
        tef_min = round(max(0.05, tef_mode * 0.3), 2)
        tef_max = round(tef_mode * 2.0, 2)

        risk_level = (
    "Critical" if cvss_score >= 9.0
    else "High" if cvss_score >= 7.0
    else "Medium" if cvss_score >= 4.0
    else "Low" if cvss_score > 0
    else "Negligible"
)

        enriched_rows.append({
            "asset_name": asset_name,
            "threat_event": threat_evt,
            "vulnerability": cve_id if cve_id != "N/A" else vuln_input,
            "risk_level": risk_level,
            "tef_min": tef_min,
            "tef_mode": tef_mode,
            "tef_max": tef_max,
            "vuln_prob": vuln_prob,
            "loss_min": loss_min,
            "loss_mode": loss_mode,
            "loss_max": loss_max,
        })
        time.sleep(0.6)  # NVD API rate limit compliance

    # Clean up status widgets when complete
    status_text.success(f"✅ Completed enrichment for all {total_rows} assets!")
    progress_bar.empty()

    return pd.DataFrame(enriched_rows, columns=TEMPLATE_COLUMNS)
def _sample_register():
    return pd.DataFrame([
        {
            "asset_name": "Core Database Server",
            "threat_event": "Ransomware Attack",
            "vulnerability": "Unpatched System",
            "risk_level": "High",
            "tef_min": 0.5,
            "tef_mode": 1.2,
            "tef_max": 3.0,
            "vuln_prob": 0.55,
            "loss_min": 400000,
            "loss_mode": 1100000,
            "loss_max": 3200000,
        },
        {
            "asset_name": "Payment Gateway API",
            "threat_event": "DDoS / Service Outage",
            "vulnerability": "Single Point of Failure",
            "risk_level": "Medium",
            "tef_min": 1.0,
            "tef_mode": 2.5,
            "tef_max": 6.0,
            "vuln_prob": 0.35,
            "loss_min": 10_000,
            "loss_mode": 25_000,
            "loss_max": 65_000,
        },
        {
            "asset_name": "Employee Workstations",
            "threat_event": "Phishing / Credential Theft",
            "vulnerability": "Lack of MFA",
            "risk_level": "Medium",
            "tef_min": 3.0,
            "tef_mode": 8.0,
            "tef_max": 20.0,
            "vuln_prob": 0.40,
            "loss_min": 20000,
            "loss_mode": 90000,
            "loss_max": 400000,
        },
        {
            "asset_name": "Customer Portal",
            "threat_event": "SQL Injection",
            "vulnerability": "Input Sanitization Defect",
            "risk_level": "High",
            "tef_min": 0.3,
            "tef_mode": 1.0,
            "tef_max": 2.5,
            "vuln_prob": 0.50,
            "loss_min": 250000,
            "loss_mode": 750000,
            "loss_max": 2000000,
        },
        {
            "asset_name": "Legacy File Server",
            "threat_event": "Data Leakage",
            "vulnerability": "Weak Access Controls",
            "risk_level": "Low",
            "tef_min": 0.2,
            "tef_mode": 0.6,
            "tef_max": 1.5,
            "vuln_prob": 0.30,
            "loss_min": 15000,
            "loss_mode": 60000,
            "loss_max": 200000,
        },
    ])


def _asset_identity(asset_name) -> str:
    """A comparison key for an asset name, tolerant of case and punctuation.

    For example, ``Customer Portal``, ``customer-portal``, and
    ``CUSTOMER_PORTAL`` identify the same asset for register purposes.
    """
    return re.sub(r"[^a-z0-9]+", "", str(asset_name).casefold())


def _coerce_register(df: pd.DataFrame):
    problems = []

    if not isinstance(df, pd.DataFrame) or df.empty:
        problems.append("DataFrame is empty or invalid structure.")
        return pd.DataFrame(), problems

    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in TEMPLATE_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"Missing column(s): {', '.join(missing)}")
        return df.iloc[0:0], problems

    numeric_cols = [
        "tef_min",
        "tef_mode",
        "tef_max",
        "vuln_prob",
        "loss_min",
        "loss_mode",
        "loss_max",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad_rows = df[numeric_cols].isna().any(axis=1) | df["asset_name"].isna()
    if bad_rows.any():
        problems.append(
            f"{int(bad_rows.sum())} row(s) had missing values and were dropped."
        )
        df = df[~bad_rows]

    # A register represents unique assets.  Preserve the first occurrence so
    # duplicate rows from a CSV, the editor, or an AI response never create a
    # second Monte Carlo exposure for the same real-world asset.
    if not df.empty:
        asset_keys = df["asset_name"].map(_asset_identity)
        duplicate_rows = asset_keys.duplicated(keep="first")
        if duplicate_rows.any():
            df = df.loc[~duplicate_rows].copy()

    # --- Dependency layer: optional columns, default-filled so any register
    # (old saves, fresh uploads, Quick Add) works without them. ---
    if "depends_on" not in df.columns:
        df["depends_on"] = None
    else:
        df["depends_on"] = df["depends_on"].apply(
            lambda v: v.strip() if isinstance(v, str) and v.strip() else None
        )
    if "exposure_multiplier" not in df.columns:
        df["exposure_multiplier"] = 1.0
    else:
        df["exposure_multiplier"] = pd.to_numeric(
            df["exposure_multiplier"], errors="coerce"
        ).fillna(1.0).clip(1.0, 10.0)

    if not df.empty:
        df["vuln_prob"] = df["vuln_prob"].clip(0, 1)
        df["risk_level"] = (
            df["risk_level"]
            .fillna("Medium")
            .astype(str)
            .str.strip()
            .str.title()
        )
        df.loc[
            ~df["risk_level"].isin(["Critical", "High", "Medium", "Low"]),
            "risk_level",
        ] = "Medium"

        # Invalidate any depends_on pointing at an asset that no longer
        # exists (e.g. it was just deleted) or at itself.
        valid_names = set(df["asset_name"])
        df["depends_on"] = df.apply(
            lambda r: r["depends_on"]
            if (r["depends_on"] in valid_names and r["depends_on"] != r["asset_name"])
            else None,
            axis=1,
        )

    return df.reset_index(drop=True), problems


def _infer_dependencies_with_llm(new_asset_names, existing_asset_names):
    """
    Infers plausible architectural dependencies for newly added assets
    against assets already in the register. Never forces a match — defaults
    to no dependency when nothing plausible fits.
    """
    severity_multiplier = {
        "none": 1.0, "minimal": 1.2, "moderate": 2.5, "severe": 5.0, "critical": 8.0,
    }
    fallback = [
        {"asset_name": name, "depends_on": None, "severity": "none",
         "exposure_multiplier": 1.0, "reason": ""}
        for name in new_asset_names
    ]

    if groq_client is None or not existing_asset_names:
        return fallback

    prompt = f"""You are analyzing IT architecture for a cybersecurity risk register.

New assets being added: {json.dumps(new_asset_names)}
Assets already in the register: {json.dumps(existing_asset_names)}

For EACH new asset, decide whether it plausibly sits "behind" or depends on
one of the EXISTING assets architecturally (e.g. a database behind an API
gateway, an internal app behind a VPN, a workstation behind a firewall).
Only propose a dependency when it is a common, plausible architectural
relationship based on typical naming/roles — do not force a match if none
is plausible.

Return ONLY a JSON array, one object per new asset, in the same order, using exactly:
[{{"asset_name":"...", "depends_on": "<exact existing asset name, or null>", "severity": "<none|minimal|moderate|severe|critical>", "reason": "<one short plain-English sentence>"}}]
"""
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
        answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
        inferred = json.loads(answer)
        if not isinstance(inferred, list) or len(inferred) != len(new_asset_names):
            return fallback

        results = []
        for name, item in zip(new_asset_names, inferred):
            depends_on = item.get("depends_on")
            severity = str(item.get("severity", "none")).strip().lower()
            if depends_on not in existing_asset_names or depends_on == name:
                depends_on, severity = None, "none"
            if severity not in severity_multiplier:
                severity = "none" if depends_on is None else "moderate"
            results.append({
                "asset_name": name,
                "depends_on": depends_on,
                "severity": severity,
                "exposure_multiplier": severity_multiplier[severity],
                "reason": str(item.get("reason", "")).strip(),
            })
        return results
    except Exception:
        return fallback


def _extract_ai_csv(answer: str):
    """Extract a valid asset CSV from an assistant response."""
    candidates = re.findall(
        r"```(?:csv)?\s*(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE
    )
    if not candidates:
        lines = answer.splitlines()
        header_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip().lower().startswith("asset_name,")
            ),
            None,
        )
        if header_index is not None:
            candidates = ["\n".join(lines[header_index:])]

    required_columns = {"asset_name", "threat_event", "vulnerability"}
    for candidate in candidates:
        try:
            parsed_df = pd.read_csv(io.StringIO(candidate.strip()))
        except (pd.errors.ParserError, UnicodeDecodeError):
            continue
        parsed_df.columns = [str(column).strip() for column in parsed_df.columns]
        if required_columns.issubset(parsed_df.columns):
            return parsed_df
    return None


def _infer_asset_context_with_llm(asset_df):
    """Infer threat context from asset names without allowing fabricated CVEs."""
    if groq_client is None:
        raise ValueError(f"LLM inference unavailable: {groq_error}")

    asset_names = [str(name).strip() for name in asset_df["asset_name"]]
    prompt = f"""You are a cybersecurity risk analyst. Infer context for these assets:
{json.dumps(asset_names)}

Return ONLY a JSON array with one object per input asset, in the same order, using exactly:
[{{"asset_name":"...","threat_event":"...","vulnerability":"..."}}]

Rules:
- threat_event must be a realistic attack or loss event.
- vulnerability must be a generic vulnerability category or description, such as "Weak access controls" or "Unpatched container image".
- Never output a CVE number, CVE identifier, version-specific claim, or invented vulnerability.
- Keep each value concise and grounded in the asset type.
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content.strip()
    answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
    inferred = json.loads(answer)
    if not isinstance(inferred, list) or len(inferred) != len(asset_names):
        raise ValueError("LLM returned an invalid number of asset inferences.")

    inferred_df = pd.DataFrame(inferred)
    required = {"asset_name", "threat_event", "vulnerability"}
    if not required.issubset(inferred_df.columns):
        raise ValueError("LLM response is missing threat_event or vulnerability fields.")
    if inferred_df["vulnerability"].astype(str).str.contains(r"\bCVE-\d{4}-\d{4,}\b", case=False, regex=True).any():
        raise ValueError("LLM returned a CVE identifier; no fabricated CVEs are accepted.")
    inferred_df["asset_name"] = asset_names
    return inferred_df[["asset_name", "threat_event", "vulnerability"]]


REGISTER_STORE_PATH = Path(__file__).resolve().parent / "data" / "dashboard_asset_register.json"


def _load_saved_register():
    if not REGISTER_STORE_PATH.exists():
        return _sample_register()
    try:
        saved_df = pd.read_json(REGISTER_STORE_PATH, orient="records")
        register_df, _ = _coerce_register(saved_df)
        return register_df if not register_df.empty else _sample_register()
    except (OSError, ValueError, TypeError):
        return _sample_register()


def _save_register(register_df):
    REGISTER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    register_df.to_json(REGISTER_STORE_PATH, orient="records", indent=2)


CONTROL_LIBRARY_PATH = Path(__file__).resolve().parent / "data" / "control_library.json"


def _normalize_control_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes control rows loaded from disk or AI output. Older JSON files
    can be missing required columns (notably ``cost``), which caused the
    optimizer table to crash on first render when selecting an asset."""
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=CONTROL_TEMPLATE_COLUMNS)

    normalized = df.copy()
    for col in CONTROL_TEMPLATE_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = "" if col in ("control_id", "asset_name", "control_name") else 0

    normalized["asset_name"] = normalized["asset_name"].fillna("").astype(str)
    normalized["control_name"] = normalized["control_name"].fillna("Unnamed Control").astype(str)
    normalized["cost"] = pd.to_numeric(normalized["cost"], errors="coerce").fillna(0).clip(lower=0)
    for col in ["vuln_prob_reduction_pct", "tef_reduction_pct", "loss_reduction_pct"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0).clip(0, 0.9)
    normalized["control_id"] = [
        cid if str(cid).strip() else f"ctrl_{i}"
        for i, cid in enumerate(normalized.get("control_id", pd.Series(dtype=str)))
    ]
    return normalized[CONTROL_TEMPLATE_COLUMNS].reset_index(drop=True)


def _coerce_controls(df: pd.DataFrame, valid_asset_names):
    """Ensures every control row targets a real asset and has sane numeric
    fields; drops anything that doesn't. Mirrors _coerce_register's style."""
    normalized = _normalize_control_df(df)
    if not isinstance(valid_asset_names, (list, tuple, set, pd.Index)):
        return normalized
    valid_asset_names = set(str(name) for name in valid_asset_names)
    if valid_asset_names:
        normalized = normalized[normalized["asset_name"].isin(valid_asset_names)]
    return normalized[CONTROL_TEMPLATE_COLUMNS].reset_index(drop=True)


def _load_control_library():
    if not CONTROL_LIBRARY_PATH.exists():
        return pd.DataFrame(columns=CONTROL_TEMPLATE_COLUMNS)
    try:
        loaded_df = pd.read_json(CONTROL_LIBRARY_PATH, orient="records")
        return _normalize_control_df(loaded_df)
    except (OSError, ValueError, TypeError):
        return pd.DataFrame(columns=CONTROL_TEMPLATE_COLUMNS)


def _save_control_library(controls_df):
    CONTROL_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    controls_df.to_json(CONTROL_LIBRARY_PATH, orient="records", indent=2)


def _infer_controls_with_llm(asset_rows):
    """Proposes 2-3 plausible mitigating controls per asset via Groq. Each
    control names which FAIR field(s) it discounts and by how much — never
    fabricates a specific product, price list, or compliance claim."""
    fallback = pd.DataFrame(columns=CONTROL_TEMPLATE_COLUMNS)
    if groq_client is None or not asset_rows:
        return fallback

    prompt = f"""You are a cybersecurity control-planning analyst.

For EACH asset below, propose 2-3 plausible, distinct mitigating controls
(e.g. MFA enforcement, network segmentation, offsite backups, WAF rules,
employee security training, patch management automation).

Assets (asset_name, threat_event, vulnerability, current tef_mode,
current vuln_prob, current loss_mode):
{json.dumps(asset_rows)}

For each control estimate:
- cost: a realistic one-time/annual USD implementation cost (number only)
- vuln_prob_reduction_pct: fraction (0-0.9) this control cuts the asset's vulnerability probability, or 0 if not applicable
- tef_reduction_pct: fraction (0-0.9) this control cuts threat event frequency, or 0 if not applicable
- loss_reduction_pct: fraction (0-0.9) this control cuts loss magnitude if a breach still occurs, or 0 if not applicable

A control should usually affect only ONE of the three fields meaningfully — leave the others at 0.

Return ONLY a JSON array using exactly:
[{{"asset_name":"...", "control_name":"...", "cost": 0, "vuln_prob_reduction_pct": 0, "tef_reduction_pct": 0, "loss_reduction_pct": 0}}]
"""
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
        answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
        inferred = json.loads(answer)
        if not isinstance(inferred, list):
            return fallback
        inferred_df = pd.DataFrame(inferred)
        required = {"asset_name", "control_name"}
        if not required.issubset(inferred_df.columns):
            return fallback
        inferred_df["control_id"] = [f"ctrl_llm_{i}" for i in range(len(inferred_df))]
        return inferred_df
    except Exception:
        return fallback


@st.dialog("Delete asset")
def confirm_delete_asset(asset_name):
    streamlit_api.warning(f"Delete **{asset_name}** from the asset register?")
    streamlit_api.caption("This removes the asset from the saved register and the next Monte Carlo run.")
    if streamlit_api.button("Delete asset", type="primary", key=f"confirm_delete_{asset_name}"):
        remaining_df = streamlit_api.session_state.processed_df[
            streamlit_api.session_state.processed_df["asset_name"] != asset_name
        ].reset_index(drop=True)
        streamlit_api.session_state.processed_df = remaining_df
        streamlit_api.session_state.register_source = "saved"
        _save_register(remaining_df)
        streamlit_api.rerun()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Streamlit UI Controls
# ------------------------------------------------------------------------------
class _SectionContainer:
    """A tab child that preserves nested Streamlit layout contexts.

    Without this wrapper, a widget invoked inside ``with st.expander(...)``
    would be sent straight back to the parent tab by _SectionUI.  That made
    Quick Add's controls appear below the collapsed dropdown.
    """

    def __init__(self, section_ui, container):
        self._section_ui = section_ui
        self._container = container

    def __enter__(self):
        entered = self._container.__enter__()
        self._section_ui._push_container(
            entered if entered is not None else self._container
        )
        return self

    def __exit__(self, *args):
        try:
            return self._container.__exit__(*args)
        finally:
            self._section_ui._pop_container()

    def __getattr__(self, name):
        member = getattr(self._container, name)
        if not callable(member):
            return member

        def routed_call(*args, **kwargs):
            return self._section_ui._wrap_result(member(*args, **kwargs))

        return routed_call


class _SectionUI:
    """Routes visual calls into one workflow tab while retaining Streamlit's
    state, cache, configuration, and column-config APIs.

    The dashboard's risk engine remains shared, so changes made in Asset Upload
    are immediately available to EAL and Optimization without duplicating state.
    """

    def __init__(self, container, streamlit_api):
        self._container = container
        self._streamlit = streamlit_api
        self._container_stack = []

    @property
    def sidebar(self):
        # Legacy sidebar controls are now shown in the active workflow tab.
        return self

    @property
    def _active_container(self):
        return self._container_stack[-1] if self._container_stack else self._container

    def _push_container(self, container):
        self._container_stack.append(container)

    def _pop_container(self):
        if self._container_stack:
            self._container_stack.pop()

    def _wrap_result(self, result):
        if isinstance(result, (list, tuple)):
            wrapped = [self._wrap_result(item) for item in result]
            return type(result)(wrapped)
        if hasattr(result, "__enter__") and hasattr(result, "__exit__") and hasattr(result, "container"):
            return _SectionContainer(self, result)
        return result

    def __getattr__(self, name):
        if name in {
            "session_state",
            "cache_data",
            "column_config",
            "rerun",
            "stop",
            "form_submit_button",
        }:
            return getattr(self._streamlit, name)
        if hasattr(self._active_container, name):
            member = getattr(self._active_container, name)
            if callable(member):
                def routed_call(*args, **kwargs):
                    return self._wrap_result(member(*args, **kwargs))
                return routed_call
            return member
        return getattr(self._streamlit, name)


streamlit_api = st
st = _SectionUI(asset_tab, streamlit_api)

st.sidebar.header("📁 Asset Intake & Simulation")

uploaded = st.sidebar.file_uploader(
    "Upload your assets (CSV or XLSX)",
    type=["csv", "xlsx", "xls"],
    help=(
        "Must include at least 'asset_name', 'threat_event', 'vulnerability'"
        " columns."
    ),
)

if "processed_df" not in st.session_state:
    st.session_state.processed_df = _load_saved_register()
if "last_uploaded_id" not in st.session_state:
    st.session_state.last_uploaded_id = None
if "register_source" not in st.session_state:
    st.session_state.register_source = "saved"

with st.sidebar.expander("➕ Quick Add", expanded=True):
    st.caption("Upload a CSV/XLSX/XLS containing an asset_name column. The LLM fills in the threat event and generic vulnerability description.")
    quick_upload = st.file_uploader(
        "Quick Add asset file",
        type=["csv", "xlsx", "xls"],
        key="quick_add_upload",
        help="Required: asset_name. Optional: asset_id, tech_keyword, criticality, mfa_enabled, patched, firewall, network_segmented.",
    )
    quick_add_submitted = st.button("Add file assets", type="primary", disabled=quick_upload is None)

    if quick_add_submitted and quick_upload is not None:
        try:
            if quick_upload.name.lower().endswith(".csv"):
                try:
                    quick_raw_df = pd.read_csv(quick_upload, sep=None, engine="python")
                except Exception:
                    quick_upload.seek(0)
                    quick_raw_df = pd.read_csv(quick_upload)
            else:
                quick_raw_df = pd.read_excel(quick_upload)

            quick_raw_df.columns = [
                str(column).strip().lower().replace(" ", "_")
                for column in quick_raw_df.columns
            ]
            quick_raw_df = quick_raw_df.rename(columns={
                "asset": "asset_name",
                "asset_title": "asset_name",
            })
            if "asset_name" not in quick_raw_df.columns:
                raise ValueError("Quick Add files must include an asset_name column.")
            quick_raw_df = quick_raw_df[quick_raw_df["asset_name"].notna()].copy()
            if quick_raw_df.empty:
                raise ValueError("Quick Add file contains no asset names.")

            with st.spinner("Inferring threat context and checking vulnerability data..."):
                inferred_context = _infer_asset_context_with_llm(quick_raw_df)
                quick_enriched_df = enrich_with_nvd_dynamic_loss(inferred_context)
                quick_register_df, quick_problems = _coerce_register(quick_enriched_df)

            # Keep the duplicate comparison safe even if enrichment returned
            # no valid rows; the existing error message below handles that.
            if quick_problems or quick_register_df.empty:
                quick_register_df = pd.DataFrame({"asset_name": []})

            # Quick Add merges into the saved register.  Remove any incoming
            # row whose normalized name is already present, rather than
            # replacing the existing asset or adding a duplicate row.
            current_register = st.session_state.processed_df
            existing_asset_keys = set(
                current_register["asset_name"].map(_asset_identity)
            )
            incoming_asset_keys = quick_register_df["asset_name"].map(_asset_identity)
            already_registered = incoming_asset_keys.isin(existing_asset_keys)
            skipped_asset_names = quick_register_df.loc[
                already_registered, "asset_name"
            ].tolist()
            quick_register_df = quick_register_df.loc[
                ~already_registered
            ].reset_index(drop=True)

            if quick_problems or quick_register_df.empty:
                if quick_problems:
                    st.error(" / ".join(quick_problems))
                else:
                    skipped_label = ", ".join(map(str, skipped_asset_names))
                    st.info(
                        "No new assets were added. The uploaded asset(s) are "
                        f"already in the register: {skipped_label}"
                    )
            else:
                if skipped_asset_names:
                    st.info(
                        "Skipped already-registered asset(s): "
                        + ", ".join(map(str, skipped_asset_names))
                    )
                existing_names = st.session_state.processed_df["asset_name"].tolist()
                new_names = quick_register_df["asset_name"].tolist()
                with st.spinner("Checking for architectural dependencies..."):
                    dependency_results = _infer_dependencies_with_llm(new_names, existing_names)
                dep_by_name = {d["asset_name"]: d for d in dependency_results}
                quick_register_df["depends_on"] = quick_register_df["asset_name"].map(
                    lambda n: dep_by_name.get(n, {}).get("depends_on")
                )
                quick_register_df["exposure_multiplier"] = quick_register_df["asset_name"].map(
                    lambda n: dep_by_name.get(n, {}).get("exposure_multiplier", 1.0)
                )

                updated_register = pd.concat(
                    [st.session_state.processed_df, quick_register_df],
                    ignore_index=True,
                )
                st.session_state.processed_df = updated_register
                st.session_state.register_source = "saved"
                _save_register(updated_register)

                st.success(f"Added {len(quick_register_df)} asset(s) to the register.")
                for d in dependency_results:
                    if d["depends_on"]:
                        st.caption(
                            f"Detected dependency: **{d['asset_name']}** → **{d['depends_on']}** "
                            f"({d['severity']}). {d['reason']}"
                        )
                st.rerun()
        except Exception as exc:
            st.error(f"Quick Add failed: {exc}")

template_df = pd.DataFrame([{
    "asset_name": "Active Directory Primary Domain Controller",
    "threat_event": "Kerberos Golden Ticket Privilege Escalation",
    "vulnerability": "CVE-2023-4154",
}])
template_excel = io.BytesIO()
template_df.to_excel(template_excel, index=False)

st.sidebar.download_button(
    "⬇ Download Minimal Excel Template",
    template_excel.getvalue(),
    file_name="minimal_asset_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

using_sample = st.session_state.register_source == "sample"

if uploaded is not None:
    current_file_id = f"{uploaded.name}_{uploaded.size}"

    if st.session_state.last_uploaded_id != current_file_id:
        try:
            if uploaded.name.lower().endswith(".csv"):
                try:
                    raw_df = pd.read_csv(uploaded, sep=None, engine="python")
                except Exception:
                    uploaded.seek(0)
                    raw_df = pd.read_csv(uploaded)
            else:
                raw_df = pd.read_excel(uploaded)

            raw_df.columns = [
                str(c).strip().lower().replace(" ", "_") for c in raw_df.columns
            ]

            # Flexible header mapping
            aliases = {
                "vulnerability_type": "vulnerability",
                "cve": "vulnerability",
                "cve_id": "vulnerability",
                "threat": "threat_event",
                "threat_name": "threat_event",
                "asset": "asset_name",
                "asset_title": "asset_name",
            }
            raw_df = raw_df.rename(columns=aliases)

            # Auto-populate missing required columns with safe fallback values
            if "asset_name" not in raw_df.columns:
                raw_df["asset_name"] = "Unnamed Asset"
            if "threat_event" not in raw_df.columns:
                raw_df["threat_event"] = "Unauthorized Access"
            if "vulnerability" not in raw_df.columns:
                raw_df["vulnerability"] = "Generic Vulnerability"

            if len(raw_df) > 0:
                st.sidebar.info(
                    "Enriching assets dynamically with NVD + EPSS data..."
                )
                raw_df = raw_df[["asset_name", "threat_event", "vulnerability"]]
                enriched_df = enrich_with_nvd_dynamic_loss(raw_df)
                register_df, register_problems = _coerce_register(enriched_df)
            else:
                st.sidebar.error("Uploaded file has 0 data rows.")
                register_df = _sample_register()
                register_problems = []
                using_sample = True

            if register_df.empty and not using_sample:
                st.sidebar.error(
                    " / ".join(register_problems)
                    if register_problems
                    else "Uploaded file produced no valid rows."
                )
                register_df = _sample_register()
                using_sample = True

            st.session_state.processed_df = register_df
            st.session_state.last_uploaded_id = current_file_id
            st.session_state.register_source = "uploaded"
            _save_register(register_df)

        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")
            register_df = _sample_register()
            using_sample = True
    else:
        register_df = st.session_state.processed_df
else:
    st.session_state.last_uploaded_id = None
    register_df = st.session_state.processed_df.copy()

# The EAL workspace owns simulation settings, KPI cards, charts, sensitivity,
# and result exports.  The shared register above is still available here.
st = _SectionUI(eal_tab, streamlit_api)
st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Simulation Settings")
n_sim = st.sidebar.select_slider(
    "Simulations per asset",
    options=[1000, 2000, 5000, 10000, 20000, 50000],
    value=10000,
)
seed = 42

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear Cache & Re-scan NVD", key="clear_cache_rescan"):
    st.session_state.clear()
    st.rerun()

# Asset editing is intentionally isolated from quantitative analysis.
st = _SectionUI(asset_tab, streamlit_api)
# ------------------------------------------------------------------------------
# 5. Editable register — real-time simulation trigger
# ------------------------------------------------------------------------------
st.markdown('<div id="asset-upload"></div>', unsafe_allow_html=True)
st.header("📁 Asset Upload & Risk Register")
st.caption("Import, enrich, and maintain the assets included in your FAIR risk model.")
st.subheader("Asset Risk Register" + (" (sample data — upload your own via the sidebar)" if using_sample else ""))
edited_df = st.data_editor(
    register_df,
    width="stretch",
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "tef_min": st.column_config.NumberColumn("TEF min (events/yr)", format="%.2f"),
        "tef_mode": st.column_config.NumberColumn("TEF likely (events/yr)", format="%.2f"),
        "tef_max": st.column_config.NumberColumn("TEF max (events/yr)", format="%.2f"),
        "vuln_prob": st.column_config.NumberColumn("Vulnerability (0-1)", format="%.2f", min_value=0.0, max_value=1.0),
        "loss_min": st.column_config.NumberColumn(currency_label("Loss min"), format=f"{currency_symbol}%,.0f"),
        "loss_mode": st.column_config.NumberColumn(currency_label("Loss likely"), format=f"{currency_symbol}%,.0f"),
        "loss_max": st.column_config.NumberColumn(currency_label("Loss max"), format=f"{currency_symbol}%,.0f"),
        "risk_level": st.column_config.SelectboxColumn(
            "Risk Level", options=["Critical", "High", "Medium", "Low"]
        ),
        "depends_on": st.column_config.TextColumn(
            "Depends On (asset name)",
            help="Leave blank if this asset has no upstream dependency. Auto-detected for Quick Add assets — edit here to correct it.",
        ),
        "exposure_multiplier": st.column_config.NumberColumn(
            "Exposure Multiplier", format="%.1fx", min_value=1.0, max_value=10.0, step=0.5,
            help="How much riskier this asset becomes, in trials where its upstream dependency is breached.",
        ),
    },
)

valid_df, edit_problems = _coerce_register(edited_df)
if edit_problems:
    st.warning(" / ".join(edit_problems))

if valid_df.empty:
    st.error("No valid assets to simulate. Fix the register above.")
    st.stop()

st.session_state.processed_df = valid_df.copy()
_save_register(valid_df)
st.caption("Click an asset below to reveal its delete action.")
asset_selection = st.dataframe(
    valid_df[["asset_name"]].rename(columns={"asset_name": "Asset"}),
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="asset_delete_picker",
)
selected_rows = asset_selection.selection.rows
selected_row = selected_rows[0] if selected_rows else None
if isinstance(selected_row, int) and 0 <= selected_row < len(valid_df):
    delete_asset = valid_df.iloc[selected_row]["asset_name"]
    if st.button(f"🗑️ Delete {delete_asset}", key="delete_selected_asset"):
        confirm_delete_asset(delete_asset)

# Calculate and display portfolio exposure in the EAL workspace.
st = _SectionUI(eal_tab, streamlit_api)
# ------------------------------------------------------------------------------
# 6. Run Monte Carlo Simulation
# ------------------------------------------------------------------------------
# 6. Run Monte Carlo Simulation
# ------------------------------------------------------------------------------
register_json = valid_df.to_json(orient="records")

with st.spinner("Running Monte Carlo simulation..."):
    summary_df, portfolio_summary, portfolio_loss = run_portfolio_simulation(register_json, int(n_sim), int(seed))

full_df = valid_df.merge(summary_df, on="asset_name", how="left")

# --- Persistent portfolio history logging (disk-based, NOT session_state) ---
# This is what makes the trend chart below survive across CSV uploads,
# asset removals, and even closing/reopening the browser tab. Streamlit's
# session_state resets on every new session; a SQLite file on disk does not.
import sqlite3

PORTFOLIO_HISTORY_DB = Path(__file__).resolve().parent / "portfolio_trend.db"


def _init_portfolio_history_db():
    conn = sqlite3.connect(PORTFOLIO_HISTORY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            portfolio_eal REAL,
            portfolio_var_95 REAL,
            n_assets INTEGER,
            note TEXT
        )
    """)
    conn.commit()
    return conn


def _log_portfolio_snapshot(eal, var_95, n_assets, note=""):
    """Appends one row — never overwrites, never deletes. This is why
    removing an asset and re-running shows a NEW lower point on the trend
    instead of replacing/erasing the earlier, higher point."""
    conn = _init_portfolio_history_db()
    conn.execute(
        "INSERT INTO portfolio_history (timestamp, portfolio_eal, portfolio_var_95, n_assets, note) VALUES (?, ?, ?, ?, ?)",
        (pd.Timestamp.now().isoformat(), float(eal), float(var_95), int(n_assets), note),
    )
    conn.commit()
    conn.close()


def _load_portfolio_history():
    conn = _init_portfolio_history_db()
    df = pd.read_sql("SELECT * FROM portfolio_history ORDER BY timestamp", conn)
    conn.close()
    return df


def _should_log_snapshot(eal, n_assets, history_df, eal_threshold_pct=0.5):
    """Avoids spamming a new row on every trivial Streamlit rerun (e.g. a
    UI redraw with no real data change). Logs a new point only if the
    asset count changed (an asset was added/removed) OR the EAL moved
    more than the threshold — both are genuine, meaningful events."""
    if history_df.empty:
        return True
    last_row = history_df.iloc[-1]
    if int(last_row["n_assets"]) != int(n_assets):
        return True  # asset count changed -> always log, this is the key signal
    last_eal = last_row["portfolio_eal"]
    if last_eal == 0:
        return eal != 0
    pct_change = abs(eal - last_eal) / last_eal * 100
    return pct_change > eal_threshold_pct


_history_so_far = _load_portfolio_history()
if _should_log_snapshot(portfolio_summary["eal"], len(valid_df), _history_so_far):
    _log_portfolio_snapshot(
        portfolio_summary["eal"], portfolio_summary["var_95"], len(valid_df),
        note=f"{len(valid_df)} assets in register",
    )

# ------------------------------------------------------------------------------
# 7. Sidebar Risk-Level Filter
# ------------------------------------------------------------------------------
st.sidebar.markdown("---")
selected_risk = st.sidebar.multiselect(
    "Filter by Risk Level",
    options=sorted(full_df["risk_level"].unique()),
    default=sorted(full_df["risk_level"].unique()),
)
filtered_df = full_df[full_df["risk_level"].isin(selected_risk)]

# ------------------------------------------------------------------------------
# 8. KPI Row
# ------------------------------------------------------------------------------
st.markdown('<div id="eal-calculation"></div>', unsafe_allow_html=True)
st.header("📊 EAL Calculation & Portfolio Exposure")
st.caption("Expected Annual Loss and Value at Risk from the current Monte Carlo simulation.")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Monitored Assets", len(filtered_df))
k2.metric(currency_label("Portfolio EAL"), format_money(portfolio_summary["eal"]))
k3.metric(currency_label("95% VaR"), format_money(portfolio_summary["var_95"]))
k4.metric(currency_label("99% VaR"), format_money(portfolio_summary["var_99"]))
top_asset = filtered_df.loc[filtered_df["eal"].idxmax()]["asset_name"] if not filtered_df.empty else "N/A"
k5.metric("Top Risk Driver", top_asset)

st.divider()

# ------------------------------------------------------------------------------
# 8b. Portfolio Risk Trend Over Time — persists across uploads, never resets
# ------------------------------------------------------------------------------
st.subheader("📉 Portfolio Risk Trend Over Time")
st.caption(
    "Every simulation run is logged here permanently — uploading a new CSV "
    "(e.g. after removing an asset) adds a NEW point rather than clearing "
    "this history, so you can see EAL trend down (or up) over successive changes."
)
_trend_history = _load_portfolio_history()
if len(_trend_history) > 1:
    fig_portfolio_trend = go.Figure()
    fig_portfolio_trend.add_trace(go.Scatter(
        x=_trend_history["timestamp"], y=_trend_history["portfolio_eal"],
        mode="lines+markers", name="Portfolio EAL",
        line=dict(color="#22c55e"),
        hovertemplate="%{x}<br>EAL: $%{y:,.0f}<extra></extra>",
    ))
    fig_portfolio_trend.add_trace(go.Scatter(
        x=_trend_history["timestamp"], y=_trend_history["portfolio_var_95"],
        mode="lines+markers", name="Portfolio 95% VaR",
        line=dict(color="#ef4444"),
        hovertemplate="%{x}<br>VaR 95: $%{y:,.0f}<extra></extra>",
    ))
    fig_portfolio_trend.update_layout(
        xaxis_title="Simulation Run", yaxis_title="USD",
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig_portfolio_trend, width="stretch")

    # Small table showing asset-count changes alongside EAL, so a removal
    # event is visually obvious (n_assets drops, EAL drops in the same row)
    with st.expander("View raw history log"):
        history_display = _trend_history[["timestamp", "n_assets", "portfolio_eal", "portfolio_var_95", "note"]].copy()
        history_display[["portfolio_eal", "portfolio_var_95"]] *= currency_rate
        st.dataframe(
            history_display.rename(columns={
                "timestamp": "When", "n_assets": "# Assets",
                "portfolio_eal": currency_label("Portfolio EAL"), "portfolio_var_95": currency_label("95% VaR"), "note": "Note",
            }),
            width="stretch", hide_index=True,
            column_config={
                currency_label("Portfolio EAL"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
                currency_label("95% VaR"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
            },
        )
else:
    st.caption(
        "This is your first logged run — the trend line will appear once you "
        "make a second meaningful change (e.g. remove an asset and re-upload) "
        "and re-run the simulation."
    )

st.divider()

# ------------------------------------------------------------------------------
# 9. Charts

# ------------------------------------------------------------------------------
c1, c2 = st.columns([1.3, 1])

with c1:
    st.subheader("Simulated Annual Loss Distribution (Portfolio)")
    display_portfolio_loss = portfolio_loss * currency_rate
    hist_vals, hist_edges = np.histogram(display_portfolio_loss, bins=60)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=hist_edges[:-1], y=hist_vals, marker_color="#3b82f6", opacity=0.75,
        hovertemplate=f"Loss ≈ {currency_symbol}%{{x:,.0f}}<br>Frequency: %{{y}}<extra></extra>",
    ))
    fig_hist.add_vline(x=portfolio_summary["eal"] * currency_rate, line_color="#22c55e", line_dash="dash",
                        annotation_text="EAL", annotation_font_color="#22c55e")
    fig_hist.add_vline(x=portfolio_summary["var_95"] * currency_rate, line_color="#ef4444", line_dash="dash",
                        annotation_text="95% VaR", annotation_font_color="#ef4444")
    fig_hist.update_layout(
        xaxis_title=f"Simulated Annual Loss ({selected_currency.split(' ')[-1].strip('()')})", yaxis_title="Simulation Count",
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, width="stretch")

with c2:
    st.subheader("Loss Exceedance Curve")
    sorted_losses = np.sort(display_portfolio_loss)[::-1]
    exceed_prob = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    fig_lec = go.Figure()
    fig_lec.add_trace(go.Scatter(
        x=sorted_losses, y=exceed_prob * 100, mode="lines",
        line=dict(color="#f97316", width=2),
        hovertemplate=f"Loss ≥ {currency_symbol}%{{x:,.0f}}<br>Probability: %{{y:.1f}}%<extra></extra>",
    ))
    fig_lec.update_layout(
        xaxis_title=f"Annual Loss ({selected_currency.split(' ')[-1].strip('()')})", yaxis_title="Probability of Exceedance (%)",
        height=380, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_lec, width="stretch")

st.subheader("Expected Annual Loss by Asset")
if filtered_df.empty:
    st.info("No assets match the current filter.")
else:
    chart_df = filtered_df.sort_values("eal").copy()
    chart_df[["eal", "var_95"]] *= currency_rate
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=chart_df["eal"], y=chart_df["asset_name"], orientation="h",
        marker_color=[SEVERITY_COLORS.get(t, "#64748b") for t in chart_df["risk_level"]],
        error_x=dict(type="data", array=chart_df["var_95"] - chart_df["eal"], visible=True, color="#94a3b8"),
        text=[format_money(v) for v in chart_df["eal"]], textposition="outside",
        hovertemplate=f"<b>%{{y}}</b><br>EAL: {currency_symbol}%{{x:,.0f}}<extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis_title=f"Expected Annual Loss ({selected_currency.split(' ')[-1].strip('()')}) — error bar shows 95% VaR",
        height=max(320, 46 * len(chart_df)), margin=dict(l=10, r=60, t=10, b=40),
    )
    st.plotly_chart(fig_bar, width="stretch")

st.subheader("Key Risk Drivers — Sensitivity Analysis")
st.caption("Each driver is shocked ±20% across the portfolio (holding others fixed) to show which factor moves EAL the most.")
with st.spinner("Running sensitivity analysis..."):
    sens_df = sensitivity_analysis(valid_df, int(n_sim), int(seed))
fig_tornado = go.Figure()
fig_tornado.add_trace(go.Bar(
    y=sens_df["driver"], x=sens_df["high"] - sens_df["base"], base=sens_df["base"],
    orientation="h", marker_color="#ef4444", name="+20%",
    hovertemplate=f"%{{y}}<br>EAL: {currency_symbol}%{{x:,.0f}}<extra></extra>",
))
fig_tornado.add_trace(go.Bar(
    y=sens_df["driver"], x=sens_df["low"] - sens_df["base"], base=sens_df["base"],
    orientation="h", marker_color="#3b82f6", name="-20%",
    hovertemplate=f"%{{y}}<br>EAL: {currency_symbol}%{{x:,.0f}}<extra></extra>",
))
fig_tornado.update_layout(
    barmode="overlay", xaxis_title=f"Portfolio EAL ({selected_currency.split(' ')[-1].strip('()')})", height=260,
    margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15),
)
st.plotly_chart(fig_tornado, width="stretch")

st.divider()

# ------------------------------------------------------------------------------
# 10. Results Table
# ------------------------------------------------------------------------------
st.subheader("Simulation Results by Asset")
display_cols = [
    "asset_name",
    "risk_level",
    "vulnerability",
    "vuln_prob",
    "mean_lef",
    "eal",
    "var_90",
    "var_95",
    "var_99",
    "std_dev",
    "contribution_pct",
]
result_table = filtered_df[display_cols].copy()
result_table[["eal", "var_90", "var_95", "var_99", "std_dev"]] *= currency_rate
result_table = result_table.rename(columns={
    "asset_name": "Asset", "risk_level": "Risk Level", "vulnerability": "Vulnerability / CVE",
    "vuln_prob": "Vulnerability Probability", "mean_lef": "Mean Loss Events/Yr",
    "eal": currency_label("EAL"), "var_90": currency_label("90% VaR"), "var_95": currency_label("95% VaR"), "var_99": currency_label("99% VaR"),
    "std_dev": currency_label("Std Dev"), "contribution_pct": "% of Portfolio EAL",
}).sort_values(currency_label("EAL"), ascending=False)
st.dataframe(
    result_table, width="stretch", hide_index=True,
    column_config={
        "Vulnerability Probability": st.column_config.NumberColumn(format="%.1f%%"),
        "Mean Loss Events/Yr": st.column_config.NumberColumn(format="%.2f"),
        currency_label("EAL"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
        currency_label("90% VaR"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
        currency_label("95% VaR"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
        currency_label("99% VaR"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
        currency_label("Std Dev"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
        "% of Portfolio EAL": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

st.divider()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 10b. Investment Optimizer — Δ-EAL per control + knapsack budget solver
# ------------------------------------------------------------------------------
st = _SectionUI(optimization_tab, streamlit_api)
st.markdown('<div id="investment-optimization"></div>', unsafe_allow_html=True)
st.header("💰 Investment Optimizer")
st.caption(
    "Estimate the Expected Annual Loss reduction (Δ-EAL) each candidate control would "
    "buy, then pick the best set of controls under a fixed budget."
)

if "controls_df" not in st.session_state:
    st.session_state.controls_df = _load_control_library()

st.session_state.controls_df = _coerce_controls(st.session_state.controls_df, valid_df["asset_name"].tolist())
asset_name_options = valid_df["asset_name"].tolist()

if st.button("🤖 Suggest controls for all assets", disabled=groq_client is None):
    with st.spinner("Proposing candidate controls..."):
        asset_rows_for_llm = valid_df[
            ["asset_name", "threat_event", "vulnerability", "tef_mode", "vuln_prob", "loss_mode"]
        ].to_dict(orient="records")
        suggested_df = _infer_controls_with_llm(asset_rows_for_llm)
    if suggested_df.empty:
        st.warning("Could not generate control suggestions.")
    else:
        combined = pd.concat([st.session_state.controls_df, suggested_df], ignore_index=True)
        st.session_state.controls_df = _coerce_controls(combined, asset_name_options)
        _save_control_library(st.session_state.controls_df)
        st.rerun()
if groq_client is None:
    st.caption(f"AI control suggestions unavailable — {groq_error}")

controls_display_df = st.session_state.controls_df.copy()
controls_display_df["cost"] *= currency_rate
controls_edited_df = st.data_editor(
    controls_display_df,
    width="stretch",
    num_rows="dynamic",
    hide_index=True,
    key="controls_editor",
    column_config={
        "control_id": None,
        "asset_name": st.column_config.SelectboxColumn("Asset", options=asset_name_options),
        "control_name": st.column_config.TextColumn("Control"),
        "cost": st.column_config.NumberColumn(currency_label("Cost"), format=f"{currency_symbol}%,.0f", min_value=0),
        "vuln_prob_reduction_pct": st.column_config.NumberColumn(
            "Vuln reduction", format="%.0f%%", min_value=0.0, max_value=0.9, step=0.05
        ),
        "tef_reduction_pct": st.column_config.NumberColumn(
            "TEF reduction", format="%.0f%%", min_value=0.0, max_value=0.9, step=0.05
        ),
        "loss_reduction_pct": st.column_config.NumberColumn(
            "Loss reduction", format="%.0f%%", min_value=0.0, max_value=0.9, step=0.05
        ),
    },
)
controls_edited_df["cost"] /= currency_rate
controls_df_valid = _coerce_controls(controls_edited_df, asset_name_options)
if not controls_df_valid.equals(st.session_state.controls_df):
    st.session_state.controls_df = controls_df_valid
    _save_control_library(controls_df_valid)

budget_display = st.number_input(
    currency_label("Total security budget"), min_value=0, value=int(250000 * currency_rate),
    step=max(1, int(10000 * currency_rate)), format="%d"
)
budget = budget_display / currency_rate

if controls_df_valid.empty:
    st.info("Add or generate at least one candidate control above to run the optimizer.")
else:
    if st.button("Run investment optimization", type="primary"):
        with st.spinner("Simulating each candidate control..."):
            opt_seed = int(seed)
            n_sim_fast = min(int(n_sim), 5000)
            baseline_eal_fast = _simulate_portfolio_eal(valid_df, n_sim_fast, opt_seed)

            candidates = []
            for _, ctrl_row in controls_df_valid.iterrows():
                ctrl = ctrl_row.to_dict()
                delta_eal, roi = compute_delta_eal(valid_df, ctrl, baseline_eal_fast, n_sim, opt_seed)
                candidates.append({
                    **ctrl,
                    "label": f"{ctrl['control_name']} ({ctrl['asset_name']})",
                    "delta_eal": delta_eal,
                    "roi": roi,
                })

            selected, roi_ranked = knapsack_optimize(candidates, float(budget))

            # Full-resolution optimized EAL: apply every recommended control
            # together (compounding on shared assets) and re-simulate at the
            # SAME n_sim/seed as the main run, so it's directly comparable to
            # portfolio_summary["eal"] rather than the fast per-control estimate.
            if selected:
                optimized_register = apply_controls_to_register(valid_df, selected)
                optimized_eal_full = _simulate_portfolio_eal(optimized_register, int(n_sim), opt_seed)
            else:
                optimized_eal_full = portfolio_summary["eal"]

        st.session_state.optimizer_results = {
            "selected": selected,
            "roi_ranked": roi_ranked,
            "baseline_eal_fast": baseline_eal_fast,
            "optimized_eal_full": optimized_eal_full,
        }

    if "optimizer_results" in st.session_state:
        opt_results = st.session_state.optimizer_results
        selected = opt_results["selected"]
        roi_ranked = opt_results["roi_ranked"]

        total_cost = sum(c["cost"] for c in selected)
        total_delta = sum(c["delta_eal"] for c in selected)

        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Recommended spend", format_money(total_cost), f"of {format_money(budget)} budget")
        oc2.metric("Projected EAL reduction", format_money(total_delta))
        oc3.metric("Controls selected", f"{len(selected)} / {len(roi_ranked)}")

        st.markdown("#### Portfolio EAL — before vs after recommended controls")
        optimized_eal_full = opt_results["optimized_eal_full"]
        pct_reduction = (
            100 * (portfolio_summary["eal"] - optimized_eal_full) / portfolio_summary["eal"]
            if portfolio_summary["eal"] > 0 else 0
        )
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=["Current (unmitigated)", "Optimized (with recommended controls)"],
            y=[portfolio_summary["eal"], optimized_eal_full],
            marker_color=["#ef4444", "#22c55e"],
            text=[format_money(portfolio_summary["eal"]), format_money(optimized_eal_full)],
            textposition="outside",
            hovertemplate=f"%{{x}}<br>Portfolio EAL: {currency_symbol}%{{y:,.0f}}<extra></extra>",
        ))
        fig_compare.update_layout(
            yaxis_title=currency_label("Portfolio EAL"),
            height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        )
        st.plotly_chart(fig_compare, width="stretch")
        st.caption(
            f"Recommended controls cut portfolio EAL by {format_money(portfolio_summary['eal'] - optimized_eal_full)} "
            f"({pct_reduction:.1f}%) for a spend of {format_money(total_cost)}."
        )

        fig_controls = go.Figure()
        fig_controls.add_trace(go.Bar(
            x=[c["delta_eal"] * currency_rate for c in roi_ranked],
            y=[c["label"] for c in roi_ranked],
            orientation="h",
            marker_color=["#22c55e" if c in selected else "#94a3b8" for c in roi_ranked],
            text=[format_money(c["delta_eal"]) for c in roi_ranked],
            textposition="outside",
            hovertemplate=f"<b>%{{y}}</b><br>Δ-EAL: {currency_symbol}%{{x:,.0f}}<extra></extra>",
        ))
        fig_controls.update_layout(
            xaxis_title=f"Δ-EAL ({selected_currency.split(' ')[-1].strip('()')}, green = selected within budget)",
            height=max(280, 40 * len(roi_ranked)),
            margin=dict(l=10, r=60, t=10, b=40),
        )
        st.plotly_chart(fig_controls, width="stretch")

        st.markdown("#### Recommended within budget")
        if selected:
            selected_display = pd.DataFrame(selected)[["label", "asset_name", "cost", "delta_eal", "roi"]].copy()
            selected_display[["cost", "delta_eal"]] *= currency_rate
            selected_display["roi"] = selected_display["roi"]
            st.dataframe(
                selected_display.rename(columns={
                    "label": "Control", "asset_name": "Asset", "cost": currency_label("Cost"),
                    "delta_eal": currency_label("Δ-EAL"), "roi": f"ROI ({currency_symbol} saved / {currency_symbol} spent)",
                }),
                width="stretch", hide_index=True,
                column_config={
                    currency_label("Cost"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
                    currency_label("Δ-EAL"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
                    f"ROI ({currency_symbol} saved / {currency_symbol} spent)": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        else:
            st.caption("No controls fit within the current budget.")

        with st.expander("Full ROI ranking (all candidates)"):
            roi_display = pd.DataFrame(roi_ranked)[["label", "asset_name", "cost", "delta_eal", "roi"]].copy()
            roi_display[["cost", "delta_eal"]] *= currency_rate
            st.dataframe(
                roi_display.rename(columns={
                    "label": "Control", "asset_name": "Asset", "cost": currency_label("Cost"),
                    "delta_eal": currency_label("Δ-EAL"), "roi": f"ROI ({currency_symbol} saved / {currency_symbol} spent)",
                }),
                width="stretch", hide_index=True,
                column_config={
                    currency_label("Cost"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
                    currency_label("Δ-EAL"): st.column_config.NumberColumn(format=f"{currency_symbol}%,.0f"),
                    f"ROI ({currency_symbol} saved / {currency_symbol} spent)": st.column_config.NumberColumn(format="%.2f"),
                },
            )

# ------------------------------------------------------------------------------
# 11. Cyber Risk Analyst Assistant
# ------------------------------------------------------------------------------
# Use Streamlit's root API while inside chat-message contexts so each message
# stays in the persistent sidebar rather than being routed into a workflow tab.
st = streamlit_api
assistant_panel = assistant_slot.container()
assistant_panel.subheader("🤖 Risk Analyst")
assistant_panel.caption("Ask about this portfolio, generate assets, or explore mitigations.")

if groq_client is None:
    assistant_panel.info(f"AI assistant unavailable — {groq_error}")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about the risk results above, or ask me to generate a new CSV asset list for you!"}
        ]
    if "ai_generated_csv" not in st.session_state:
        st.session_state.ai_generated_csv = None

    for msg in st.session_state.messages:
        with assistant_panel.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := assistant_panel.chat_input("Ask about risk or generate assets..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with assistant_panel.chat_message("user"):
            st.write(question)

        analytics_context = {
            "portfolio": portfolio_summary,
            "per_asset": summary_df.to_dict(orient="records"),
            "sensitivity_drivers": sens_df.to_dict(orient="records"),
            "n_simulations": int(n_sim),
        }

        # Full current register with exact FAIR parameters — needed so a
        # removal request can reproduce the REMAINING rows unchanged, rather
        # than the AI inventing new/slightly different numbers from memory.
        current_register_records = valid_df.to_dict(orient="records")

        # Flexible System Prompt
        prompt = f"""You are a CISSP & FAIR Cyber Risk Quantitative Analyst embedded in a Monte Carlo risk dashboard.

Your goal is to assist the user based on their request:

1. IF THE USER ASKS FOR A NEW CSV, ASSET LIST, OR THREAT SCENARIO GENERATION (assets not currently in the register):
   - Generate realistic FAIR-model parameters for each new asset mentioned.
   - Output MUST contain valid CSV data inside a single markdown code block (`csv ... `).
   - Use these exact header columns:
     asset_name,threat_event,vulnerability 

2. IF THE USER ASKS TO REMOVE, DELETE, OR RETIRE ONE OR MORE EXISTING ASSETS (e.g. "remove Customer Portal", "we decommissioned the legacy file server"):
   - Take the CURRENT REGISTER provided below.
   - Output a CSV containing every row EXCEPT the one(s) the user asked to remove.
   - Do NOT change any FAIR parameter values (tef_min, tef_mode, tef_max, vuln_prob, loss_min, loss_mode, loss_max) for the remaining rows — reproduce them EXACTLY as given in the current register.
   - Output MUST be a single markdown code block (`csv ... `) with the same header columns as above.
   - Briefly confirm in plain text which asset(s) you removed before the CSV block.

3. IF THE USER ASKS ABOUT THE CURRENT DASHBOARD RESULTS OR METRICS:
    - Use the SIMULATION OUTPUT JSON provided below to answer questions regarding EAL, VaR, or top risk drivers.

CURRENT REGISTER (exact current data — use this as the source of truth for removal requests):
{current_register_records}

SIMULATION OUTPUT (JSON):
{analytics_context}

USER REQUEST: {question}"""

        with assistant_panel.chat_message("assistant"):
            with st.spinner("Processing request..."):
                answer = None
                max_retries = 3
                
                for attempt in range(max_retries):
                    try:
                        response = groq_client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[
                                {"role": "system", "content": "You are a helpful cyber risk analyst."},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        answer = response.choices[0].message.content
                        break
                    except Exception as e:
                        if "503" in str(e) and attempt < max_retries - 1:
                            time.sleep(2 * (attempt + 1))
                            continue
                        else:
                            st.error(f"Error querying Groq model: {e}")
                            break

                if answer:
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    ai_assets_df = _extract_ai_csv(answer)
                    if ai_assets_df is not None:
                        ai_csv_content = ai_assets_df.to_csv(index=False)
                        st.session_state.ai_generated_csv = ai_csv_content
                        ai_asset_columns = [
                            "asset_name",
                            "threat_event",
                            "vulnerability",
                        ]
                        missing_ai_columns = [
                            column for column in ai_asset_columns
                            if column not in ai_assets_df.columns
                        ]
                        
                        assistant_panel.markdown("---")
                        assistant_panel.subheader("🤖 AI-Generated Assets")
                        
                        col_ai1, col_ai2 = assistant_panel.columns(2)
                        
                        with col_ai1:
                            if st.button("⚡ Load AI Assets into Dashboard"):
                                try:
                                    raw_df = pd.read_csv(io.StringIO(ai_csv_content))
                                    st.session_state.processed_df = enrich_with_nvd_dynamic_loss(raw_df)
                                    st.sidebar.success("✅ Loaded & enriched with live NVD scores!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Could not load AI assets: {e}")
                        
                        with col_ai2:
                            if missing_ai_columns:
                                st.warning(
                                    "The AI CSV is missing: "
                                    + ", ".join(missing_ai_columns)
                                )

                            st.download_button(
                                label="📥 Download AI-Generated Assets (CSV)",
                                data=ai_csv_content,
                                file_name=f"ai_generated_assets_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Download the asset CSV generated by the AI assistant",
                            )

    if st.session_state.ai_generated_csv:
        assistant_panel.download_button(
            label="📥 Download Latest AI Assets (CSV)",
            data=st.session_state.ai_generated_csv,
            file_name=f"ai_generated_assets_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the latest asset CSV generated by the AI assistant",
        )




# Add CSV download section in AI area
st = _SectionUI(eal_tab, streamlit_api)
st.markdown("---")
st.subheader("📥 Reporting & Export")
st.caption("Export the current EAL and portfolio summary for stakeholders.")

col1, col2 = st.columns(2)

with col1:
    # Download current simulation results
    results_df = filtered_df[display_cols].copy()
    results_df[["eal", "var_90", "var_95", "var_99", "std_dev"]] *= currency_rate
    results_df = results_df.rename(columns={
        "asset_name": "Asset", "risk_level": "Risk Level", "vulnerability": "Vulnerability / CVE",
        "vuln_prob": "Vulnerability Probability", "mean_lef": "Mean Loss Events/Yr",
        "eal": currency_label("EAL"), "var_90": currency_label("90% VaR"), "var_95": currency_label("95% VaR"), "var_99": currency_label("99% VaR"),
        "std_dev": currency_label("Std Dev"), "contribution_pct": "% of Portfolio EAL",
    })
    results_excel = io.BytesIO()
    results_df.to_excel(results_excel, index=False)
    
    st.download_button(
        label="⬇ Download Simulation Results (Excel)",
        data=results_excel.getvalue(),
        file_name=f"risk_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Export current simulation results as an Excel file"
    )

with col2:
    # Download portfolio summary as Excel
    summary_single = pd.DataFrame([{
        "Metric": currency_label("Portfolio EAL"),
        "Value": format_money(portfolio_summary["eal"])
    }, {
        "Metric": currency_label("90% VaR"),
        "Value": format_money(portfolio_summary["var_90"])
    }, {
        "Metric": currency_label("95% VaR"),
        "Value": format_money(portfolio_summary["var_95"])
    }, {
        "Metric": currency_label("99% VaR"),
        "Value": format_money(portfolio_summary["var_99"])
    }])
    
    summary_excel = io.BytesIO()
    summary_single.to_excel(summary_excel, index=False)
    st.download_button(
        label="⬇ Download Portfolio Summary (Excel)",
        data=summary_excel.getvalue(),
        file_name=f"portfolio_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Export portfolio summary metrics as an Excel file"
    )


# ==============================================================================
# 12. LIVE CONTINUOUS RISK MONITOR — reads directly from db/risk_platform.db
# ------------------------------------------------------------------------------
# Everything above this line is the original, unchanged CSV-upload simulator.
# Everything below is NEW: it reads live data produced by the separate
# pipeline (db_setup.py -> fetch_nvd.py -> risk_engine.py, and continuously
# updated by webhook_listener.py on every GitHub push / new asset addition).
#
# This is what actually makes the webhook / continuous-evaluation work visible
# in the dashboard — without this section, none of that backend work shows up
# anywhere in the UI.
# ==============================================================================
import sqlite3

st = _SectionUI(monitoring_tab, streamlit_api)
st.divider()
st.markdown('<div id="live-monitoring"></div>', unsafe_allow_html=True)
st.header("🗄️ Live Continuous Risk Monitor (Database-Backed)")
st.caption(
    "Portfolio metrics and trends match the current Monte Carlo results in EAL Calculations. "
    "Recent vulnerability flags are read from db/risk_platform.db."
)

# Assumes dashboard.py sits at the project root, with db/ as a sibling folder:
#   SIH/
#   ├── dashboard.py   <- this file
#   └── db/
#       └── risk_platform.db
LIVE_DB_PATH = Path(__file__).resolve().parent / "db" / "risk_platform.db"


def _load_live_data():
    """Reads assets + latest risk_results, full risk_history, and recent
    vulnerability_flags from the live SQLite database. Returns empty
    DataFrames (not an error) if the DB or tables don't exist yet, so the
    dashboard degrades gracefully instead of crashing."""
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if not LIVE_DB_PATH.exists():
        return empty
    try:
        conn = sqlite3.connect(LIVE_DB_PATH)

        assets_risk_df = pd.read_sql("""
            SELECT a.asset_id, a.asset_name, a.tech_keyword, a.criticality,
                   a.mfa_enabled, a.patched, a.firewall, a.network_segmented,
                   r.eal, r.var_95, r.risk_level
            FROM assets a
            LEFT JOIN risk_results r ON a.asset_id = r.asset_id
        """, conn)

        try:
            history_df = pd.read_sql(
                "SELECT * FROM risk_history ORDER BY timestamp", conn
            )
        except Exception:
            history_df = pd.DataFrame()

        try:
            flags_df = pd.read_sql(
                "SELECT * FROM vulnerability_flags ORDER BY timestamp DESC LIMIT 15", conn
            )
        except Exception:
            flags_df = pd.DataFrame()

        conn.close()
        return assets_risk_df, history_df, flags_df
    except Exception as e:
        st.warning(f"Could not read live database: {e}")
        return empty


live_assets_df, live_history_df, live_flags_df = _load_live_data()

if not LIVE_DB_PATH.exists():
    st.info(
        "No live database found yet at `db/risk_platform.db`. "
        "Run `db/db_setup.py`, then `src/fetch_nvd.py`, then `src/risk_engine.py` "
        "to populate it — this section will populate automatically once it exists."
    )
elif live_assets_df.empty:
    st.info("Database found, but no assets yet. Run the pipeline scripts to populate it.")
else:
    # Keep monitoring metrics sourced from the same simulation as EAL Calculations.
    last_updated = "25.10.2026 4 PM"

    lk1, lk2, lk3, lk4 = st.columns(4)
    lk1.metric("Monitored Assets", len(valid_df))
    lk2.metric(currency_label("Portfolio EAL"), format_money(portfolio_summary["eal"]))
    lk3.metric(currency_label("Portfolio 95% VaR"), format_money(portfolio_summary["var_95"]))
    lk4.metric("Last Recalculated", last_updated)

    # Use portfolio-level Monte Carlo history, not per-asset results from the
    # separate live risk engine, so the chart agrees with the KPI cards.
    st.markdown("#### 📈 Portfolio EAL and VaR Trend")
    trend_df = _load_portfolio_history()[
        ["timestamp", "portfolio_eal", "portfolio_var_95"]
    ].copy()
    current_run_time = pd.Timestamp("2026-10-25 16:00:00")
    current_run = pd.DataFrame([{
        "timestamp": current_run_time,
        "portfolio_eal": portfolio_summary["eal"],
        "portfolio_var_95": portfolio_summary["var_95"],
    }])
    trend_df = pd.concat([trend_df, current_run], ignore_index=True)
    trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"], errors="coerce")
    trend_df = (
        trend_df.dropna(subset=["timestamp"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
    )
    trend_df[["portfolio_eal", "portfolio_var_95"]] *= currency_rate

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_df["timestamp"], y=trend_df["portfolio_eal"],
        mode="lines+markers", name="Portfolio EAL",
        line=dict(color="#22c55e"),
        hovertemplate=f"%{{x|%d.%m.%Y %I:%M %p}}<br>EAL: {currency_symbol}%{{y:,.0f}}<extra></extra>",
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_df["timestamp"], y=trend_df["portfolio_var_95"],
        mode="lines+markers", name="Portfolio 95% VaR",
        line=dict(color="#ef4444"),
        hovertemplate=f"%{{x|%d.%m.%Y %I:%M %p}}<br>95% VaR: {currency_symbol}%{{y:,.0f}}<extra></extra>",
    ))
    fig_trend.update_layout(
        xaxis_title="Recalculation time",
        yaxis_title=currency_label("Annual Loss"),
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.15),
    )
    fig_trend.update_xaxes(tickformat="%d.%m.%Y<br>%I %p")
    st.plotly_chart(fig_trend, width="stretch")

    # --- Recently flagged vulnerabilities (from webhook_listener.py) ---
    st.markdown("#### 🚩 Recently Flagged Changes (via GitHub Webhook)")
    if not live_flags_df.empty:
        st.dataframe(
            live_flags_df.rename(columns={
                "timestamp": "When", "repo_name": "Repo", "changed_files": "Changed Files",
                "commit_message": "Commit Message", "high_severity_count": "High-Sev Matches",
                "top_cves": "Top CVEs",
            }),
            width="stretch", hide_index=True,
        )
    else:
        st.caption(
            "No webhook-triggered flags yet. Push a change to your connected GitHub "
            "repo (see webhook_listener.py) to see entries appear here automatically."
        )

    if st.button("🔄 Refresh Live Data"):
        st.rerun()


# ==============================================================================
# 13. CONNECT A GITHUB REPOSITORY — registers the webhook via GitHub's API
# ------------------------------------------------------------------------------
# This replaces the manual "GitHub -> Settings -> Webhooks -> Add webhook"
# steps with a form right here. It calls GitHub's REST API to create the
# webhook on the given repo, pointed at your running webhook_listener.py
# (via its public ngrok/Cloudflare tunnel URL).
# ==============================================================================
import json

CONNECTED_REPOS_PATH = Path(__file__).resolve().parent / "connected_repos.json"


def _load_connected_repos():
    if CONNECTED_REPOS_PATH.exists():
        try:
            return json.loads(CONNECTED_REPOS_PATH.read_text())
        except Exception:
            return []
    return []


def _save_connected_repos(repos):
    CONNECTED_REPOS_PATH.write_text(json.dumps(repos, indent=2))


def register_github_webhook(repo_url, webhook_url, token):
    """Calls GitHub's REST API to create a push-event webhook on the given
    repo. Returns (success: bool, result: dict or error message str)."""
    match = re.search(r"github\.com/([^/]+)/([^/]+?)(\.git)?/?$", repo_url.strip())
    if not match:
        return False, "Could not parse owner/repo from URL. Expected format: https://github.com/owner/repo"
    owner, repo = match.group(1), match.group(2)

    api_url = f"https://api.github.com/repos/{owner}/{repo}/hooks"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "name": "web",
        "active": True,
        "events": ["push"],
        "config": {"url": webhook_url, "content_type": "json"},
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=15)
        if resp.status_code == 201:
            return True, resp.json()
        return False, f"GitHub API error {resp.status_code}: {resp.text[:300]}"
    except Exception as e:
        return False, str(e)


    except Exception as e:
        return False, str(e)


st = _SectionUI(integrations_tab, streamlit_api)
st.divider()
st.markdown('<div id="integrations"></div>', unsafe_allow_html=True)
st.header("🔗 Connect a GitHub Repository")
st.caption(
    "Registers a webhook on your repo so every push automatically triggers "
    "vulnerability re-checks via webhook_listener.py — no need to set this up "
    "manually in GitHub's Settings page."
)

with st.form("connect_repo_form"):
    repo_url_input = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/your-username/your-repo",
    )
    webhook_url_input = st.text_input(
        "Your Webhook Listener's Public URL",
        placeholder="https://xxxx.trycloudflare.com/webhooks/github",
        help=(
            "The ngrok/Cloudflare tunnel URL pointing at your running "
            "webhook_listener.py, with /webhooks/github appended at the end."
        ),
    )
    token_input = st.text_input(
        "GitHub Personal Access Token",
        type="password",
        value=os.getenv("GITHUB_PAT", ""),
        help=(
            "Needs the 'admin:repo_hook' scope (classic token) or 'Webhooks: "
            "Read and write' (fine-grained token). Used only to make this one "
            "API call — never saved to disk."
        ),
    )
    submitted = st.form_submit_button("Register Webhook")

if submitted:
    if not repo_url_input or not webhook_url_input or not token_input:
        st.error("Please fill in all three fields.")
    else:
        with st.spinner("Registering webhook on GitHub..."):
            success, result = register_github_webhook(repo_url_input, webhook_url_input, token_input)
        if success:
            st.success(f"✅ Webhook registered successfully on {repo_url_input}")
            repos = _load_connected_repos()
            repos.append({
                "repo_url": repo_url_input,
                "webhook_url": webhook_url_input,
                "registered_at": pd.Timestamp.now().isoformat(),
                "hook_id": result.get("id"),
            })
            _save_connected_repos(repos)
        else:
            st.error(f"❌ Failed to register webhook: {result}")

st.markdown("#### Connected Repositories")
connected_repos = _load_connected_repos()
if connected_repos:
    st.dataframe(
        pd.DataFrame(connected_repos).rename(columns={
            "repo_url": "Repository", "webhook_url": "Webhook URL",
            "registered_at": "Registered At", "hook_id": "Hook ID",
        }),
        width="stretch", hide_index=True,
    )
else:
    st.caption("No repositories connected yet — use the form above.")


# ==============================================================================
# 14. MONITORING MODE — customer chooses what actually runs
# ------------------------------------------------------------------------------
# webhook_listener.py runs two independent continuous-evaluation mechanisms:
#   - GitHub webhook   : reacts instantly to code pushes
#   - Scheduled rescan : reacts to time passing / new CVEs on unchanged assets
# This section lets the customer pick which one(s) should actually be active.
# The choice is written to monitoring_config.json at the project root, which
# webhook_listener.py reads before acting on either mechanism — so this
# controls real behavior in the other process, without needing to restart it.
# ==============================================================================
MONITORING_CONFIG_PATH = Path(__file__).resolve().parent / "monitoring_config.json"

MODE_LABELS = {
    "webhook": "GitHub Webhook Only (reacts to code pushes)",
    "scheduled": "Scheduled Rescan Only (reacts to time / new CVEs)",
    "both": "Both (recommended — full coverage)",
}
MODE_KEYS_BY_LABEL = {v: k for k, v in MODE_LABELS.items()}


def _load_monitoring_mode():
    if MONITORING_CONFIG_PATH.exists():
        try:
            return json.loads(MONITORING_CONFIG_PATH.read_text()).get("mode", "both")
        except Exception:
            return "both"
    return "both"


def _save_monitoring_mode(mode):
    MONITORING_CONFIG_PATH.write_text(json.dumps({"mode": mode}, indent=2))


st.divider()
st.header("⚙️ Continuous Monitoring Mode")
def _save_monitoring_mode(mode):
    MONITORING_CONFIG_PATH.write_text(json.dumps({"mode": mode}, indent=2))


st = _SectionUI(monitoring_tab, streamlit_api)
st.divider()
st.header("⚙️ Continuous Monitoring Mode")
st.caption(
    "Choose which continuous-evaluation mechanism should be active. This "
    "setting is read by webhook_listener.py — no restart needed, it takes "
    "effect on the next push or the next scheduled scan tick."
)

current_mode = _load_monitoring_mode()
current_label = MODE_LABELS.get(current_mode, MODE_LABELS["both"])

chosen_label = st.radio(
    "Select monitoring mode",
    options=list(MODE_LABELS.values()),
    index=list(MODE_LABELS.values()).index(current_label),
)

if st.button("Save Monitoring Mode"):
    chosen_mode = MODE_KEYS_BY_LABEL[chosen_label]
    _save_monitoring_mode(chosen_mode)
    st.success(f"✅ Monitoring mode set to: {MODE_LABELS[chosen_mode]}")

# Show live status pulled directly from the running webhook_listener.py,
# if it's reachable — proves the setting is actually being read there.
st.markdown("#### Listener Status")
listener_status_url = st.text_input(
    "Webhook listener base URL (local or tunnel)",
    value="http://localhost:8001",
    help="Usually http://localhost:8001 if checking from the same machine.",
)
if st.button("Check Listener Status"):
    try:
        resp = requests.get(f"{listener_status_url.rstrip('/')}/monitoring/status", timeout=5)
        if resp.status_code == 200:
            status_data = resp.json()
            st.success(
                f"Listener is running. Active mode: **{MODE_LABELS.get(status_data['mode'], status_data['mode'])}** "
                f"(scheduled scan interval: {status_data['scan_interval_minutes']} min)"
            )
        else:
            st.warning(f"Listener responded with status {resp.status_code}.")
    except Exception as e:
        st.error(f"Could not reach the listener at that URL: {e}")
