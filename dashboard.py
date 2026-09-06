"""
Cyber Risk Quantification Dashboard
------------------------------------
FAIR-style quantitative risk analysis using a numpy-vectorized Monte Carlo
engine with real-time NVD API vulnerability score enrichment.

Run with:  streamlit run dashboard.py
"""

import io
import os
from pathlib import Path
import time
import requests
import numpy as np
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
)
st.title("🛡️ Cyber Risk Quantification Dashboard")
st.caption("FAIR-model Monte Carlo simulation • Real-time NVD Vulnerability Scoring • recomputes live as you edit data")

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


def simulate_asset(row, n_sim, rng, max_events_cap=60):
    tef = pert_sample(row["tef_min"], row["tef_mode"], row["tef_max"], n_sim, rng)
    vuln = np.clip(rng.normal(row["vuln_prob"], row["vuln_prob"] * 0.15 + 0.02, n_sim), 0.0, 1.0)
    lef = tef * vuln

    counts = rng.poisson(lef)
    max_count = int(min(max(counts.max(), 1), max_events_cap))
    counts = np.minimum(counts, max_count)

    magnitudes = pert_sample(
        row["loss_min"], row["loss_mode"], row["loss_max"],
        size=(n_sim, max_count), rng=rng,
    )
    col_idx = np.arange(max_count)
    mask = col_idx[None, :] < counts[:, None]
    annual_loss = np.where(mask, magnitudes, 0.0).sum(axis=1)
    return annual_loss, lef


@st.cache_data(show_spinner=False)
def run_portfolio_simulation(register_json: str, n_sim: int, seed: int):
    register = pd.read_json(io.StringIO(register_json), orient="records")
    rng = np.random.default_rng(seed)

    per_asset_losses = {}
    per_asset_lef = {}
    for _, row in register.iterrows():
        losses, lef = simulate_asset(row, n_sim, rng)
        per_asset_losses[row["asset_name"]] = losses
        per_asset_lef[row["asset_name"]] = float(np.mean(lef))

    loss_matrix = np.column_stack(list(per_asset_losses.values()))
    portfolio_loss = loss_matrix.sum(axis=1)

    summary_rows = []
    for name in per_asset_losses:
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
        for _, row in register_df.iterrows():
            r = row.copy()
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
            losses, _ = simulate_asset(r, n_sim_fast, rng)
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
    industry_avg_breach_cost = INDUSTRY_AVG_BREACH_COST.get(
        sector, INDUSTRY_AVG_BREACH_COST["default"]
    )
    severity_factor = np.clip(impact_score / 10.0, 0.05, 1.0)
    likelihood_factor = (
        epss_score
        if epss_score is not None
        else np.clip(exploitability_score / 10.0, 0.05, 1.0)
    )

    loss_mode = round(industry_avg_breach_cost * severity_factor, -3)
    loss_min = round(loss_mode * (0.3 + 0.4 * (1 - likelihood_factor)), -3)
    loss_max = round(loss_mode * (2.0 + 3.0 * likelihood_factor), -3)

    return loss_min, loss_mode, loss_max, likelihood_factor


def infer_sector(asset_name):
    name_lower = str(asset_name).lower()
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
        tef_mode = round(max(0.5, likelihood_factor * 8.0), 1)
        tef_min = round(max(0.1, tef_mode * 0.3), 1)
        tef_max = round(tef_mode * 2.5, 1)

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
            "loss_min": 80000,
            "loss_mode": 300000,
            "loss_max": 900000,
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

    return df.reset_index(drop=True), problems


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


# ------------------------------------------------------------------------------
# Streamlit UI Controls
# ------------------------------------------------------------------------------
st.sidebar.header("📁 Asset Register")

uploaded = st.sidebar.file_uploader(
    "Upload your assets (CSV or XLSX)",
    type=["csv", "xlsx", "xls"],
    help=(
        "Must include at least 'asset_name', 'threat_event', 'vulnerability'"
        " columns."
    ),
)

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

using_sample = False

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "last_uploaded_id" not in st.session_state:
    st.session_state.last_uploaded_id = None

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

        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")
            register_df = _sample_register()
            using_sample = True
    else:
        register_df = st.session_state.processed_df
else:
    st.session_state.last_uploaded_id = None
    st.session_state.processed_df = None
    register_df = _sample_register()
    using_sample = True

st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Monte Carlo Settings")
n_sim = st.sidebar.select_slider(
    "Simulations per asset",
    options=[1000, 2000, 5000, 10000, 20000, 50000],
    value=10000,
)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear Cache & Re-scan NVD"):
    st.session_state.clear()
    st.rerun()
# ------------------------------------------------------------------------------
# 5. Editable register — real-time simulation trigger
# ------------------------------------------------------------------------------
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
        "loss_min": st.column_config.NumberColumn("Loss min ($)", format="$%,.0f"),
        "loss_mode": st.column_config.NumberColumn("Loss likely ($)", format="$%,.0f"),
        "loss_max": st.column_config.NumberColumn("Loss max ($)", format="$%,.0f"),
        "risk_level": st.column_config.SelectboxColumn(
            "Risk Level", options=["Critical", "High", "Medium", "Low"]
        ),
    },
)

valid_df, edit_problems = _coerce_register(edited_df)
if edit_problems:
    st.warning(" / ".join(edit_problems))

if valid_df.empty:
    st.error("No valid assets to simulate. Fix the register above.")
    st.stop()

# ------------------------------------------------------------------------------
# 6. Run Monte Carlo Simulation
# ------------------------------------------------------------------------------
register_json = valid_df.to_json(orient="records")

with st.spinner("Running Monte Carlo simulation..."):
    summary_df, portfolio_summary, portfolio_loss = run_portfolio_simulation(register_json, int(n_sim), int(seed))

full_df = valid_df.merge(summary_df, on="asset_name", how="left")

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
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Monitored Assets", len(filtered_df))
k2.metric("Portfolio EAL", f"${portfolio_summary['eal']:,.0f}")
k3.metric("95% VaR", f"${portfolio_summary['var_95']:,.0f}")
k4.metric("99% VaR", f"${portfolio_summary['var_99']:,.0f}")
top_asset = filtered_df.loc[filtered_df["eal"].idxmax()]["asset_name"] if not filtered_df.empty else "N/A"
k5.metric("Top Risk Driver", top_asset)

st.divider()

# ------------------------------------------------------------------------------
# 9. Charts
# ------------------------------------------------------------------------------
c1, c2 = st.columns([1.3, 1])

with c1:
    st.subheader("Simulated Annual Loss Distribution (Portfolio)")
    hist_vals, hist_edges = np.histogram(portfolio_loss, bins=60)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=hist_edges[:-1], y=hist_vals, marker_color="#3b82f6", opacity=0.75,
        hovertemplate="Loss ≈ $%{x:,.0f}<br>Frequency: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(x=portfolio_summary["eal"], line_color="#22c55e", line_dash="dash",
                        annotation_text="EAL", annotation_font_color="#22c55e")
    fig_hist.add_vline(x=portfolio_summary["var_95"], line_color="#ef4444", line_dash="dash",
                        annotation_text="95% VaR", annotation_font_color="#ef4444")
    fig_hist.update_layout(
        xaxis_title="Simulated Annual Loss (USD)", yaxis_title="Simulation Count",
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, width="stretch")

with c2:
    st.subheader("Loss Exceedance Curve")
    sorted_losses = np.sort(portfolio_loss)[::-1]
    exceed_prob = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    fig_lec = go.Figure()
    fig_lec.add_trace(go.Scatter(
        x=sorted_losses, y=exceed_prob * 100, mode="lines",
        line=dict(color="#f97316", width=2),
        hovertemplate="Loss ≥ $%{x:,.0f}<br>Probability: %{y:.1f}%<extra></extra>",
    ))
    fig_lec.update_layout(
        xaxis_title="Annual Loss (USD)", yaxis_title="Probability of Exceedance (%)",
        height=380, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_lec, width="stretch")

st.subheader("Expected Annual Loss by Asset")
if filtered_df.empty:
    st.info("No assets match the current filter.")
else:
    chart_df = filtered_df.sort_values("eal")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=chart_df["eal"], y=chart_df["asset_name"], orientation="h",
        marker_color=[SEVERITY_COLORS.get(t, "#64748b") for t in chart_df["risk_level"]],
        error_x=dict(type="data", array=chart_df["var_95"] - chart_df["eal"], visible=True, color="#94a3b8"),
        text=[f"${v:,.0f}" for v in chart_df["eal"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>EAL: $%{x:,.0f}<extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis_title="Expected Annual Loss (USD) — error bar shows 95% VaR",
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
    hovertemplate="%{y}<br>EAL: $%{x:,.0f}<extra></extra>",
))
fig_tornado.add_trace(go.Bar(
    y=sens_df["driver"], x=sens_df["low"] - sens_df["base"], base=sens_df["base"],
    orientation="h", marker_color="#3b82f6", name="-20%",
    hovertemplate="%{y}<br>EAL: $%{x:,.0f}<extra></extra>",
))
fig_tornado.update_layout(
    barmode="overlay", xaxis_title="Portfolio EAL (USD)", height=260,
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
result_table = filtered_df[display_cols].rename(columns={
    "asset_name": "Asset", "risk_level": "Risk Level", "vulnerability": "Vulnerability / CVE",
    "vuln_prob": "Vulnerability Probability", "mean_lef": "Mean Loss Events/Yr",
    "eal": "EAL ($)", "var_90": "90% VaR ($)", "var_95": "95% VaR ($)", "var_99": "99% VaR ($)",
    "std_dev": "Std Dev ($)", "contribution_pct": "% of Portfolio EAL",
}).sort_values("EAL ($)", ascending=False)
st.dataframe(
    result_table, width="stretch", hide_index=True,
    column_config={
        "Vulnerability Probability": st.column_config.NumberColumn(format="%.1f%%"),
        "Mean Loss Events/Yr": st.column_config.NumberColumn(format="%.2f"),
        "EAL ($)": st.column_config.NumberColumn(format="$%,.0f"),
        "90% VaR ($)": st.column_config.NumberColumn(format="$%,.0f"),
        "95% VaR ($)": st.column_config.NumberColumn(format="$%,.0f"),
        "99% VaR ($)": st.column_config.NumberColumn(format="$%,.0f"),
        "Std Dev ($)": st.column_config.NumberColumn(format="$%,.0f"),
        "% of Portfolio EAL": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

st.divider()

# ------------------------------------------------------------------------------
# 11. Cyber Risk Analyst Assistant
# ------------------------------------------------------------------------------
st.subheader("🤖 Cyber Risk Analyst Assistant")

if groq_client is None:
    st.info(f"AI assistant unavailable — {groq_error}")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about the risk results above, or ask me to generate a new CSV asset list for you!"}
        ]
    if "ai_generated_csv" not in st.session_state:
        st.session_state.ai_generated_csv = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask a question or request a new asset CSV..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        analytics_context = {
            "portfolio": portfolio_summary,
            "per_asset": summary_df.to_dict(orient="records"),
            "sensitivity_drivers": sens_df.to_dict(orient="records"),
            "n_simulations": int(n_sim),
        }

        # Flexible System Prompt
        prompt = f"""You are a CISSP & FAIR Cyber Risk Quantitative Analyst embedded in a Monte Carlo risk dashboard.

Your goal is to assist the user based on their request:

1. IF THE USER ASKS FOR A NEW CSV, ASSET LIST, OR THREAT SCENARIO GENERATION:
   - Generate realistic FAIR-model parameters for each asset mentioned.
   - Output MUST contain valid CSV data inside a single markdown code block (`csv ... `).
   - Use these exact header columns:
     asset_name,threat_event,vulnerability,risk_level,tef_min,tef_mode,tef_max,vuln_prob,loss_min,loss_mode,loss_max

2. IF THE USER ASKS ABOUT THE CURRENT DASHBOARD RESULTS OR METRICS:
    - Use the SIMULATION OUTPUT JSON provided below to answer questions regarding EAL, VaR, or top risk drivers.

SIMULATION OUTPUT (JSON):
{analytics_context}

USER REQUEST: {question}"""

        with st.chat_message("assistant"):
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
                        
                        st.markdown("---")
                        st.subheader("🤖 AI-Generated Assets")
                        
                        col_ai1, col_ai2 = st.columns(2)
                        
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
        st.download_button(
            label="📥 Download Latest AI Assets (CSV)",
            data=st.session_state.ai_generated_csv,
            file_name=f"ai_generated_assets_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the latest asset CSV generated by the AI assistant",
        )


# Add CSV download section in AI area
st.markdown("---")
st.subheader("📥 Export Results")

col1, col2 = st.columns(2)

with col1:
    # Download current simulation results
    results_df = filtered_df[display_cols].rename(columns={
        "asset_name": "Asset", "risk_level": "Risk Level", "vulnerability": "Vulnerability / CVE",
        "vuln_prob": "Vulnerability Probability", "mean_lef": "Mean Loss Events/Yr",
        "eal": "EAL ($)", "var_90": "90% VaR ($)", "var_95": "95% VaR ($)", "var_99": "99% VaR ($)",
        "std_dev": "Std Dev ($)", "contribution_pct": "% of Portfolio EAL",
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
        "Metric": "Portfolio EAL ($)",
        "Value": f"${portfolio_summary['eal']:,.0f}"
    }, {
        "Metric": "90% VaR ($)",
        "Value": f"${portfolio_summary['var_90']:,.0f}"
    }, {
        "Metric": "95% VaR ($)",
        "Value": f"${portfolio_summary['var_95']:,.0f}"
    }, {
        "Metric": "99% VaR ($)",
        "Value": f"${portfolio_summary['var_99']:,.0f}"
    }, {
        "Metric": "Std Dev ($)",
        "Value": f"${portfolio_summary['std_dev']:,.0f}"
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