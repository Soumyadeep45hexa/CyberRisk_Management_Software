import io
import re
import csv
import pandas as pd

REQUIRED_COLUMNS = {"asset_name", "threat_event", "vulnerability"}


def _read_markdown_csv_candidates(answer: str):
    """Return every fenced csv block from an assistant response."""
    return re.findall(
        r"```(?:csv)?\s*(.*?)```",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )


def _try_dataframe_from_candidate(candidate: str):
    """Parse a csv string into a dataframe when its columns match.

    Accepts the normal Markdown code block format and the no-header row
    pattern shown by the chat answer in the screenshot.
    """
    try:
        parsed_df = pd.read_csv(io.StringIO(candidate.strip()))
    except Exception:
        try:
            parsed_rows = list(csv.reader(io.StringIO(candidate.strip())))
            if not parsed_rows:
                return None
            # If the answer is a headerless three-column list, convert it
            # into a dataframe using the canonical AI asset columns.
            if len(parsed_rows[0]) == 3 and parsed_rows[0][0].strip().lower() != "asset_name":
                records = []
                for row in parsed_rows:
                    if len(row) != 3:
                        continue
                    records.append({
                        "asset_name": row[0].strip(),
                        "threat_event": row[1].strip(),
                        "vulnerability": row[2].strip(),
                    })
                if records:
                    return pd.DataFrame(records)
            return None
        except Exception:
            return None

    parsed_df.columns = [str(column).strip() for column in parsed_df.columns]
    if REQUIRED_COLUMNS.issubset(parsed_df.columns):
        return parsed_df
    return None


def parse_ai_assets_to_dataframe(answer: str):
    """Parse an assistant answer into a valid AI asset dataframe.

    Supports:
    1. Markdown CSV fences.
    2. Plain header-line fallback using asset_name,threat_event,vulnerability.
    3. Headerless lines such as:
       Proprietary Source Code, Code exfiltration, Weak repo permissions
    """
    candidates = _read_markdown_csv_candidates(answer)
    if candidates:
        for candidate in candidates:
            parsed_df = _try_dataframe_from_candidate(candidate)
            if parsed_df is not None:
                return parsed_df

    # Header-line fallback
    lines = answer.splitlines()
    header_index = next(
        (idx for idx, line in enumerate(lines)
         if line.strip().lower().startswith("asset_name,")),
        None,
    )
    if header_index is not None:
        parsed_df = _try_dataframe_from_candidate("\n".join(lines[header_index:]))
        if parsed_df is not None:
            return parsed_df

    # Final fallback for bare lines without a header, matching the screenshot
    # response shape: asset_name, threat_event, vulnerability.
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if stripped.lower().startswith("asset_name,"):
            continue
        if stripped.count(",") < 2:
            continue
        try:
            row = next(csv.reader([stripped]))
        except Exception:
            continue
        if len(row) != 3:
            continue
        asset_name, threat_event, vulnerability = [piece.strip().strip('"\'') for piece in row]
        if not asset_name:
            continue
        rows.append({
            "asset_name": asset_name,
            "threat_event": threat_event,
            "vulnerability": vulnerability,
        })

    if rows:
        return pd.DataFrame(rows)

    return None


def ai_assets_to_excel_bytes(ai_assets_df):
    """Return AI-generated asset rows as an Excel workbook payload.

    The workbook must be byte-for-byte suitable for a Streamlit
    download_button with an Excel MIME type.
    """
    output = io.BytesIO()
    ai_assets_df.to_excel(output, index=False)
    return output.getvalue()
