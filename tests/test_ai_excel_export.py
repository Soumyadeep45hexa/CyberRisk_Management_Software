import pandas as pd

from src.excel_download import ai_assets_to_excel_bytes


def test_ai_assets_to_excel_bytes_looks_like_xlsx():
    df = pd.DataFrame([
        {"asset_name": "Example Asset", "threat_event": "Phishing", "vulnerability": "Weak access controls"}
    ])

    payload = ai_assets_to_excel_bytes(df)

    assert payload.startswith(b"PK")
    assert payload.endswith(b"\n") is False
