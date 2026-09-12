#!/usr/bin/env python3
"""Regenerate tests/fixtures/quanthockey_sample.xlsx from the real data/ export.

The fixture deliberately reproduces the REAL QuantHockey schema (50 columns,
two-row header, 'info' + 'QuantHockey' sheets) rather than a hand-written
subset. Tests written against invented column names such as 'PlusMinus' or
'ShotAttempts' are what allowed the BS/BLK alias bug to survive -- see
yahoo_fantasy_bot-zop.

Rows are chosen to cover the cases that matter for scoring:
  - elite forwards and defencemen with a full season of games
  - mid-tier forwards
  - 2-GP call-ups, which expose whether shrinkage is actually applied
  - goalies with a wide games-played spread, including a 1-GP backup

Run from the repo root:  python tests/fixtures/make_fixture.py
"""
from pathlib import Path

import pandas as pd

SOURCE = Path("data/QuantHockey_2024-2025.xlsx")
DEST = Path("tests/fixtures/quanthockey_sample.xlsx")
SHEET = "QuantHockey"
HEADER_ROW = 1


def build(src: pd.DataFrame) -> pd.DataFrame:
    groups = [
        src[src["Pos"] == "F"].nlargest(2, "P").index.tolist(),
        src[(src["Pos"] == "F") & (src["GP"].between(60, 82))].iloc[40:42].index.tolist(),
        src[src["Pos"] == "D"].nlargest(2, "P").index.tolist(),
        src[(src["Pos"] != "G") & (src["GP"] <= 2)].head(2).index.tolist(),
        src[src["Pos"] == "G"].nlargest(3, "GP").index.tolist(),
        src[src["Pos"] == "G"].nsmallest(1, "GP").index.tolist(),
    ]
    idx = [i for grp in groups for i in grp]
    out = src.loc[idx].reset_index(drop=True)
    out["Rk"] = range(1, len(out) + 1)
    return out


def main() -> int:
    if not SOURCE.exists():
        print(f"Source export not found: {SOURCE}")
        return 1
    src = pd.read_excel(SOURCE, sheet_name=SHEET, header=HEADER_ROW)
    fixture = build(src)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(DEST) as writer:
        # Mirror the real file: an 'info' sheet, then the data sheet with the
        # real two-row header layout (one banner row above the column names).
        pd.DataFrame(
            [["Source:", "QuantHockey.com"], ["Note:", "Trimmed test fixture"]]
        ).to_excel(writer, sheet_name="info", index=False, header=False)
        fixture.to_excel(writer, sheet_name=SHEET, index=False, startrow=1)
    print(f"Wrote {DEST} ({len(fixture)} rows, {len(fixture.columns)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
