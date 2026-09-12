#!/usr/bin/env python3
"""Generate tests/fixtures/quanthockey_goalie_sample.xlsx.

There is no real QuantHockey goalie-stats export in data/ yet (that's
tracked as yahoo_fantasy_bot-nr7, blocking end-to-end validation of the
yahoo_fantasy_bot-fow --goalie-input path). This hand-built fixture follows
the same shape a real QuantHockey goalie export would have -- a two-row
header ('info' sheet + 'QuantHockey' sheet with a banner row above the real
column header row) -- with realistic W/GA/SV/SO values for the four goalies
already present in tests/fixtures/quanthockey_sample.xlsx (see
tests/fixtures/make_fixture.py), plus one goalie NOT present in the skater
sample, to exercise the "matched vs unmatched" paths of --goalie-input.

Run from the repo root:  python tests/fixtures/make_goalie_fixture.py
"""
from pathlib import Path

import pandas as pd

DEST = Path("tests/fixtures/quanthockey_goalie_sample.xlsx")
SHEET = "QuantHockey"

# Matches the four goalies in quanthockey_sample.xlsx (Andrei Vasilevski,
# Connor Hellebuyck, Samuel Montembeault, Kaapo Kähkönen), plus one goalie
# ("Extra Goalie") who does NOT appear in the skater sample, to prove
# unmatched rows in goalie_stats_df are simply ignored.
ROWS = [
    {"Name": "Andrei Vasilevski", "Team": "TBL", "Pos": "G", "GP": 63, "W": 38, "GA": 165, "SV": 1750, "SO": 3},
    {"Name": "Connor Hellebuyck", "Team": "WPG", "Pos": "G", "GP": 63, "W": 42, "GA": 140, "SV": 1800, "SO": 5},
    {"Name": "Samuel Montembeault", "Team": "MTL", "Pos": "G", "GP": 62, "W": 30, "GA": 180, "SV": 1650, "SO": 2},
    {"Name": "Kaapo Kähkönen", "Team": "COL", "Pos": "G", "GP": 1, "W": 1, "GA": 2, "SV": 25, "SO": 0},
    {"Name": "Extra Goalie", "Team": "NYR", "Pos": "G", "GP": 20, "W": 10, "GA": 45, "SV": 500, "SO": 1},
]


def main() -> int:
    df = pd.DataFrame(ROWS)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(DEST) as writer:
        pd.DataFrame(
            [["Source:", "hand-built (no real QuantHockey goalie export yet -- yahoo_fantasy_bot-nr7)"],
             ["Note:", "Fixture for yahoo_fantasy_bot-fow --goalie-input testing"]]
        ).to_excel(writer, sheet_name="info", index=False, header=False)
        df.to_excel(writer, sheet_name=SHEET, index=False, startrow=1)
    print(f"Wrote {DEST} ({len(df)} rows, {len(df.columns)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
