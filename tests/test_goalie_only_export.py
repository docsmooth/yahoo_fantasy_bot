"""Regression tests for yahoo_fantasy_bot-vlm.

The QuantHockey skater export is a top-1000-by-points list. Goalies score
~0-2 "skater" points, so most goalies fall outside that list in any given
season -- on the real 2025-2026 data, the skater sheet contains only 69 of
the 98 rows in the matching goalie export. The 29 missing goalies (e.g.
Connor Hellebuyck, Igor Shesterkin, Jeremy Swayman, Dustin Wolf) were scored
correctly (their combined_ranking_score came out right, since the real
goalie stats are matched in via --goalie-input/goalie_stats_df regardless of
which skater file a row lives in) but Name/Team/Pos are carried through the
merge from the SKATER frame(s) only -- specifically Team/Pos are carried
through from the newest (idx==0) skater file only (yahoo_fantasy_bot-gl7),
so any goalie missing from THAT particular file came out with Team=NaN,
Pos=NaN, invisible to a `Pos == 'G'` filter, and gp_f0/raw_score_f0 stuck at
0.0 (the skater-file columns, never populated for a row that was never a
skater-file row at all).

Fixtures are hand-built per the make_fixture.py/test_goalie_multiseason.py
convention: real column names, header row 1, an 'info' banner sheet plus a
'QuantHockey' data sheet.
"""
from pathlib import Path

import pandas as pd
import pytest

from yahoo_fantasy_bot import scoring


def _write_quanthockey_xlsx(path, rows):
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([["Source:", "test fixture"]]).to_excel(
            writer, sheet_name="info", index=False, header=False)
        df.to_excel(writer, sheet_name="QuantHockey", index=False, startrow=1)


def _goalie_row(**overrides):
    row = {
        "Name": "Test Goalie", "Team": "WPG", "Age": 30, "GP": 60,
        "GAA": 2.20, "SV%": 0.915, "W": 40, "L": 15, "GA": 130, "SV": 1500,
        "SOG": 1630, "SO": 5, "TIME": "3540:00", "G": 0, "A": 0, "P": 0, "PIM": 2,
    }
    row.update(overrides)
    return row


def _skater_row(**overrides):
    row = {
        "Name": "Test Forward", "Team": "TOR", "Pos": "C", "GP": 82,
        "G": 40, "A": 40, "P": 80, "PIM": 20, "+/-": 10, "SHOTS": 250, "BS": 10,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Case 1: a goalie present in the goalie export but in NO skater file at all.
# ---------------------------------------------------------------------------

def test_goalie_absent_from_every_skater_file_gets_a_real_row(tmp_path):
    """A goalie who never appears as a row in any --input skater file (the
    real-world 2-of-98 case: e.g. Arvid Soderblom, Jacob Fowler in the real
    2025-2026 data) must still show up in score_multiple_files' output, with
    Pos='G', a real Team, real goalie_gp/goalie_raw_score, and a non-zero
    combined_ranking_score -- not be silently dropped because it was never a
    row in any skater frame to begin with.

    Pre-fix: this player has NO row at all in the merged output (KeyError /
    empty lookup below), because goalie_stats_df is only ever left-merged
    onto rows that already exist in a skater file.
    """
    skater_file = tmp_path / "QuantHockey_2025-2026.xlsx"
    _write_quanthockey_xlsx(skater_file, [_skater_row()])

    goalie_file = tmp_path / "QuantHockey-Goalies_2025-2026.xlsx"
    _write_quanthockey_xlsx(goalie_file, [
        _goalie_row(Name="Ghost Goalie", Team="CHI", GP=26, W=8, GA=93, SV=681, SO=1),
        _goalie_row(Name="Other Starter", Team="BOS", GP=60, W=35, GA=140, SV=1600, SO=4),
    ])

    merged = scoring.score_multiple_files(
        [str(skater_file)], sheet_name="QuantHockey", header=1, decay=0.5,
        goalie_input=[str(goalie_file)],
    )

    assert "Ghost Goalie" in set(merged["Name"])
    row = merged.set_index("Name").loc["Ghost Goalie"]

    assert row["Pos"] == "G"
    assert row["Team"] == "CHI"
    assert row["goalie_gp"] == pytest.approx(26)
    expected_raw = scoring.DEFAULT_GOALIE_WEIGHTS["W"] * 8 + scoring.DEFAULT_GOALIE_WEIGHTS["GA"] * 93 \
        + scoring.DEFAULT_GOALIE_WEIGHTS["SV"] * 681 + scoring.DEFAULT_GOALIE_WEIGHTS["SO"] * 1
    assert row["goalie_raw_score"] == pytest.approx(expected_raw)
    assert row["combined_ranking_score"] > 0.0
    # Every per-file skater column must be zero-filled -- this player never
    # contributed to any skater file.
    assert row["gp_f0"] == 0.0
    assert row["raw_score_f0"] == 0.0


def test_pos_g_filter_returns_every_goalie_in_the_export(tmp_path):
    """Filtering the output board by Pos == 'G' must return every goalie in
    the goalie export, including ones with no skater-file row at all."""
    skater_file = tmp_path / "QuantHockey_2025-2026.xlsx"
    _write_quanthockey_xlsx(skater_file, [_skater_row()])

    goalie_file = tmp_path / "QuantHockey-Goalies_2025-2026.xlsx"
    goalie_names = ["Ghost Goalie A", "Ghost Goalie B", "Ghost Goalie C"]
    _write_quanthockey_xlsx(goalie_file, [
        _goalie_row(Name=n, GP=40 + i, W=20 + i, GA=100 + i, SV=1000 + i, SO=i)
        for i, n in enumerate(goalie_names)
    ])

    merged = scoring.score_multiple_files(
        [str(skater_file)], sheet_name="QuantHockey", header=1, decay=0.5,
        goalie_input=[str(goalie_file)],
    )

    g_names = set(merged.loc[merged["Pos"] == "G", "Name"])
    assert g_names == set(goalie_names)


# ---------------------------------------------------------------------------
# Case 2: a goalie present in an OLDER skater file, but absent from the
# NEWEST one (the real-world Hellebuyck/Shesterkin/Swayman/Wolf case).
# ---------------------------------------------------------------------------

def test_goalie_missing_from_newest_skater_file_keeps_team_pos_and_score(tmp_path):
    """A goalie who appears in an older skater file but NOT the newest one
    must still get Pos='G' and a real Team (Team/Pos are otherwise carried
    through from the newest file only), and -- critically -- its
    combined_ranking_score must equal the shrunk_per_game it actually earned
    from the one file it IS present in, not get diluted by a phantom
    zero-weighted contribution from the file it's absent from.

    This pins the exact regression this fix could introduce: naively
    trusting a Pos=='G' backfill to gate "don't multiply a goalie's weight
    by GP" (yahoo_fantasy_bot-6e6) -- without also checking the goalie
    actually has a row in that particular file -- hands the file this goalie
    is ABSENT from a nonzero flat weight (instead of the correct zero),
    dragging the combined score toward 0.
    """
    newest_skater = tmp_path / "QuantHockey_2025-2026.xlsx"
    _write_quanthockey_xlsx(newest_skater, [_skater_row(Name="Filler Skater")])

    older_skater = tmp_path / "QuantHockey_2024-2025.xlsx"
    _write_quanthockey_xlsx(older_skater, [
        _skater_row(Name="Filler Skater"),
        {"Name": "Old Timer Goalie", "Team": "WPG", "Pos": "G", "GP": 60,
         "G": 0, "A": 0, "PIM": 0},
    ])

    goalie_file = tmp_path / "QuantHockey-Goalies_2024-2025.xlsx"
    _write_quanthockey_xlsx(goalie_file, [
        _goalie_row(Name="Old Timer Goalie", Team="WPG", GP=60, W=47, GA=125, SV=1539, SO=8),
    ])

    merged = scoring.score_multiple_files(
        [str(newest_skater), str(older_skater)],
        sheet_name="QuantHockey", header=1, decay=0.5,
        goalie_input=[str(goalie_file)],
    )

    row = merged.set_index("Name").loc["Old Timer Goalie"]
    assert row["Pos"] == "G"
    assert row["Team"] == "WPG"
    # Absent from the newest file entirely.
    assert row["gp_f0"] == 0.0
    assert row["raw_score_f0"] == 0.0
    # Present (and real-stats-matched) in the older file.
    assert row["gp_f1"] == pytest.approx(60)
    assert row["raw_score_f1"] == pytest.approx(823.4)  # 5*47 - 3*125 + 0.6*1539 + 5*8

    # The only file this goalie actually contributed to is idx=1, so its
    # combined_shrunk_per_game must equal shrunk_per_game_f1 exactly -- the
    # (correctly zero-weighted) idx=0 contribution must not move the needle.
    assert row["combined_shrunk_per_game"] == pytest.approx(row["shrunk_per_game_f1"])
    assert row["combined_ranking_score"] > 400  # sanity: nowhere near the
    # ~1/3-diluted value the phantom-weight bug would produce


def test_goalie_gp_and_raw_score_columns_are_real_even_when_gp_f0_is_zero(tmp_path):
    """goalie_gp/goalie_raw_score must reflect the goalie's REAL stats
    (sourced from the goalie export) regardless of whether this player has a
    row in the newest skater file -- gp_f0/raw_score_f0 read 0.0 in exactly
    this scenario and must not be the only place GP/raw score are visible.
    """
    newest_skater = tmp_path / "QuantHockey_2025-2026.xlsx"
    _write_quanthockey_xlsx(newest_skater, [_skater_row()])

    older_skater = tmp_path / "QuantHockey_2024-2025.xlsx"
    _write_quanthockey_xlsx(older_skater, [
        _skater_row(),
        {"Name": "Vintage Goalie", "Team": "NYR", "Pos": "G", "GP": 55,
         "G": 0, "A": 0, "PIM": 0},
    ])

    goalie_file = tmp_path / "QuantHockey-Goalies_2024-2025.xlsx"
    _write_quanthockey_xlsx(goalie_file, [
        _goalie_row(Name="Vintage Goalie", Team="NYR", GP=55, W=30, GA=120, SV=1450, SO=6),
    ])

    merged = scoring.score_multiple_files(
        [str(newest_skater), str(older_skater)],
        sheet_name="QuantHockey", header=1, decay=0.5,
        goalie_input=[str(goalie_file)],
    )
    row = merged.set_index("Name").loc["Vintage Goalie"]

    assert row["gp_f0"] == 0.0
    assert row["raw_score_f0"] == 0.0
    assert row["goalie_gp"] == pytest.approx(55)
    expected_raw = 5 * 30 - 3 * 120 + 0.6 * 1450 + 5 * 6
    assert row["goalie_raw_score"] == pytest.approx(expected_raw)
