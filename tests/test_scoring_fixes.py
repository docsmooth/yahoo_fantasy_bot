"""Regression tests for the yahoo_fantasy_bot/scoring.py bug fixes below.

Each test in this file is written to FAIL against the pre-fix code (i.e.
against the behavior pinned in tests/test_scoring_characterization.py before
it was updated). See that file for the "this is what used to happen"
counterparts.
"""
from pathlib import Path

import pandas as pd
import pytest

from yahoo_fantasy_bot import scoring

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_DATA_2024_25 = REPO_ROOT / "data" / "QuantHockey_2024-2025.xlsx"


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-t0p: rank on shrunk_per_game, not raw per_game
# ---------------------------------------------------------------------------

def test_ranking_score_derives_from_shrunk_per_game_t0p(quanthockey_sample_df):
    """ranking_score/projected_total must be shrunk_per_game-derived. Against
    the pre-fix code, projected_total == per_game * projected_games, which
    would fail this equality (per_game != shrunk_per_game for Justin Robidas,
    a 2-GP call-up in this fixture).
    """
    out = scoring.score_dataframe(quanthockey_sample_df, k=20.0, projected_games=82)

    assert (out["projected_total"] == out["shrunk_per_game"] * 82).all()
    assert (out["ranking_score"] == out["projected_total"]).all()

    robidas = out.loc[out["Name"] == "Justin Robidas"].iloc[0]
    assert robidas["per_game"] != pytest.approx(robidas["shrunk_per_game"])
    # projected_total must NOT equal the raw (unshrunk) projection for a
    # player whose raw and shrunk per-game rates differ.
    assert robidas["projected_total"] != pytest.approx(robidas["per_game"] * 82)


def test_per_game_projection_column_preserves_raw_view_t0p(quanthockey_sample_df):
    """The raw (unshrunk) per-game projection must still be available under
    a new name, per_game_projection, so nothing is lost by ranking on the
    shrunk estimate.
    """
    out = scoring.score_dataframe(quanthockey_sample_df, k=20.0, projected_games=82)
    assert "per_game_projection" in out.columns
    assert (out["per_game_projection"] == out["per_game"] * 82).all()


@pytest.mark.skipif(not REAL_DATA_2024_25.exists(), reason="data/QuantHockey_2024-2025.xlsx not present")
def test_ranking_score_real_data_parekh_vs_draisaitl_t0p():
    """The original bug report: on the real 2024-25 export, Zayne Parekh (1
    GP) used to rank #1 overall (ranking_score ~918) ahead of Leon Draisaitl
    (71 GP, ranking_score ~887). After the fix, full-season stars top the
    list and 1-GP call-ups fall well down.
    """
    df = pd.read_excel(REAL_DATA_2024_25, sheet_name="QuantHockey", header=1)
    out = scoring.score_dataframe(df, k=20.0, projected_games=82)
    ranked = out.sort_values("ranking_score", ascending=False).reset_index(drop=True)

    top3_names = set(ranked.iloc[:3]["Name"])
    assert top3_names == {"Leon Draisaitl", "Nathan MacKinnon", "Nikita Kucherov"}

    parekh_rank = ranked.index[ranked["Name"] == "Zayne Parekh"][0]
    draisaitl_rank = ranked.index[ranked["Name"] == "Leon Draisaitl"][0]
    assert draisaitl_rank == 0
    # Parekh (1 GP) must fall well down the list, not anywhere near the top.
    assert parekh_rank > 100
    assert parekh_rank > draisaitl_rank


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-chg: league_mean_per_game excludes goalies
# ---------------------------------------------------------------------------

def test_league_mean_excludes_goalies_chg():
    """A synthetic mixed skater/goalie frame: two skaters with per_game 10
    and 20 (mean 15) and two goalie rows (detected via Pos == 'G', no
    goalie stats) that would contribute per_game == 0.0 each if included.
    Pre-fix, league_mean_per_game would be dragged down to 7.5 (average of
    [10, 20, 0, 0]); post-fix it must equal the skater-only mean of 15.0.
    """
    rows = [
        {"Name": "Skater A", "Pos": "F", "G": 10, "A": 0, "GP": 10},   # raw=60, per_game=6... use weights below instead
        {"Name": "Skater B", "Pos": "F", "G": 20, "A": 0, "GP": 10},
        {"Name": "Goalie A", "Pos": "G", "GP": 10},
        {"Name": "Goalie B", "Pos": "G", "GP": 10},
    ]
    df = pd.DataFrame(rows)
    # Use G-only weights so per_game is easy to hand-verify: raw = 6*G.
    out = scoring.score_dataframe(df, k=20.0, projected_games=82, weights={"A": 0.0})

    skater_a = out.loc[out["Name"] == "Skater A"].iloc[0]
    skater_b = out.loc[out["Name"] == "Skater B"].iloc[0]
    assert skater_a["per_game"] == pytest.approx(6.0)   # 60/10
    assert skater_b["per_game"] == pytest.approx(12.0)  # 120/10

    league_mean = out["league_mean_per_game"].iloc[0]
    assert league_mean == pytest.approx((6.0 + 12.0) / 2)  # skater-only mean == 9.0
    assert league_mean != pytest.approx((6.0 + 12.0 + 0.0 + 0.0) / 4)  # NOT the goalie-contaminated 4.5


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-zop: BS resolves via the BLK alias
# ---------------------------------------------------------------------------

def test_bs_column_scores_as_blocks_zop():
    """A row using the real QuantHockey 'BS' column name for blocked shots
    must contribute BLK weight * BS to raw_score. Pre-fix, 'BS' was not in
    _COLUMN_ALIASES['BLK'] at all, so this contributed 0.
    """
    row = pd.Series({"G": 0, "A": 0, "BS": 10})
    score = scoring.score_row(row)
    assert score == pytest.approx(scoring.DEFAULT_WEIGHTS["BLK"] * 10)


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-gl7: Team/Pos carried through score_multiple_files
# ---------------------------------------------------------------------------

def test_score_multiple_files_carries_team_and_pos_gl7(tmp_path, quanthockey_sample_df):
    """Team/Pos must survive score_multiple_files' merge. Pre-fix, only
    [Name, gp, shrunk_per_game, projected_total, raw_score] (suffixed) plus
    combined_* columns were kept -- Team/Pos were silently dropped.
    """
    f0 = tmp_path / "season.xlsx"
    with pd.ExcelWriter(f0) as w:
        quanthockey_sample_df.to_excel(w, sheet_name="QuantHockey", index=False, startrow=1)

    merged = scoring.score_multiple_files([str(f0)], sheet_name="QuantHockey", header=1)
    assert "Team" in merged.columns
    assert "Pos" in merged.columns
    kucherov = merged.loc[merged["Name"] == "Nikita Kucherov"].iloc[0]
    assert kucherov["Team"] == "TBL"
    assert kucherov["Pos"] == "F"


def test_rank_players_csv_has_team_pos_and_per_file_scores_gl7(tmp_path):
    """End-to-end smoke test of scripts/rank_players.py's own column
    selection (not just score_multiple_files): the emitted CSV must contain
    Name/Team/Pos plus per-file suffixed score columns, not just Name plus
    three duplicated combined_* columns.
    """
    import subprocess
    import sys

    df = pd.DataFrame([
        {"Name": "Test Player A", "Team": "TOR", "Pos": "C", "GP": 10, "G": 2, "A": 3, "SHOTS": 15},
        {"Name": "Test Goalie", "Team": "TOR", "Pos": "G", "GP": 8, "G": 0, "A": 0},
    ])
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "QuantHockey_test.xlsx"
    with pd.ExcelWriter(file_path) as w:
        df.to_excel(w, sheet_name="QuantHockey", index=False, startrow=1)

    out_csv = tmp_path / "ranked_players.csv"
    rank_players = REPO_ROOT / "scripts" / "rank_players.py"
    p = subprocess.run(
        [sys.executable, str(rank_players), "--input", str(file_path), "--out", str(out_csv)],
        capture_output=True,
    )
    assert p.returncode == 0, p.stderr.decode()
    out_df = pd.read_csv(out_csv)
    assert "Team" in out_df.columns
    assert "Pos" in out_df.columns
    assert "raw_score_f0" in out_df.columns
    assert "gp_f0" in out_df.columns
    # not just Name + duplicated combined_* columns
    assert set(out_df.columns) != {"Name", "combined_shrunk_per_game", "combined_projected_total", "combined_ranking_score"}


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-6e6: goalie_mask wired to the real Pos column
# ---------------------------------------------------------------------------

def test_goalie_weight_not_multiplied_by_gp_6e6(tmp_path):
    """When weight_by_games=True, a goalie's per-file weight must be the
    plain decay factor (no GP multiplier), while a skater's weight is
    decay*gp. Pre-fix, goalie_mask could never be True (it looked for a
     'position' column that never existed in the merged frame), so every
    goalie's weight WAS multiplied by GP just like a skater's.

    Uses real in-sheet W/GA/SV/SO stats (differing per file) so the
    goalie's shrunk_per_game genuinely differs between the two files --
    with identical values in each file (e.g. an all-fabricated fallback),
    every weighting scheme produces the same weighted average and the test
    can't discriminate old vs. new behavior.
    """
    newest = pd.DataFrame([{
        "Name": "Some Goalie", "Pos": "G", "GP": 60,
        "W": 30, "GA": 150, "SV": 1600, "SO": 2,
    }])
    older = pd.DataFrame([{
        "Name": "Some Goalie", "Pos": "G", "GP": 5,
        "W": 2, "GA": 15, "SV": 140, "SO": 0,
    }])
    f0 = tmp_path / "season.xlsx"
    f1 = tmp_path / "season_older.xlsx"
    with pd.ExcelWriter(f0) as w:
        newest.to_excel(w, sheet_name="QuantHockey", index=False, startrow=1)
    with pd.ExcelWriter(f1) as w:
        older.to_excel(w, sheet_name="QuantHockey", index=False, startrow=1)

    # goalie_method='stats' makes the OLD code compute a genuinely
    # stats-derived (and differing) per_game for each file too, so this
    # test isolates the goalie_mask/weighting fix specifically rather than
    # conflating it with yahoo_fantasy_bot-fow's separate "use real goalie
    # stats regardless of goalie_method" behavior.
    merged = scoring.score_multiple_files(
        [str(f0), str(f1)], sheet_name="QuantHockey", header=1, decay=0.5,
        weight_by_games=True, goalie_method="stats",
    )
    goalie = merged.iloc[0]
    assert goalie["shrunk_per_game_f0"] != pytest.approx(goalie["shrunk_per_game_f1"])

    # weight_f0 = 0.5**0 = 1 (NOT 1*60), weight_f1 = 0.5**1 = 0.5 (NOT 0.5*5)
    expected_fixed = (
        goalie["shrunk_per_game_f0"] * 1.0 + goalie["shrunk_per_game_f1"] * 0.5
    ) / 1.5
    expected_buggy_gp_weighted = (
        goalie["shrunk_per_game_f0"] * 60 + goalie["shrunk_per_game_f1"] * 2.5
    ) / 62.5
    assert expected_fixed != pytest.approx(expected_buggy_gp_weighted)
    assert goalie["combined_shrunk_per_game"] == pytest.approx(expected_fixed)


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-w2u: score_multiple_files([]) raises a clear error
# ---------------------------------------------------------------------------

def test_score_multiple_files_empty_raises_valueerror_w2u():
    """Pre-fix this raised AttributeError: 'NoneType' object has no
    attribute 'columns'. Must now be a clear ValueError naming the problem.
    """
    with pytest.raises(ValueError, match="no file_paths given"):
        scoring.score_multiple_files([])


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-fow: score goalies from a real goalie-stats export
# ---------------------------------------------------------------------------

def test_goalie_input_scores_matched_goalies_for_real_fow(
    quanthockey_sample_df, quanthockey_goalie_sample_df
):
    """With a matching --goalie-input-style goalie_stats_df, goalie rows
    get a REAL raw_score (DEFAULT_GOALIE_WEIGHTS applied to W/GA/SV/SO),
    not 0.0, and are NOT flagged as fabricated.
    """
    out = scoring.score_dataframe(
        quanthockey_sample_df, goalie_stats_df=quanthockey_goalie_sample_df,
    )
    goalies = out.loc[out["Pos"] == "G"].set_index("Name")
    assert len(goalies) == 4

    vasilevski = goalies.loc["Andrei Vasilevski"]
    expected_raw = (
        scoring.DEFAULT_GOALIE_WEIGHTS["W"] * 38
        + scoring.DEFAULT_GOALIE_WEIGHTS["GA"] * 165
        + scoring.DEFAULT_GOALIE_WEIGHTS["SV"] * 1750
        + scoring.DEFAULT_GOALIE_WEIGHTS["SO"] * 3
    )
    assert vasilevski["raw_score"] == pytest.approx(expected_raw)
    assert vasilevski["raw_score"] != 0.0
    assert not vasilevski["goalie_stats_fabricated"]

    # per_game should be derived from the real raw_score / gp (gp from the
    # main sheet, GP=63 for Vasilevski), not a fabricated GP-based constant.
    assert vasilevski["per_game"] == pytest.approx(expected_raw / 63)
    assert vasilevski["shrunk_per_game"] == pytest.approx(vasilevski["per_game"])  # goalies exempt from shrink


def test_goalie_input_unmatched_goalie_still_fabricated_and_warns_fow(
    capsys, quanthockey_sample_df, quanthockey_goalie_sample_df
):
    """A goalie present in the skater sheet but absent from goalie_stats_df
    must still fall back to the fabricated GP-based estimate AND must be
    flagged via goalie_stats_fabricated -- a fabricated number must never
    reach a report unmarked.
    """
    goalie_stats_missing_one = quanthockey_goalie_sample_df[
        quanthockey_goalie_sample_df["Name"] != "Kaapo Kähkönen"
    ]
    out = scoring.score_dataframe(
        quanthockey_sample_df, goalie_stats_df=goalie_stats_missing_one,
        source_name="test-source.xlsx",
    )
    goalies = out.loc[out["Pos"] == "G"].set_index("Name")

    kahkonen = goalies.loc["Kaapo Kähkönen"]
    assert kahkonen["raw_score"] == 0.0
    assert kahkonen["goalie_stats_fabricated"]

    vasilevski = goalies.loc["Andrei Vasilevski"]
    assert not vasilevski["goalie_stats_fabricated"]

    captured = capsys.readouterr()
    assert "test-source.xlsx" in captured.err
    assert "1 goalie row" in captured.err


def test_no_goalie_input_warns_loudly_with_file_and_count_fow(capsys, quanthockey_sample_df):
    """When goalie rows are present and NO --goalie-input/goalie_stats_df is
    supplied at all, scoring must warn loudly on stderr naming the source
    and the count of affected goalies (all 4 in this fixture) -- not
    silently fabricate.
    """
    out = scoring.score_dataframe(quanthockey_sample_df, source_name="my_export.xlsx")
    goalies = out.loc[out["Pos"] == "G"]
    assert goalies["goalie_stats_fabricated"].all()

    captured = capsys.readouterr()
    assert "my_export.xlsx" in captured.err
    assert "4 goalie row" in captured.err
