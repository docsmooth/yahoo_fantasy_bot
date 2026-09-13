"""Goalies must be shrunk toward a goalie league mean (yahoo_fantasy_bot-bq3).

Goalies were exempt from shrinkage, which reproduced yahoo_fantasy_bot-t0p on
the goalie side: a single strong start projected over a full season. Against
the real 2024-2025 goalie export, Erik Portillo (1 GP, 1 W, 1 GA, 28 SV) scored
raw 18.8 -> per_game 18.8 -> projected 1541.6, ranking first overall ahead of
every skater.
"""
import pandas as pd
import pytest

from yahoo_fantasy_bot import scoring


def _goalie_frame(rows):
    """Build a goalie-export-shaped frame (real 18-column schema, no Pos)."""
    return pd.DataFrame(rows, columns=[
        "Rk", "Name", "Team", "Age", "GP", "GAA", "SV%", "W", "L", "GA",
        "SV", "SOG", "SO", "TIME", "G", "A", "P", "PIM",
    ])


@pytest.fixture
def one_game_wonder_vs_starters():
    # Portillo's real 2024-2025 line, against three full-season starters.
    return _goalie_frame([
        [1, "Starter A",     "AAA", 30, 60, 2.20, .920, 35, 20, 130, 1500, 1630, 5, 3600, 0, 1, 1, 2],
        [2, "Starter B",     "BBB", 28, 55, 2.50, .915, 30, 20, 140, 1400, 1540, 3, 3300, 0, 0, 0, 0],
        [3, "Starter C",     "CCC", 26, 50, 2.70, .910, 25, 20, 135, 1300, 1435, 2, 3000, 0, 0, 0, 2],
        [4, "Erik Portillo", "DDD", 24,  1, 1.00, .966,  1,  0,   1,   28,   29, 0,   60, 0, 0, 0, 0],
    ])


def test_one_game_goalie_does_not_outrank_starters(one_game_wonder_vs_starters):
    out = scoring.score_dataframe(one_game_wonder_vs_starters).set_index("Name")

    portillo = out.loc["Erik Portillo"]
    # His raw rate really is the best in the frame -- that is the trap.
    assert portillo["per_game"] == pytest.approx(18.8)
    assert portillo["per_game"] > out.loc["Starter A", "per_game"]

    # But one game must not survive shrinkage as the top projection.
    ranked = out.sort_values("ranking_score", ascending=False)
    assert ranked.index[0] != "Erik Portillo", (
        "a 1-GP goalie ranked first; goalie shrinkage is not being applied")
    assert portillo["shrunk_per_game"] < portillo["per_game"]


def test_one_game_goalie_pulled_near_the_goalie_mean(one_game_wonder_vs_starters):
    out = scoring.score_dataframe(one_game_wonder_vs_starters).set_index("Name")
    portillo = out.loc["Erik Portillo"]
    goalie_mean = portillo["goalie_league_mean_per_game"]

    # With gp=1 and k=20 the prior dominates: the estimate should sit far
    # closer to the goalie mean than to his own one-game rate.
    assert abs(portillo["shrunk_per_game"] - goalie_mean) < \
           abs(portillo["shrunk_per_game"] - portillo["per_game"])


def test_goalies_shrink_toward_goalie_mean_not_skater_mean():
    """Goalie per-game scale is several times the skater scale. Shrinking
    goalies toward the skater mean would crush every goalie."""
    skaters = pd.DataFrame([
        {"Name": "Skater A", "Pos": "F", "GP": 80, "G": 40, "A": 50, "SOG": 250, "PIM": 20, "+/-": 10, "BS": 30},
        {"Name": "Skater B", "Pos": "F", "GP": 78, "G": 30, "A": 40, "SOG": 200, "PIM": 25, "+/-": 5,  "BS": 25},
    ])
    goalies = _goalie_frame([
        [1, "Starter A", "AAA", 30, 60, 2.2, .920, 35, 20, 130, 1500, 1630, 5, 3600, 0, 1, 1, 2],
        [2, "Backup B",  "BBB", 24,  5, 3.0, .890,  2,  3,  15,  120,  135, 0,  300, 0, 0, 0, 0],
    ])
    # Give the skater frame goalie rows to match on, as the real pipeline does.
    skaters = pd.concat([skaters, pd.DataFrame([
        {"Name": "Starter A", "Pos": "G", "GP": 60},
        {"Name": "Backup B",  "Pos": "G", "GP": 5},
    ])], ignore_index=True)

    out = scoring.score_dataframe(skaters, goalie_stats_df=goalies).set_index("Name")
    skater_mean = out["league_mean_per_game"].iloc[0]
    goalie_mean = out["goalie_league_mean_per_game"].iloc[0]

    assert goalie_mean != pytest.approx(skater_mean)
    # The thin-sample goalie is pulled toward the GOALIE mean, so it must stay
    # on the goalie scale rather than collapsing onto the skater mean.
    backup = out.loc["Backup B", "shrunk_per_game"]
    assert abs(backup - goalie_mean) < abs(backup - skater_mean)
