import pandas as pd

from yahoo_fantasy_bot import scoring


def test_score_row_and_dataframe():
    # Create synthetic rows with clear stats
    rows = [
        {
            "Player": "A",
            "G": 30,
            "A": 40,
            "PlusMinus": 10,
            "PPG": 5,
            "PPA": 8,
            "SHG": 0,
            "SHA": 0,
            "ShotAttempts": 200,
            "PIM": 20,
            "BLK": 30,
            "GP": 82,
        },
        {
            "Player": "B",
            "G": 10,
            "A": 12,
            "PlusMinus": -5,
            "PPG": 1,
            "PPA": 2,
            "SHG": 0,
            "SHA": 0,
            "ShotAttempts": 80,
            "PIM": 10,
            "BLK": 5,
            "GP": 40,
        },
        {
            "Player": "C",
            "G": 0,
            "A": 0,
            "PlusMinus": 0,
            "PPG": 0,
            "PPA": 0,
            "SHG": 0,
            "SHA": 0,
            "ShotAttempts": 10,
            "PIM": 0,
            "BLK": 0,
            "GP": 0,
        },
    ]

    df = pd.DataFrame(rows)
    out = scoring.score_dataframe(df, k=20.0, projected_games=82)

    # Check raw_score for player A: compute expected manually
    expected_A = (
        6 * 30
        + 4 * 40
        + 2 * 10
        + 2 * 5
        + 2 * 8
        + 2 * 0
        + 2 * 0
        + 0.6 * 200
        - 1 * 20
        + 1 * 30
    )
    assert abs(out.loc[0, "raw_score"] - expected_A) < 1e-6

    # Player B raw score sanity
    expected_B = (
        6 * 10 + 4 * 12 + 2 * -5 + 2 * 1 + 2 * 2 + 0.6 * 80 - 1 * 10 + 1 * 5
    )
    assert abs(out.loc[1, "raw_score"] - expected_B) < 1e-6

    # per_game should be raw_score / gp when gp>0
    assert abs(out.loc[0, "per_game"] - (out.loc[0, "raw_score"] / 82)) < 1e-9
    assert abs(out.loc[1, "per_game"] - (out.loc[1, "raw_score"] / 40)) < 1e-9

    # Player C has gp==0 so per_game=0 and shrunk_per_game==league_mean
    assert out.loc[2, "per_game"] == 0.0
    assert out.loc[2, "shrunk_per_game"] == out.loc[0, "league_mean_per_game"]

    # ranking_score should equal projected_total
    assert "ranking_score" in out.columns
    assert abs(out.loc[0, "ranking_score"] - out.loc[0, "projected_total"]) < 1e-9
