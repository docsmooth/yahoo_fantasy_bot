import pandas as pd
import tempfile
import os

from yahoo_fantasy_bot import scoring


def test_shrinkage_k_effect():
    # Two players: one with many games, one with few. Increase k should pull small-sample player toward league mean.
    rows = [
        {"Name": "Full", "G": 20, "A": 20, "ShotAttempts": 200, "GP": 82},
        {"Name": "Small", "G": 1, "A": 0, "ShotAttempts": 5, "GP": 1},
    ]
    df = pd.DataFrame(rows)
    out_k1 = scoring.score_dataframe(df, k=1.0, projected_games=82)
    out_k100 = scoring.score_dataframe(df, k=100.0, projected_games=82)

    # small player's shrunk_per_game should be closer to league_mean when k is larger
    small_k1 = out_k1.loc[out_k1['Name'] == 'Small', 'shrunk_per_game'].iloc[0]
    small_k100 = out_k100.loc[out_k100['Name'] == 'Small', 'shrunk_per_game'].iloc[0]
    league_mean = out_k1.loc[0, 'league_mean_per_game']

    assert abs(small_k100 - league_mean) < abs(small_k1 - league_mean)


def test_normalize_file_weights():
    # Create two small dataframes mimicking two files; first file player has gp large, second file small
    rows_new = [
        {"Name": "P1", "G": 10, "A": 10, "ShotAttempts": 100, "GP": 82},
        {"Name": "P2", "G": 5, "A": 5, "ShotAttempts": 50, "GP": 82},
    ]
    rows_old = [
        {"Name": "P1", "G": 20, "A": 20, "ShotAttempts": 200, "GP": 82},
        {"Name": "P2", "G": 0, "A": 0, "ShotAttempts": 0, "GP": 0},
    ]
    # write temp excel files
    tmp1 = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    tmp2 = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    pd.DataFrame(rows_new).to_excel(tmp1.name, sheet_name='QuantHockey', index=False, header=True)
    pd.DataFrame(rows_old).to_excel(tmp2.name, sheet_name='QuantHockey', index=False, header=True)

    merged = scoring.score_multiple_files([tmp1.name, tmp2.name], sheet_name='QuantHockey', header=0, decay=0.5, weight_by_games=True, normalize_file_weights=False)
    merged_norm = scoring.score_multiple_files([tmp1.name, tmp2.name], sheet_name='QuantHockey', header=0, decay=0.5, weight_by_games=True, normalize_file_weights=True)

    # With normalization, P2 should not be dominated by older file where gp==0
    # combined_shrunk_per_game should be computed without inf/nan and be finite
    assert 'combined_shrunk_per_game' in merged.columns
    assert 'combined_shrunk_per_game' in merged_norm.columns
    assert merged['combined_shrunk_per_game'].notna().all()
    assert merged_norm['combined_shrunk_per_game'].notna().all()

    # cleanup
    os.unlink(tmp1.name)
    os.unlink(tmp2.name)
