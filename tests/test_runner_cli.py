import subprocess
from pathlib import Path
import sys

def test_help_shows_usage():
    # Ensure the script returns 0 for --help
    p = subprocess.run([sys.executable, 'scripts/rank_players.py', '--help'], capture_output=True)
    assert p.returncode == 0
    assert b'Usage' in p.stdout or b'usage' in p.stdout.lower()


def test_smoke_run_creates_csv(tmp_path):
    # create a tiny synthetic Excel file with a QuantHockey-like sheet
    import pandas as pd
    df = pd.DataFrame([
        {'Name': 'Test Player A', 'Pos': 'C', 'GP': 10, 'G': 2, 'A': 3, 'SHOTS': 15},
        {'Name': 'Test Goalie', 'Pos': 'G', 'GP': 8, 'G': 0, 'A': 0},
    ])
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    file_path = data_dir / 'QuantHockey_test.xlsx'
    with pd.ExcelWriter(file_path) as w:
        df.to_excel(w, sheet_name='QuantHockey', index=False, startrow=1)

    # run the script pointing at the synthetic file
    out_csv = tmp_path / 'ranked_players.csv'
    p = subprocess.run([sys.executable, 'scripts/rank_players.py', '--input', str(file_path), '--out', str(out_csv)], capture_output=True)
    assert p.returncode == 0
    assert out_csv.exists()