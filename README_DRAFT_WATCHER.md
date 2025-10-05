Draft watcher
=============

A small utility to watch a Yahoo fantasy league draft and notify you when picks are made.

Usage
-----

Run from the repo root (use the virtualenv python):

```bash
./env/bin/python scripts/draft_watcher.py --league-id nhl.l.2354566 --oauth-file oauth2.json --interval 10 --use-gui --resolve-names
```

Example: run ranking CLI
-----------------------

You can run the ranking CLI from the same repo to compute scores and export a CSV. Example with per-file weight normalization:

```bash
./env/bin/python scripts/rank_players.py --decay 0.5 --normalize-file-weights --k 20 --projected-games 82 --out ranked_players.csv
```

Options
- `--league-id`: (required) league id, e.g. `nhl.l.2354566`
- `--oauth-file`: (required) path to your `oauth2.json` credentials file
- `--interval`: polling interval in seconds (default 15)
- `--use-gui`: enable desktop notifications via `notify-send` (Linux)
- `--resolve-names`: batch-resolve player IDs to names using Yahoo API and a local cache

Behavior
- Writes a CSV log into `logs/draft-<league-id>.csv` with columns: timestamp,pick,round,team_key,player_id,player_name
- When `--resolve-names` is used, player details are cached under `.cache/player_details-<league-id>.pkl`

Notes
- If `notify-send` is not available, GUI notifications are skipped silently.
- If Yahoo API access is denied (HTTP 999 errors), the script will print the error and continue polling.
