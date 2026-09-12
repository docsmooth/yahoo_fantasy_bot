# Draft watcher

A small utility that polls a Yahoo! fantasy league draft and notifies you as
picks are made. Useful during a live draft, alongside the rankings produced by
`scripts/rank_players.py` (see the main [README](README.rst)).

## Usage

Run from the repo root:

```bash
python scripts/draft_watcher.py \
    --league-id nhl.l.XXXXXX \
    --oauth-file oauth2.json \
    --interval 10 \
    --use-gui \
    --resolve-names
```

## Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `--league-id` | yes | — | League id, e.g. `nhl.l.XXXXXX` |
| `--oauth-file` | yes | — | Path to your `oauth2.json` credentials |
| `--interval` | no | 15 | Polling interval in seconds |
| `--use-gui` | no | off | Desktop notifications via `notify-send` (Linux) |
| `--resolve-names` | no | off | Resolve player IDs to names via the Yahoo! API, with a local cache |
| `--since-pick` | no | 0 | Only report picks after this pick number |
| `--use-bot` | no | off | Instantiate the main `ManagerBot` and reuse its cache directory |
| `--config` | no | — | Path to `my.cfg`; required when `--use-bot` is given |

## Behavior

- Appends every new pick to `logs/draft-<league-id>.csv` with columns
  `timestamp,pick,round,team_key,player_id,player_name`.
- With `--resolve-names`, player details are cached at
  `.cache/player_details-<league-id>.pkl` to keep Yahoo! API calls down. Name
  resolution is batched (25 ids per request).
- Both paths are relative to the current directory, so run from the repo root.
- Press Ctrl-C to stop.

## Notes

- If `notify-send` is not installed, GUI notifications are skipped silently.
- If Yahoo! API access is denied (HTTP 999 rate-limit errors), the error is
  printed and polling continues.
- `--use-bot` falls back to a plain `League` connection if `ManagerBot` cannot be
  constructed; the failure is printed but is not fatal.

## Credentials

`oauth2.json` holds live Yahoo! credentials and is covered by `.gitignore`.
Never commit it. See `oauth2.json.example` for the file's shape.
