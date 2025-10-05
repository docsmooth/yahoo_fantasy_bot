#!/usr/bin/env python3
"""Draft watcher: poll Yahoo draft_results and notify on new picks.

Usage: run with --league-id and --oauth-file. It will poll every --interval
seconds and print new picks to stdout. If --use-gui is set and `notify-send`
is available, it will also pop up desktop notifications on Linux.
"""
import argparse
import time
import subprocess
import sys
from datetime import datetime

try:
    from yahoo_oauth import OAuth2
    import yahoo_fantasy_api as yfa
except Exception as e:
    print("Missing dependency: yahoo_oauth and yahoo_fantasy_api are required.")
    raise


def notify_cli(message):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")


def notify_gui(summary, body):
    # Use notify-send on Linux desktops if available
    try:
        subprocess.run(["notify-send", summary, body], check=False)
    except FileNotFoundError:
        # notify-send not installed / not available
        pass


def resolve_player_name(league, player_id):
    try:
        details = league.player_details(int(player_id))
        if isinstance(details, list) and len(details) > 0:
            return details[0].get('name', {}).get('full') or details[0].get('name')
        elif isinstance(details, dict):
            return details.get('name', {}).get('full') or details.get('name')
    except Exception:
        return str(player_id)
    return str(player_id)


def main():
    p = argparse.ArgumentParser(description='Watch Yahoo league draft and notify on new picks')
    p.add_argument('--league-id', required=True, help='League id (e.g. nhl.l.2354566)')
    p.add_argument('--oauth-file', required=True, help='Path to oauth2.json')
    p.add_argument('--interval', type=int, default=15, help='Polling interval in seconds')
    p.add_argument('--use-gui', action='store_true', help='Use desktop notifications via notify-send')
    p.add_argument('--since-pick', type=int, default=0, help='Start reporting after this pick number')
    p.add_argument('--resolve-names', action='store_true', help='Resolve player IDs to names via Yahoo player_details()')
    args = p.parse_args()

    sc = OAuth2(None, None, from_file=args.oauth_file)
    lg = yfa.League(sc, args.league_id)

    last_seen = args.since_pick
    notify_cli(f"Starting draft watcher for {args.league_id}, polling every {args.interval}s")

    try:
        while True:
            try:
                picks = lg.draft_results()
            except Exception as e:
                notify_cli(f"Error fetching draft_results: {e}")
                picks = []

            # picks are a list of dicts with keys like 'pick', 'round', 'team_key', 'player_id'
            new_picks = [p for p in picks if 'pick' in p and int(p['pick']) > last_seen]
            # Sort by pick number to ensure order
            new_picks.sort(key=lambda x: int(x['pick']))

            for pck in new_picks:
                pick_num = int(pck.get('pick'))
                rnd = pck.get('round')
                team = pck.get('team_key')
                pid = pck.get('player_id')
                if args.resolve_names and pid is not None:
                    name = resolve_player_name(lg, pid)
                else:
                    # sometimes the API returns a nested player_key instead of player_id
                    name = str(pid)
                msg = f"Pick {pick_num} (R{rnd}) {team}: {name}"
                notify_cli(msg)
                if args.use_gui:
                    notify_gui(f"Draft Pick {pick_num}", msg)
                last_seen = max(last_seen, pick_num)

            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        notify_cli("Draft watcher stopped by user")


if __name__ == '__main__':
    main()
