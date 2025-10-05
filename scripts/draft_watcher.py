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
from yahoo_fantasy_bot import utils
import os

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


def resolve_player_names_batch(league, ids, cache):
    """Resolve a list of player IDs to names using cache + Yahoo as fallback.

    ids: iterable of ints
    cache: PlayerDetailsCache instance
    returns: dict id -> name
    """
    ids = [int(i) for i in ids]
    result = {}
    missing = []
    for pid in ids:
        cached = cache.get(pid)
        if cached:
            # cached may be a dict with nested 'name' structure
            if isinstance(cached, dict):
                name = cached.get('name')
                if isinstance(name, dict):
                    name = name.get('full') or name.get('full')
                result[pid] = name or str(pid)
            else:
                result[pid] = str(cached)
        else:
            missing.append(pid)

    # Fetch missing in chunks using player_details (max 25 per call)
    if missing:
        for i in range(0, len(missing), 25):
            chunk = missing[i:i+25]
            try:
                details = league.player_details(chunk)
            except Exception as e:
                # On error, fallback to IDs
                for pid in chunk:
                    result[pid] = str(pid)
                continue

            # player_details may return a list in same order as chunk
            mapping = {}
            for d in details:
                pid = int(d.get('player_id') or d.get('player_key', '').split('.')[-1])
                mapping[pid] = d
                # extract name
                name = None
                if isinstance(d.get('name'), dict):
                    name = d['name'].get('full') or d['name'].get('full')
                elif isinstance(d.get('name'), str):
                    name = d.get('name')
                result[pid] = name or str(pid)

            # update cache and save
            try:
                cache.update_many(mapping)
                cache.save()
            except Exception:
                pass

    return result


def main():
    p = argparse.ArgumentParser(description='Watch Yahoo league draft and notify on new picks')
    p.add_argument('--league-id', required=True, help='League id (e.g. nhl.l.2354566)')
    p.add_argument('--oauth-file', required=True, help='Path to oauth2.json')
    p.add_argument('--use-bot', action='store_true', help='Instantiate the main ManagerBot and reuse its cache paths')
    p.add_argument('--config', help='Path to config (my.cfg) to pass to ManagerBot when --use-bot is set')
    p.add_argument('--interval', type=int, default=15, help='Polling interval in seconds')
    p.add_argument('--use-gui', action='store_true', help='Use desktop notifications via notify-send')
    p.add_argument('--since-pick', type=int, default=0, help='Start reporting after this pick number')
    p.add_argument('--resolve-names', action='store_true', help='Resolve player IDs to names via Yahoo player_details()')
    args = p.parse_args()

    sc = OAuth2(None, None, from_file=args.oauth_file)

    cache_dir = os.path.join('.cache')
    # Optionally instantiate ManagerBot to reuse its cache dir
    if args.use_bot:
        if not args.config:
            print("--use-bot requires --config <path-to-my.cfg>")
            sys.exit(1)
        # Lazy import to avoid pulling bot deps when not requested
        from configparser import ConfigParser
        cfg = ConfigParser()
        cfg.read(args.config)
        try:
            from yahoo_fantasy_bot.bot import ManagerBot
            bot_inst = ManagerBot(cfg, reset_cache=False, ignore_status=True)
            lg = bot_inst.lg
            # try to reuse bot cache dir
            cache_dir = cfg['Cache'].get('dir', cache_dir)
        except Exception as e:
            notify_cli(f"Failed to instantiate ManagerBot: {e}")
            # fall back to direct League
            lg = yfa.League(sc, args.league_id)
    else:
        lg = yfa.League(sc, args.league_id)

    # Setup a simple player details cache file
    cache_file = os.path.join(cache_dir, f'player_details-{args.league_id}.pkl')
    player_cache = utils.PlayerDetailsCache(cache_file)

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

            if new_picks:
                # Collect all player IDs that we need to resolve
                pids = []
                for pck in new_picks:
                    pid = pck.get('player_id')
                    if pid is not None:
                        pids.append(int(pid))

                resolved = {}
                if args.resolve_names and len(pids) > 0:
                    # Batch-resolve all missing ids using the cache helper
                    resolved = resolve_player_names_batch(lg, pids, player_cache)

                # Append picks to a CSV log and notify
                logs_dir = os.path.join('logs')
                os.makedirs(logs_dir, exist_ok=True)
                log_file = os.path.join(logs_dir, f'draft-{args.league_id}.csv')
                write_header = not os.path.exists(log_file)
                with open(log_file, 'a', encoding='utf-8') as lf:
                    if write_header:
                        lf.write('timestamp,pick,round,team_key,player_id,player_name\n')

                    for pck in new_picks:
                        pick_num = int(pck.get('pick'))
                        rnd = pck.get('round')
                        team = pck.get('team_key')
                        pid = pck.get('player_id')
                        if args.resolve_names and pid is not None:
                            name = resolved.get(int(pid), str(pid))
                        else:
                            name = str(pid)

                        msg = f"Pick {pick_num} (R{rnd}) {team}: {name}"
                        notify_cli(msg)
                        if args.use_gui:
                            notify_gui(f"Draft Pick {pick_num}", msg)

                        # write to CSV
                        ts = datetime.now().isoformat()
                        lf.write(f'"{ts}",{pick_num},{rnd},"{team}",{pid},"{name}"\n')

                        last_seen = max(last_seen, pick_num)

            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        notify_cli("Draft watcher stopped by user")


if __name__ == '__main__':
    main()
