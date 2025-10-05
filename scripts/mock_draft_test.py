#!/usr/bin/env python3
"""Simulate a mock snake draft using the ranking and produce results.

This is a small tool to simulate drafting N teams with M rounds by taking the
top ranked available player each pick (greedy). It writes `mock_draft_results.csv`.
"""
import argparse
from pathlib import Path
import pandas as pd

from yahoo_fantasy_bot import scoring


def simulate_snake_draft(scored_df, num_teams=10, rounds=15):
    # Create picks order for snake draft
    picks = []
    for r in range(rounds):
        order = list(range(num_teams)) if r % 2 == 0 else list(reversed(range(num_teams)))
        for t in order:
            picks.append({'round': r+1, 'team': t})

    available = scored_df.sort_values('combined_ranking_score' if 'combined_ranking_score' in scored_df.columns else 'ranking_score', ascending=False).copy()
    teams = {i: [] for i in range(num_teams)}
    for pick in picks:
        # pop the top available
        if available.empty:
            break
        plyr = available.iloc[0]
        teams[pick['team']].append(plyr.to_dict())
        available = available.iloc[1:]

    # normalize to dataframe
    rows = []
    for t, players in teams.items():
        for slot, p in enumerate(players, start=1):
            rows.append({'team': t, 'slot': slot, 'name': p.get('Name'), 'team_abbr': p.get('Team'), 'score': p.get('combined_ranking_score', p.get('ranking_score'))})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--top-teams', type=int, default=10)
    p.add_argument('--rounds', type=int, default=15)
    p.add_argument('--out', default='mock_draft_results.csv')
    args = p.parse_args(argv)

    # Use the existing runner behavior to get scored players
    files = list(Path('data').glob('*.xlsx'))
    if not files:
        print('No data files found in data/ to run draft simulation')
        return 1
    scored = scoring.score_multiple_files([str(p) for p in files])
    draft_df = simulate_snake_draft(scored, num_teams=args.top_teams, rounds=args.rounds)
    draft_df.to_csv(args.out, index=False)
    print(f'Wrote mock draft results to {args.out}')


if __name__ == '__main__':
    raise SystemExit(main())
