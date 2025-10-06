#!/usr/bin/env python3
"""Simple CLI to rank players from the QuantHockey Excel and write CSV.

Usage:
  ./env/bin/python scripts/rank_players.py [--input path] [--sheet name] [--out path] [--top N]

By default it reads `data/QuantHockey_1759620509.xlsx` sheet 'QuantHockey' with header row 2,
computes our scoring and any Yahoo comparison if a Yahoo column is present, then writes
`ranked_players.csv` in the repo root and prints the top N rows to the console.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from yahoo_fantasy_bot import scoring


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/QuantHockey_1759620509.xlsx")
    p.add_argument("--sheet", default="QuantHockey")
    p.add_argument("--out", default="ranked_players.csv")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--projected-games", type=int, default=82)
    p.add_argument("--k", type=float, default=20.0, help="Shrinkage prior weight")
    p.add_argument("--decay", type=float, default=0.5, help="Decay factor for multi-file weighting (0<decay<=1)")
    p.add_argument("--sort-by", choices=['name','mtime'], default='mtime', help="How to sort discovered input files")
    p.add_argument("--reverse", action='store_true', help="Reverse the discovered file order after sorting (useful when you prefer lexicographic newest-first)")
    p.add_argument("--goalie-method", choices=['stats','gp-fallback','constant'], default='gp-fallback', help="Goalie projection method when goalie stats are missing or present")
    p.add_argument("--no-per-game", dest='compute_per_game', action='store_false', help="Disable per-game computations and use raw totals for projections")
    p.set_defaults(compute_per_game=True)
    p.add_argument("--weight-by-games", dest='weight_by_games', action='store_true', help="Multiply per-file contributions by games played (default)")
    p.add_argument("--no-weight-by-games", dest='weight_by_games', action='store_false', help="Do not weight per-file contributions by games played")
    p.set_defaults(weight_by_games=True)
    p.add_argument("--normalize-file-weights", action='store_true', help="Normalize per-file weights per-player so per-file weights sum to 1 across files")
    p.add_argument("--yahoo-points-field", default=None, help="When fetching Yahoo points, prefer this stat key (e.g. PPT, total_points)")
    p.add_argument("--fuzzy-match", type=float, default=0.0, help="Enable fuzzy name matching threshold (0-100); 0 disables fuzzy matching")
    p.add_argument("--fetch-yahoo", action='store_true', help="Attempt to fetch Yahoo Y! points via API (requires --league-id and oauthFile in config)")
    p.add_argument("--league-id", default=None, help="Yahoo league id to fetch yahoo points for (required when --fetch-yahoo is used)")
    p.add_argument("--oauth-file", default="oauth2.json", help="Path to yahoo oauth2 json file")
    args = p.parse_args(argv)

    data_dir = Path("data")
    input_path = Path(args.input) if args.input else None

    # discover .xlsx files in data/ when using default input; sort newest-first by mtime
    files = []
    if args.input and Path(args.input).exists():
        files = [Path(args.input)]
    else:
        if data_dir.exists() and data_dir.is_dir():
            paths = list(data_dir.glob("*.xlsx"))
            if args.sort_by == 'name':
                paths = sorted(paths, key=lambda p: p.name)
            else:
                paths = sorted(paths, key=lambda p: p.stat().st_mtime)
            if args.reverse:
                paths = list(reversed(paths))
            else:
                # default behavior: newest-first by mtime
                if args.sort_by == 'mtime':
                    paths = list(reversed(paths))
            files = paths
    if not files:
        print("No input files found in data/ and --input not provided.")
        return 2

    print(f"Using files (newest-first): {[str(p) for p in files]}")
    print("Scoring players from multiple files (decay=0.5, weight_by_games=True)...")
    scored = scoring.score_multiple_files(
        [str(p) for p in files],
        sheet_name=args.sheet,
        decay=args.decay,
        weight_by_games=args.weight_by_games,
        projected_games=args.projected_games,
        k=args.k,
        normalize_file_weights=args.normalize_file_weights,
        compute_per_game=args.compute_per_game,
        goalie_method=args.goalie_method,
    )

    # Optionally fetch Yahoo points via API
    if args.fetch_yahoo:
        if not args.league_id:
            print("--league-id is required when --fetch-yahoo is used")
            return 2
        # Lazy import to avoid requiring yahoo_oauth when not used
        from yahoo_oauth import OAuth2
        import yahoo_fantasy_api as yfa
        print(f"Creating OAuth session from {args.oauth_file}...")
        sc = OAuth2(None, None, from_file=args.oauth_file)
        if not sc.token_is_valid():
            sc.refresh_access_token()
        lg = yfa.League(sc, args.league_id)
        names = scored['Name'].dropna().unique().tolist()
        print(f"Fetching Yahoo points for {len(names)} players from league {args.league_id}...")
        ydf = scoring.fetch_yahoo_points_for_names(lg, names, req_type='season', prefer_field=(args.yahoo_points_field or 'PPT'))
        # merge yahoo points into scored (match on Name)
        # optionally perform fuzzy matching when merging
        if args.fuzzy_match and args.fuzzy_match > 0.0:
            try:
                from rapidfuzz import process as rf_process
                from rapidfuzz import utils as rf_utils
                print('Using rapidfuzz for fuzzy name matching')
                # build mapping from ydf.name -> yahoo_points
                ymap = dict(zip(ydf['name'].fillna('').tolist(), ydf['yahoo_points'].tolist()))
                names_list = list(ymap.keys())
                matched = []
                thresh = args.fuzzy_match
                for idx, row in scored.iterrows():
                    n = row.get('Name') or ''
                    if n in ymap:
                        matched.append(ymap[n])
                        continue
                    # find best match
                    best = rf_process.extractOne(n, names_list)
                    if best and best[1] >= thresh:
                        matched.append(ymap.get(best[0]))
                    else:
                        matched.append(None)
                scored['yahoo_score'] = matched
            except Exception:
                print('rapidfuzz not available; falling back to exact merges')
                scored = scored.merge(ydf.rename(columns={'name': 'Name', 'yahoo_points': 'yahoo_score'}), on='Name', how='left')
        else:
            scored = scored.merge(ydf.rename(columns={'name': 'Name', 'yahoo_points': 'yahoo_score'}), on='Name', how='left')

    out_path = Path(args.out)
    # select useful columns for the CSV; prefer original name column plus computed columns
    default_cols = [c for c in ["Name", "Team"] if c in scored.columns]
    computed = [c for c in ["raw_score", "gp", "per_game", "shrunk_per_game", "projected_total", "adjusted_total", "ranking_score", "yahoo_score", "delta_vs_yahoo", "combined_shrunk_per_game", "combined_projected_total", "combined_ranking_score"] if c in scored.columns]
    cols = default_cols + computed
    scored.to_csv(out_path, columns=cols, index=False)
    print(f"Wrote ranked CSV to {out_path}")

    # If yahoo comparison columns exist, write a short report
    yahoo_cols = [c for c in scored.columns if c.startswith("yahoo_score")]
    if yahoo_cols:
        # prefer the newest file yahoo_score (f0) if present
        yahoo_col = yahoo_cols[0]
        # find matching combined/our score column
        our_col = "combined_ranking_score" if "combined_ranking_score" in scored.columns else "ranking_score"
        # handle missing Team column gracefully
        cols_for_report = ["Name", our_col, yahoo_col]
        if "Team" in scored.columns:
            cols_for_report.insert(1, "Team")
        report_df = scored[cols_for_report].copy()
        report_df = report_df.rename(columns={our_col: "our_score", yahoo_col: "yahoo_score"})
        report_df["delta"] = report_df["our_score"] - report_df["yahoo_score"]
        report_path = Path("yahoo_comparison.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Wrote Yahoo comparison to {report_path}")

        # print summary stats
        have = report_df[report_df["yahoo_score"] > 0]
        if not have.empty:
            mean_delta = have["delta"].mean()
            std_delta = have["delta"].std()
            print(f"\nYahoo comparison stats (N={len(have)}): mean delta={mean_delta:.2f}, std={std_delta:.2f}")
            # top positive deltas (we > yahoo)
            print("\nTop 10 players where our score > Yahoo:")
            print(have.sort_values("delta", ascending=False).head(10)[["Name", "Team", "our_score", "yahoo_score", "delta"]].to_string(index=False))
            print("\nTop 10 players where Yahoo > our score:")
            print(have.sort_values("delta").head(10)[["Name", "Team", "our_score", "yahoo_score", "delta"]].to_string(index=False))

    # choose ranking column (prefer combined if present)
    rank_col = "combined_ranking_score" if "combined_ranking_score" in scored.columns else "ranking_score"
    topn = scored.sort_values(rank_col, ascending=False).head(args.top)
    # print a concise table
    display_cols = [c for c in ["Name", "Team", "GP", "raw_score", rank_col, "yahoo_score", "delta_vs_yahoo"] if c in topn.columns]
    print(f"\nTop {args.top} players (by {rank_col}):")
    print(topn[display_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
