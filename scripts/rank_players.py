#!/usr/bin/env python3
"""Simple CLI to rank players from QuantHockey Excel exports and write CSV.

Usage:
  ./env/bin/python scripts/rank_players.py [--input path] [--sheet name] [--out path] [--top N]

By default (no --input) it discovers all `data/*.xlsx` files, orders them
newest-first by the season parsed out of each filename (e.g.
`QuantHockey_2024-2025.xlsx`), reads sheet 'QuantHockey' with header row 2,
computes our scoring and any Yahoo comparison if a Yahoo column is present,
then writes the ranked CSV to --out (default `ranked_players.csv`, relative
to the current working directory) and prints the top N rows to the console.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from yahoo_fantasy_bot import scoring


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Path to a single QuantHockey .xlsx file. When omitted, all data/*.xlsx files are discovered and combined. When given, the path MUST exist -- it is a hard error otherwise (it will NOT silently fall back to discovery).")
    p.add_argument("--sheet", default="QuantHockey")
    p.add_argument("--out", default="ranked_players.csv")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--projected-games", type=int, default=82)
    p.add_argument("--k", type=float, default=20.0, help="Shrinkage prior weight")
    p.add_argument("--decay", type=float, default=0.5, help="Decay factor for multi-file weighting (0<decay<=1)")
    p.add_argument("--sort-by", choices=['season', 'name', 'mtime'], default='season', help="How to sort discovered input files. 'season' (default) parses the season out of the filename (e.g. QuantHockey_2024-2025.xlsx) so ordering is reproducible across machines; falls back to 'mtime' with a warning if a filename doesn't parse. 'name' and 'mtime' are explicit opt-ins.")
    p.add_argument("--reverse", action='store_true', help="Reverse the resolved (newest-first) file order, in every --sort-by mode.")
    p.add_argument("--goalie-method", choices=['stats','gp-fallback','constant'], default='gp-fallback', help="Goalie projection method used ONLY as a fallback when no real goalie stats are available (see --goalie-input)")
    p.add_argument("--goalie-input", default=None, help="Path to a separate QuantHockey-style goalie export (columns include W/GA/SV/SO) used to score goalies for real instead of fabricating a GP-based estimate (yahoo_fantasy_bot-fow)")
    p.add_argument("--goalie-sheet", default=None, help="Sheet name for --goalie-input (defaults to --sheet)")
    p.add_argument("--goalie-header", type=int, default=None, help="Header row for --goalie-input (defaults to the same header row used for --input)")
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
    return p


_SEASON_RE = re.compile(r'(\d{4})-(\d{4})')


def _parse_season_key(path):
    """Parse a sortable (start_year, end_year) key out of a filename like
    QuantHockey_2024-2025.xlsx. Returns None if the filename doesn't match."""
    m = _SEASON_RE.search(path.stem)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def resolve_input_files(args, data_dir=None):
    """Resolve the ordered (newest-first) list of input files.

    Raises SystemExit(2) if an explicitly-passed --input does not exist.
    """
    if data_dir is None:
        data_dir = Path("data")

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            sys.exit(
                f"ERROR: --input path does not exist: {input_path}. "
                f"Pass a valid path, or omit --input to discover files under "
                f"{data_dir}/ instead."
            )
        return [input_path]

    if not (data_dir.exists() and data_dir.is_dir()):
        return []

    paths = list(data_dir.glob("*.xlsx"))
    if not paths:
        return []

    sort_by = args.sort_by
    if sort_by == 'season':
        unparsed = [p for p in paths if _parse_season_key(p) is None]
        if unparsed:
            print(
                f"WARNING: could not parse a season (YYYY-YYYY) out of "
                f"filename(s): {[str(p) for p in unparsed]}; falling back "
                f"to --sort-by mtime for file ordering.",
                file=sys.stderr,
            )
            sort_by = 'mtime'

    if sort_by == 'season':
        paths = sorted(paths, key=_parse_season_key)  # oldest first
    elif sort_by == 'name':
        paths = sorted(paths, key=lambda p: p.name)  # oldest (alphabetically first) first
    else:
        paths = sorted(paths, key=lambda p: p.stat().st_mtime)  # oldest first

    # Every mode above sorts oldest-first; the natural/default presentation
    # (and what score_multiple_files' decay weighting requires) is
    # newest-first, so reverse once here. --reverse then flips that, and
    # actually does something in every mode (fixes yahoo_fantasy_bot-wy5,
    # where --sort-by mtime made --reverse a no-op because both the default
    # branch and the --reverse branch reversed the list).
    paths = list(reversed(paths))
    if args.reverse:
        paths = list(reversed(paths))

    return paths


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)

    files = resolve_input_files(args)
    if not files:
        print("No input files found in data/ and --input not provided.")
        return 2

    order_label = "oldest-first" if args.reverse else "newest-first"
    print(f"Resolved file order ({order_label}, --sort-by {args.sort_by}) "
          f"and decay weights (decay={args.decay}):")
    for idx, fpath in enumerate(files):
        weight = args.decay ** idx
        print(f"  [{idx}] {fpath}  weight={weight:.6g}")

    print("Scoring players from multiple files "
          f"(decay={args.decay}, weight_by_games={args.weight_by_games})...")
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
        goalie_input=args.goalie_input,
        goalie_sheet_name=args.goalie_sheet,
        goalie_header=args.goalie_header,
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
    # select useful columns for the CSV. `scored` always comes from
    # score_multiple_files, so per-file stat columns are suffixed (_f0, _f1,
    # ...); Name/Team/Pos are carried through unsuffixed from the newest file
    # (yahoo_fantasy_bot-gl7), and goalie_stats_fabricated_f{idx} flags any
    # fabricated (non-real-stats) goalie estimate (yahoo_fantasy_bot-fow).
    default_cols = [c for c in ["Name", "Team", "Pos"] if c in scored.columns]
    suffixed_prefixes = (
        "gp_f", "raw_score_f", "shrunk_per_game_f", "projected_total_f",
        "yahoo_score_f", "goalie_stats_fabricated_f",
    )
    per_file_cols = [c for c in scored.columns if any(c.startswith(pre) for pre in suffixed_prefixes)]
    combined_cols = [c for c in ["combined_shrunk_per_game", "combined_projected_total", "combined_ranking_score"] if c in scored.columns]
    cols = default_cols + per_file_cols + combined_cols
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
    # print a concise table; gp/raw_score are per-file (suffixed) now that
    # score_multiple_files carries them through as such -- show the newest
    # file's (f0) values, which is the most relevant "current season" view.
    gp_display_col = "gp_f0" if "gp_f0" in topn.columns else None
    raw_display_col = "raw_score_f0" if "raw_score_f0" in topn.columns else None
    yahoo_display_col = next((c for c in topn.columns if c.startswith("yahoo_score")), None)
    display_cols = [c for c in ["Name", "Team", "Pos"] if c in topn.columns]
    display_cols += [c for c in (gp_display_col, raw_display_col, rank_col, yahoo_display_col) if c]
    print(f"\nTop {args.top} players (by {rank_col}):")
    print(topn[display_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
