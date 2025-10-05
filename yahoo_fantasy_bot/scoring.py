"""Player scoring utilities for QuantHockey data.

This module implements a flexible scoring function that computes a raw
fantasy score from counting stats and a games-play adjusted score using
simple empirical-Bayes shrinkage toward the league mean per-game rate.

Weights (defaults):
 - Goal: +6
 - Assist: +4
 - Plus/Minus: +2 per point (signed)
 - Power-play goal or assist (PPG/PPA): +2 each
 - Short-handed goal or assist (SHG/SHA): +2 each
 - Shot attempts: +0.6 each
 - PIM: -1 each
 - Block: +1 each

Functions:
 - score_row / score_dataframe

The scoring functions are robust to common alternate column names returned
by different data sources. They return raw totals, per-game rates, and
shrunken/per-game estimates that account for games played.
"""
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd
from typing import List

# Default weights
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "G": 6.0,
    "A": 4.0,
    "PLUS": 2.0,  # plus/minus (signed)
    "PPG": 2.0,
    "PPA": 2.0,
    "SHG": 2.0,
    "SHA": 2.0,
    "SATT": 0.6,  # shot attempts
    "PIM": -1.0,
    "BLK": 1.0,
}


def _first_available_value(row: pd.Series, candidates: Iterable[str]) -> float:
    """Return the first non-null numeric value from row for candidate column names."""
    for c in candidates:
        if c in row and pd.notna(row[c]):
            val = row[c]
            try:
                return float(val)
            except Exception:
                return 0.0
    return 0.0


_COLUMN_ALIASES: Dict[str, Iterable[str]] = {
    "G": ("G", "Goals", "goals", "go"),
    "A": ("A", "Assists", "assists"),
    # QuantHockey uses '+/-' as the column name
    "PLUS": ("+/-", "PlusMinus", "PLUSMINUS", "PlusMinusRating", "plusminus", "PM"),
    "PPG": ("PPG", "PPGf", "PowerPlayGoals", "ppg"),
    "PPA": ("PPA", "PPAf", "PowerPlayAssists", "ppa"),
    "SHG": ("SHG", "ShortHandedGoals", "shg"),
    "SHA": ("SHA", "ShortHandedAssists", "sha"),
    # QuantHockey uses 'SHOTS' as total shot attempts
    "SATT": ("SHOTS", "SAT", "SATT", "ShotAttempts", "Shot_Attempts", "SCA", "SOG", "Shots", "shots"),
    "PIM": ("PIM", "PIMs", "PenaltiesInMinutes", "PIMs"),
    "BLK": ("BLK", "Blocks", "blocks"),
    "GP": ("GP", "GGP", "Games", "GamesPlayed", "games_played", "games"),
}


def score_row(row: pd.Series, weights: Optional[Mapping[str, float]] = None) -> float:
    """Compute the raw fantasy score for a single player row.

    The row can contain a variety of column names; common aliases are
    checked. Missing fields are treated as zero.
    Returns the raw (season-total) fantasy score.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Extract stats using aliases
    G = _first_available_value(row, _COLUMN_ALIASES["G"])
    A = _first_available_value(row, _COLUMN_ALIASES["A"])
    PLUS = _first_available_value(row, _COLUMN_ALIASES["PLUS"])
    PPG = _first_available_value(row, _COLUMN_ALIASES["PPG"])
    PPA = _first_available_value(row, _COLUMN_ALIASES["PPA"])
    SHG = _first_available_value(row, _COLUMN_ALIASES["SHG"])
    SHA = _first_available_value(row, _COLUMN_ALIASES["SHA"])
    SATT = _first_available_value(row, _COLUMN_ALIASES["SATT"])
    PIM = _first_available_value(row, _COLUMN_ALIASES["PIM"])
    BLK = _first_available_value(row, _COLUMN_ALIASES["BLK"])

    raw = 0.0
    raw += w["G"] * G
    raw += w["A"] * A
    raw += w["PLUS"] * PLUS
    raw += w["PPG"] * PPG
    raw += w["PPA"] * PPA
    raw += w["SHG"] * SHG
    raw += w["SHA"] * SHA
    raw += w["SATT"] * SATT
    raw += w["PIM"] * PIM
    raw += w["BLK"] * BLK

    return raw


def score_dataframe(
    df: pd.DataFrame,
    games_col_candidates: Optional[Iterable[str]] = None,
    k: float = 20.0,
    projected_games: int = 82,
    weights: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Score a DataFrame of players and return augmented DataFrame.

    Parameters
    - df: input DataFrame containing player stats
    - games_col_candidates: optional list of column names to consider for games-played;
      if not provided, common aliases are used
    - k: shrinkage factor (higher -> more shrinkage toward league mean)
    - projected_games: number of games to project a full season (used when
      computing `projected_total`)
    - weights: optional weights override (see `DEFAULT_WEIGHTS`)

    Adds these columns to the returned DataFrame:
    - raw_score: raw season total score
    - gp: games played inferred
    - per_game: raw_score / gp (or raw_score if gp==0)
    - league_mean_per_game: scalar (same for all rows)
    - shrunk_per_game: shrinkage-adjusted per-game estimate
    - projected_total: shrunk_per_game * projected_games
    - adjusted_total: shrunk_per_game * gp (estimates for player's counted season)
    """
    df = df.copy()
    if games_col_candidates is None:
        games_col_candidates = _COLUMN_ALIASES["GP"]

    # compute raw scores
    df["raw_score"] = df.apply(lambda r: score_row(r, weights=weights), axis=1)

    # infer GP
    df["gp"] = df.apply(lambda r: _first_available_value(r, games_col_candidates), axis=1)

    # per-game
    df["per_game"] = df.apply(lambda r: (r["raw_score"] / r["gp"]) if r["gp"] and r["gp"] > 0 else 0.0, axis=1)

    # league mean per game (exclude zeros to avoid bias from players with no GP)
    nonzero = df["gp"] > 0
    if nonzero.any():
        league_mean = df.loc[nonzero, "per_game"].mean()
    else:
        league_mean = df["per_game"].mean()

    df["league_mean_per_game"] = league_mean

    # shrink per-game toward league mean using games as sample size and k as prior weight
    def _shrink(r):
        gp = r["gp"]
        pg = r["per_game"]
        if gp and gp > 0:
            return (pg * gp + league_mean * k) / (gp + k)
        else:
            return league_mean

    df["shrunk_per_game"] = df.apply(_shrink, axis=1)
    df["projected_total"] = df["shrunk_per_game"] * projected_games
    df["adjusted_total"] = df["shrunk_per_game"] * df["gp"]

    # Provide a final ranking value, default using projected_total
    df["ranking_score"] = df["projected_total"]

    # Detect Yahoo scoring column if present and compute delta
    yahoo_candidates = ("Y! Points", "Yahoo Points", "Yahoo", "Y!", "YPoints", "YahooScore", "Y! Pts")
    found = None
    for c in yahoo_candidates:
        if c in df.columns:
            found = c
            break
    if found:
        df["yahoo_score"] = pd.to_numeric(df[found], errors="coerce").fillna(0.0)
        df["delta_vs_yahoo"] = df["ranking_score"] - df["yahoo_score"]

    return df


def score_multiple_files(
    file_paths,
    sheet_name: str = "QuantHockey",
    header: int = 1,
    decay: float = 0.5,
    weight_by_games: bool = True,
    key_name: str = "Name",
    projected_games: int = 82,
    k: float = 20.0,
    weights: Optional[Mapping[str, float]] = None,
):
    """Read multiple QuantHockey-style files and combine scores with decaying weights.

    Parameters
    - file_paths: iterable of file paths in newest-to-oldest order (newest first)
    - sheet_name/header: passed to pandas.read_excel
    - decay: multiplicative decay applied per file step (0<decay<=1). Newest file weight=1, next=decay, next=decay^2, ...
    - weight_by_games: if True multiply each file's contribution by games played for that player in that year
    - key_name: the player name column to join on (default 'Name')
    - projected_games, k, weights: passed to internal `score_dataframe` calls

    Returns an aggregated DataFrame keyed by `key_name` with columns:
    - combined_shrunk_per_game: weighted combination of per-file shrunk_per_game
    - combined_projected_total: combined_shrunk_per_game * projected_games
    - per-file columns are prefixed by year index (0 newest)
    """
    # collect per-file scored frames
    frames = []
    for idx, path in enumerate(file_paths):
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, header=header)
        except Exception as e:
            raise
        scored = score_dataframe(df, k=k, projected_games=projected_games, weights=weights)
        # keep key, gp, shrunk_per_game, projected_total, raw_score
        cols_to_keep = [key_name, "gp", "shrunk_per_game", "projected_total", "raw_score"]
        if "yahoo_score" in scored.columns:
            cols_to_keep.append("yahoo_score")
        out = scored[cols_to_keep].copy()
        # rename columns to indicate file index (0=newest)
        suffix = f"_f{idx}"
        out = out.rename(columns={
            "gp": f"gp{suffix}",
            "shrunk_per_game": f"shrunk_per_game{suffix}",
            "projected_total": f"projected_total{suffix}",
            "raw_score": f"raw_score{suffix}",
        })
        if "yahoo_score" in out.columns:
            out = out.rename(columns={"yahoo_score": f"yahoo_score{suffix}"})
        frames.append(out)

    # merge frames on key
    merged = None
    for f in frames:
        if merged is None:
            merged = f
        else:
            merged = merged.merge(f, on=key_name, how="outer")

    # replace NaN gp with 0 and NaN shrunk_per_game with 0
    for col in merged.columns:
        if col.startswith("gp") or col.startswith("shrunk_per_game") or col.startswith("projected_total") or col.startswith("raw_score"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    # compute weights and weighted aggregate
    # newest file is index 0 in file_paths
    combined_pg = []
    combined_weights = []
    for idx in range(len(file_paths)):
        suffix = f"_f{idx}"
        gp_col = f"gp{suffix}"
        spg_col = f"shrunk_per_game{suffix}"
        # base weight
        base_w = decay ** idx
        if weight_by_games:
            w = base_w * merged[gp_col]
        else:
            w = base_w
        combined_pg.append((spg_col, w))
        combined_weights.append(w)

    # numerator: sum(spg * w), denominator: sum(w)
    num = 0
    denom = 0
    for spg_col, w in combined_pg:
        num = num + (merged[spg_col] * w)
        denom = denom + w

    # avoid division by zero
    merged["combined_shrunk_per_game"] = (num / denom).fillna(0.0)
    merged["combined_projected_total"] = merged["combined_shrunk_per_game"] * projected_games
    merged["combined_ranking_score"] = merged["combined_projected_total"]

    return merged


def fetch_yahoo_points_for_names(league, names: List[str], req_type: str = "season", season: Optional[int] = None, prefer_field: str = "PPT") -> pd.DataFrame:
    """Given a yahoo_fantasy_api League instance and a list of player names,
    attempt to resolve player IDs and fetch Yahoo-provided point totals.

    This function will:
    - call league.player_details(name) to resolve a player_id for each name
      (uses the first exact/full-name match when possible)
    - batch calls league.player_stats(player_ids, req_type, season=season)
      to retrieve stats and extract the `prefer_field` (e.g. 'PPT' or 'Avg-PPT')

    Returns a DataFrame with columns: ['name', 'player_id', 'yahoo_points']
    """
    # Resolve names to player_ids
    resolved = []
    name_to_id = {}
    for name in names:
        try:
            details = league.player_details(name)
        except Exception:
            details = []
        pid = None
        full = None
        if details:
            # details may be a list of dicts
            # prefer an exact match on full name
            for d in details:
                dname = d.get('name')
                if isinstance(dname, dict):
                    fullname = dname.get('full')
                else:
                    fullname = d.get('name')
                if fullname and fullname.lower() == name.lower():
                    pid = int(d.get('player_id') or d.get('player_id'))
                    full = fullname
                    break
            if pid is None:
                # fallback to first match
                first = details[0]
                pid = int(first.get('player_id')) if first.get('player_id') else None
                if isinstance(first.get('name'), dict):
                    full = first.get('name').get('full')
                else:
                    full = first.get('name')
        name_to_id[name] = (pid, full)

    # collect unique ids
    ids = [pid for pid, f in name_to_id.values() if pid]
    yahoo_rows = []
    # batch size: 50
    B = 50
    for i in range(0, len(ids), B):
        chunk = ids[i:i+B]
        try:
            stats = league.player_stats(chunk, req_type, season=season)
        except Exception:
            stats = []
        for s in stats:
            pid = int(s.get('player_id'))
            # prefer field may be nested or uppercase key
            val = None
            if prefer_field in s:
                val = s.get(prefer_field)
            else:
                # try uppercase/lowercase variations
                for k in s.keys():
                    if k.lower() == prefer_field.lower():
                        val = s.get(k)
                        break
            # store
            yahoo_rows.append({'player_id': pid, 'yahoo_points': float(val) if val is not None else None, 'name_from_yahoo': s.get('name') or s.get('full')})

    # Map back to original names
    rows = []
    pid_to_points = {r['player_id']: r for r in yahoo_rows}
    for name, (pid, fullname) in name_to_id.items():
        if pid and pid in pid_to_points:
            rows.append({'name': name, 'player_id': pid, 'yahoo_points': pid_to_points[pid]['yahoo_points']})
        else:
            rows.append({'name': name, 'player_id': pid, 'yahoo_points': None})

    return pd.DataFrame(rows)
