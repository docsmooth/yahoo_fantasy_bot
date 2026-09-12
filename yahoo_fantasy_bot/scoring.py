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

Note: the real QuantHockey export also includes a 'HITS' column. This is
deliberately NOT scored -- there is no weight for it in DEFAULT_WEIGHTS and
no alias entry for it. That is a scoring-policy choice, not a bug (unlike
the BS/BLK alias miss tracked as yahoo_fantasy_bot-zop): don't add it
without an explicit decision to change the scoring policy.
"""
import sys
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

# Default goalie weights (user requested)
DEFAULT_GOALIE_WEIGHTS: Mapping[str, float] = {
    "W": 5.0,   # Win
    "GA": -3.0, # Goals Against (negative)
    "SV": 0.6,  # Save
    "SO": 5.0,  # Shutout
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
    # QuantHockey's real column for blocked shots is 'BS'; without it here
    # the +1/block weight in DEFAULT_WEIGHTS never applied to real data
    # (yahoo_fantasy_bot-zop).
    "BLK": ("BLK", "BS", "BkS", "BLKS", "Blocks", "blocks"),
    "GP": ("GP", "GGP", "Games", "GamesPlayed", "games_played", "games"),
}

# Goalie-specific aliases
_GOALIE_ALIASES: Dict[str, Iterable[str]] = {
    "W": ("W", "Wins", "wins"),
    "GA": ("GA", "GoalsAgainst", "goals_against", "GAa"),
    "SV": ("SV", "Saves", "saves", "SVs"),
    "SO": ("SO", "Shutouts", "shutouts"),
    # Optional rate fields
    "SVPCT": ("SV%", "SV%", "SVPct", "SavePct", "SvPct"),
    "GAA": ("GAA", "GoalsAgainstAverage", "gaa"),
    # Position hint
    "POS": ("Pos", "Position", "position"),
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

    # Detect goalie rows either by explicit position column or presence of goalie stats
    pos_val = _first_available_value(row, _GOALIE_ALIASES.get("POS", ())) if _GOALIE_ALIASES.get("POS") else None
    # _first_available_value returns float; for POS, we need raw string lookup
    pos_hint = None
    for c in _GOALIE_ALIASES.get("POS", ()):  # type: ignore[index]
        if c in row and pd.notna(row[c]):
            try:
                pos_hint = str(row[c])
            except Exception:
                pos_hint = None
            break

    # Check for goalie stat columns presence
    has_goalie_stats = False
    for alias_list in ("W", "GA", "SV", "SO", "SVPCT", "GAA"):
        if any((c in row and pd.notna(row[c])) for c in _GOALIE_ALIASES.get(alias_list, ())):
            has_goalie_stats = True
            break

    # If pos hint indicates goalie or goalie stats are present, compute goalie score
    is_goalie = False
    if pos_hint:
        try:
            if isinstance(pos_hint, str) and pos_hint.strip().lower().startswith("g"):
                is_goalie = True
        except Exception:
            is_goalie = False
    if has_goalie_stats:
        is_goalie = True

    # If goalie, compute goalie-specific raw score
    if is_goalie:
        # Extract goalie stats
        W = _first_available_value(row, _GOALIE_ALIASES.get("W", ()))
        GA = _first_available_value(row, _GOALIE_ALIASES.get("GA", ()))
        SV = _first_available_value(row, _GOALIE_ALIASES.get("SV", ()))
        SO = _first_available_value(row, _GOALIE_ALIASES.get("SO", ()))
        # Use configured goalie weights
        gw = dict(DEFAULT_GOALIE_WEIGHTS)
        # Build goalie raw score
        raw_goalie = 0.0
        raw_goalie += gw.get("W", 0.0) * W
        raw_goalie += gw.get("GA", 0.0) * GA
        raw_goalie += gw.get("SV", 0.0) * SV
        raw_goalie += gw.get("SO", 0.0) * SO
        return raw_goalie

    # Otherwise, treat as skater and compute as before
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
    compute_per_game: bool = True,
    goalie_method: str = "gp-fallback",
    weights: Optional[Mapping[str, float]] = None,
    goalie_stats_df: Optional[pd.DataFrame] = None,
    goalie_name_col: str = "Name",
    source_name: Optional[str] = None,
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
    - goalie_stats_df: optional separate QuantHockey-style goalie export
      (columns including W/GA/SV/SO) used to score goalie rows for real,
      keyed by `goalie_name_col` (see yahoo_fantasy_bot-fow). When a goalie
      row in `df` has no match here (or `goalie_stats_df` isn't given at
      all), its per_game estimate falls back to the constant/GP-based
      fabrication described below, is flagged in `goalie_stats_fabricated`,
      and a warning naming `source_name` (or the row count) is printed to
      stderr.
    - goalie_name_col: column name used to join `df` and `goalie_stats_df`
    - source_name: label used only in the stderr warning above (e.g. a file
      path), so the operator knows which input needs a --goalie-input

    Adds these columns to the returned DataFrame:
    - raw_score: raw season total score (goalie rows use real goalie stats
      when resolved via `goalie_stats_df`, or the same-sheet goalie columns
      if present; otherwise 0.0)
    - gp: games played inferred
    - per_game: raw_score / gp (or raw_score if gp==0)
    - league_mean_per_game: scalar (same for all rows); computed over
      SKATERS ONLY (excludes goalies, which are exempt from shrinkage --
      see yahoo_fantasy_bot-chg)
    - shrunk_per_game: shrinkage-adjusted per-game estimate (skaters only; goalies are not shrunk)
    - per_game_projection: per_game * projected_games -- the raw (unshrunk) view
    - projected_total: shrunk_per_game * projected_games (when
      compute_per_game=True); if compute_per_game=False projected_total=raw_score.
      Ranking is done on the shrunk estimate so small samples don't
      outrank full seasons (see yahoo_fantasy_bot-t0p).
    - adjusted_total: shrunk_per_game * gp (estimates for player's counted season)
    - goalie_stats_fabricated: True for goalie rows whose per_game is a
      fabricated GP-based/constant estimate rather than derived from real
      goalie stats (see yahoo_fantasy_bot-fow)
    """
    df = df.copy()
    if games_col_candidates is None:
        games_col_candidates = _COLUMN_ALIASES["GP"]

    # compute raw scores
    df["raw_score"] = df.apply(lambda r: score_row(r, weights=weights), axis=1)

    # detect goalies in the input DataFrame (by position column or goalie stat columns)
    def _is_goalie_row(r):
        # check position hints first
        for c in _GOALIE_ALIASES.get("POS", ()):  # type: ignore[index]
            if c in r and pd.notna(r[c]):
                try:
                    if isinstance(r[c], str) and r[c].strip().lower().startswith("g"):
                        return True
                except Exception:
                    pass
        # check for goalie stat columns
        for alias_list in ("W", "GA", "SV", "SO", "SVPCT", "GAA"):
            for c in _GOALIE_ALIASES.get(alias_list, ()):  # type: ignore[index]
                if c in r and pd.notna(r[c]):
                    return True
        return False

    df["is_goalie"] = df.apply(_is_goalie_row, axis=1)

    # infer GP
    df["gp"] = df.apply(lambda r: _first_available_value(r, games_col_candidates), axis=1)

    # yahoo_fantasy_bot-fow: if a separate goalie-stats export was supplied,
    # score those rows for real (DEFAULT_GOALIE_WEIGHTS via score_row, which
    # already takes the goalie branch once W/GA/SV/SO are present) and
    # overwrite the fabricated raw_score==0.0 for any matching goalie rows.
    df["_has_real_goalie_stats"] = False
    if goalie_stats_df is not None and len(goalie_stats_df) > 0 and goalie_name_col in df.columns:
        gsdf = goalie_stats_df.copy()
        if goalie_name_col in gsdf.columns:
            gsdf["_real_goalie_raw_score"] = gsdf.apply(lambda r: score_row(r, weights=weights), axis=1)
            gsdf = gsdf[[goalie_name_col, "_real_goalie_raw_score"]].drop_duplicates(
                subset=[goalie_name_col], keep="first"
            )
            df = df.merge(gsdf, on=goalie_name_col, how="left")
            has_real = df["is_goalie"] & df["_real_goalie_raw_score"].notna()
            df.loc[has_real, "raw_score"] = df.loc[has_real, "_real_goalie_raw_score"]
            df.loc[has_real, "_has_real_goalie_stats"] = True
            df = df.drop(columns=["_real_goalie_raw_score"])

    if compute_per_game:
        # per-game
    # For goalies: allow multiple strategies selected via `goalie_method`:
    # - 'stats': use goalie raw_score-derived per_game when goalie stats present
    # - 'gp-fallback': use GP-aware fallback scaled from goalie_mean (default)
    # - 'constant': use a constant DEFAULT_GOALIE_PER_GAME when stats missing
    # The `goalie_method` parameter controls that behavior.
        def _per_game_row(r):
            gp = float(r.get("gp") or 0.0)
            # detect goalie stat presence
            goalie_stats_present = False
            for alias_list in ("W", "GA", "SV", "SO"):
                for c in _GOALIE_ALIASES.get(alias_list, ()):  # type: ignore[index]
                    if c in r and pd.notna(r[c]):
                        goalie_stats_present = True
                        break
                if goalie_stats_present:
                    break
            if not r.get("is_goalie"):
                return (r["raw_score"] / gp) if gp and gp > 0 else 0.0

            # If we have a genuine stats-derived goalie raw_score -- either
            # from a matched yahoo_fantasy_bot-fow --goalie-input row, or
            # from goalie stat columns present on this same sheet -- use it
            # regardless of `goalie_method` (real signal always wins over
            # fabrication; `goalie_method` only controls the fallback used
            # when no real goalie stats are available at all).
            if (r.get("_has_real_goalie_stats") or goalie_stats_present) and gp and gp > 0:
                return r["raw_score"] / gp

            # For other goalie methods we'll compute per_game later via fallbacks
            return 0.0

        df["per_game"] = df.apply(_per_game_row, axis=1)

        # league mean per game -- SKATERS ONLY (yahoo_fantasy_bot-chg).
        # Goalies are exempt from shrinkage, so they have no business in the
        # shrinkage target; including them (at this point in the pipeline
        # their per_game is still 0.0, pre-fallback) dragged the mean down.
        skater_nonzero = (~df["is_goalie"]) & (df["gp"] > 0)
        if skater_nonzero.any():
            league_mean = df.loc[skater_nonzero, "per_game"].mean()
        else:
            league_mean = df["per_game"].mean()

        df["league_mean_per_game"] = league_mean

        # shrink per-game toward league mean using games as sample size and k as prior weight
        def _shrink(r):
            # Goalies are exempt from shrink -- use raw per_game
            if r.get("is_goalie"):
                return r["per_game"]
            gp = r["gp"]
            pg = r["per_game"]
            if gp and gp > 0:
                return (pg * gp + league_mean * k) / (gp + k)
            else:
                return league_mean

        df["shrunk_per_game"] = df.apply(_shrink, axis=1)
        # If goalies have missing or zero per_game (because goalie stats weren't
        # present in the source, and no --goalie-input matched them), provide a
        # conservative default so they receive meaningful projections. Compute
        # the mean per_game among goalies first.
        goalie_mean = None
        try:
            gm = df.loc[df["is_goalie"] & (df["per_game"] > 0), "per_game"]
            if not gm.empty:
                goalie_mean = gm.mean()
        except Exception:
            goalie_mean = None

        DEFAULT_GOALIE_PER_GAME = 1.98
        if goalie_mean is None or pd.isna(goalie_mean):
            goalie_mean = DEFAULT_GOALIE_PER_GAME

        # assign fallback per_game to goalies that have per_game == 0 -- these
        # are goalies for whom we have NO real stats (no --goalie-input match,
        # no same-sheet W/GA/SV/SO). yahoo_fantasy_bot-fow: warn loudly rather
        # than silently fabricating, and flag the fabricated rows so a
        # fabricated number never reaches the CSV unmarked.
        df["goalie_stats_fabricated"] = False
        mask_goalie_zero = (df["is_goalie"]) & (df["per_game"] == 0)
        if mask_goalie_zero.any():
            df.loc[mask_goalie_zero, "goalie_stats_fabricated"] = True
            count = int(mask_goalie_zero.sum())
            label = source_name or "<input>"
            print(
                f"WARNING: {count} goalie row(s) in {label} have no real "
                "goalie stats (no W/GA/SV/SO columns and no --goalie-input "
                "match); assigning a fabricated per_game estimate derived "
                "from GP/DEFAULT_GOALIE_PER_GAME instead of actual "
                "performance. Pass --goalie-input <path> with a real "
                "goalie-stats export to fix this (yahoo_fantasy_bot-fow).",
                file=sys.stderr,
            )
            if goalie_method == 'constant':
                df.loc[mask_goalie_zero, "per_game"] = float(goalie_mean)
            elif goalie_method == 'gp-fallback':
                # compute average GP among goalies (use >0 GPs when available)
                try:
                    goalie_gps = df.loc[df["is_goalie"], "gp"].astype(float)
                    avg_goalie_gp = float(goalie_gps[goalie_gps > 0].mean()) if (goalie_gps > 0).any() else 0.0
                except Exception:
                    avg_goalie_gp = 0.0

                # We'll vary the fallback per-game slightly based on a player's GP
                # so goalies with more playing time get modestly higher projections.
                # factor = 1 + scale * (gp - avg_gp) / avg_gp, clipped to [0.5, 2.0]
                scale = 0.2
                def _goalie_fallback_val(r):
                    gp = float(r.get("gp") or 0.0)
                    if avg_goalie_gp and avg_goalie_gp > 0:
                        factor = 1.0 + scale * ((gp - avg_goalie_gp) / avg_goalie_gp)
                    else:
                        factor = 1.0
                    # clip
                    if factor < 0.5:
                        factor = 0.5
                    if factor > 2.0:
                        factor = 2.0
                    return float(goalie_mean) * factor

                # apply per-row fallback
                df.loc[mask_goalie_zero, "per_game"] = df.loc[mask_goalie_zero].apply(_goalie_fallback_val, axis=1)
            else:
                # unknown method: fallback to constant
                df.loc[mask_goalie_zero, "per_game"] = float(goalie_mean)

        # Ensure goalie shrunk_per_game matches per_game after any fallback
        # (goalies are exempt from shrinkage, so their shrunk_per_game should
        # equal their per_game even after we assigned a fallback value).
        try:
            df.loc[df["is_goalie"], "shrunk_per_game"] = df.loc[df["is_goalie"], "per_game"]
        except Exception:
            # be conservative if assignment fails for any reason
            pass

        # yahoo_fantasy_bot-t0p: rank on the SHRUNK per-game rate, not the raw
        # rate, so a hot 1-2 game stretch can't outrank a full season. Keep
        # the raw (unshrunk) view available separately as
        # per_game_projection -- nothing is lost, it's just no longer the
        # default ranking basis.
        df["per_game_projection"] = df["per_game"] * projected_games
        df["projected_total"] = df["shrunk_per_game"] * projected_games
        df["adjusted_total"] = df["shrunk_per_game"] * df["gp"]

        # Provide a final ranking value, default using projected_total
        df["ranking_score"] = df["projected_total"]
    else:
        # per-game computation disabled: fall back to raw season totals for projections
        df["per_game"] = 0.0
        df["league_mean_per_game"] = 0.0
        df["shrunk_per_game"] = 0.0
        df["per_game_projection"] = df["raw_score"]
        df["projected_total"] = df["raw_score"]
        df["adjusted_total"] = df["raw_score"]
        df["ranking_score"] = df["raw_score"]
        df["goalie_stats_fabricated"] = False

    df = df.drop(columns=["_has_real_goalie_stats"])

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
    normalize_file_weights: bool = False,
    key_name: str = "Name",
    projected_games: int = 82,
    k: float = 20.0,
    compute_per_game: bool = True,
    goalie_method: str = "gp-fallback",
    weights: Optional[Mapping[str, float]] = None,
    goalie_input: Optional[str] = None,
    goalie_sheet_name: Optional[str] = None,
    goalie_header: Optional[int] = None,
):
    """Read multiple QuantHockey-style files and combine scores with decaying weights.

    Parameters
    - file_paths: iterable of file paths in newest-to-oldest order (newest first)
    - sheet_name/header: passed to pandas.read_excel
    - decay: multiplicative decay applied per file step (0<decay<=1). Newest file weight=1, next=decay, next=decay^2, ...
    - weight_by_games: if True multiply each file's contribution by games played for that player in that year
    - key_name: the player name column to join on (default 'Name')
    - projected_games, k, weights: passed to internal `score_dataframe` calls
    - goalie_input: optional path to a separate QuantHockey-style goalie
      export (columns including W/GA/SV/SO) used to score goalie rows for
      real in every file (yahoo_fantasy_bot-fow). Read once and reused
      across all `file_paths`.
    - goalie_sheet_name/goalie_header: sheet/header for `goalie_input`;
      default to `sheet_name`/`header` when not given.

    Returns an aggregated DataFrame keyed by `key_name` with columns:
    - combined_shrunk_per_game: weighted combination of per-file shrunk_per_game
    - combined_projected_total: combined_shrunk_per_game * projected_games
    - per-file columns are prefixed by year index (0 newest)
    - Team/Pos: carried through from the NEWEST file only (a player's team
      can change between seasons, so an older file's Team should not win)
    - goalie_stats_fabricated_f{idx}: per-file flag, True where that file's
      goalie per_game estimate is fabricated rather than stats-derived
    """
    if not file_paths:
        raise ValueError(
            "score_multiple_files: no file_paths given -- need at least one "
            "QuantHockey export to score (yahoo_fantasy_bot-w2u)"
        )

    goalie_stats_df = None
    if goalie_input:
        g_sheet = goalie_sheet_name if goalie_sheet_name is not None else sheet_name
        g_header = goalie_header if goalie_header is not None else header
        goalie_stats_df = pd.read_excel(goalie_input, sheet_name=g_sheet, header=g_header)

    # collect per-file scored frames
    frames = []
    for idx, path in enumerate(file_paths):
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, header=header)
        except Exception as e:
            raise
        scored = score_dataframe(
            df, k=k, projected_games=projected_games, compute_per_game=compute_per_game,
            goalie_method=goalie_method, weights=weights,
            goalie_stats_df=goalie_stats_df, goalie_name_col=key_name,
            source_name=str(path),
        )
        # keep key, gp, shrunk_per_game, projected_total, raw_score, goalie_stats_fabricated
        cols_to_keep = [key_name, "gp", "shrunk_per_game", "projected_total", "raw_score", "goalie_stats_fabricated"]
        if "yahoo_score" in scored.columns:
            cols_to_keep.append("yahoo_score")
        # Carry Name/Team/Pos through from the NEWEST file only (idx==0):
        # a player's team (and even position) can change between seasons, so
        # we don't want an older file's value to win a merge (yahoo_fantasy_bot-gl7).
        if idx == 0:
            for c in ("Team", "Pos"):
                if c in scored.columns:
                    cols_to_keep.append(c)
        out = scored[cols_to_keep].copy()
        # rename columns to indicate file index (0=newest); Team/Pos (idx==0
        # only) are left unsuffixed since there's only ever one such column.
        suffix = f"_f{idx}"
        out = out.rename(columns={
            "gp": f"gp{suffix}",
            "shrunk_per_game": f"shrunk_per_game{suffix}",
            "projected_total": f"projected_total{suffix}",
            "raw_score": f"raw_score{suffix}",
            "goalie_stats_fabricated": f"goalie_stats_fabricated{suffix}",
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

    # replace NaN gp/shrunk_per_game/projected_total/raw_score with 0, and
    # NaN goalie_stats_fabricated (player absent from that file) with False
    for col in merged.columns:
        if col.startswith("gp") or col.startswith("shrunk_per_game") or col.startswith("projected_total") or col.startswith("raw_score"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        elif col.startswith("goalie_stats_fabricated"):
            merged[col] = merged[col].fillna(False).astype(bool)

    # compute weights and weighted aggregate
    # newest file is index 0 in file_paths
    combined_pg = []
    # Build per-file weight series (per-player) so we can optionally normalize per player
    for idx in range(len(file_paths)):
        suffix = f"_f{idx}"
        gp_col = f"gp{suffix}"
        spg_col = f"shrunk_per_game{suffix}"
        base_w = decay ** idx
        # yahoo_fantasy_bot-6e6: don't multiply a goalie's contribution by GP
        # (a goalie's games-played isn't commensurate with a skater's, and
        # weight_by_games is meant to reward skaters for playing more).
        # Pos is carried through from the newest file only (see above), so
        # this mask is the same across all file indices.
        goalie_mask = pd.Series(False, index=merged.index)
        if "Pos" in merged.columns:
            pos = merged["Pos"].astype(str).str.strip().str.lower()
            goalie_mask = pos.str.startswith("g") | pos.str.contains("goal")

        if weight_by_games:
            # per-player series: multiply by gp for skaters, but not for goalies
            w = base_w * merged[gp_col]
            # where goalie_mask is True, revert to base_w (no gp multiplier)
            if goalie_mask.any():
                w = w.where(~goalie_mask, base_w)
        else:
            # scalar -> convert to series for consistent operations
            w = pd.Series([base_w] * len(merged), index=merged.index)
        combined_pg.append((spg_col, w))

    # Optionally normalize file weights per-player so the weights across files sum to 1
    if normalize_file_weights and len(combined_pg) > 0:
        # sum of weights per player
        total_w = None
        for _, w in combined_pg:
            if total_w is None:
                total_w = w.copy()
            else:
                total_w = total_w + w
        # avoid division by zero; where total_w == 0 we leave weights as zero
        for i, (spg_col, w) in enumerate(combined_pg):
            # safe division: where total_w>0, divide, else keep zero
            denom_mask = total_w > 0
            new_w = pd.Series([0.0] * len(w), index=w.index)
            if denom_mask.any():
                new_w.loc[denom_mask] = w.loc[denom_mask] / total_w.loc[denom_mask]
            combined_pg[i] = (spg_col, new_w)

    # numerator: sum(spg * w), denominator: sum(w)
    num = pd.Series([0.0] * len(merged), index=merged.index)
    denom = pd.Series([0.0] * len(merged), index=merged.index)
    for spg_col, w in combined_pg:
        num = num + (merged[spg_col] * w)
        denom = denom + w

    # avoid division by zero across players
    combined = pd.Series([0.0] * len(merged), index=merged.index)
    nonzero = denom != 0
    if nonzero.any():
        combined.loc[nonzero] = num.loc[nonzero] / denom.loc[nonzero]
    merged["combined_shrunk_per_game"] = combined.fillna(0.0)
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
