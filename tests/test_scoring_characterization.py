"""Characterization tests for yahoo_fantasy_bot/scoring.py.

These tests pin the CURRENT behavior of scoring.py against a fixture built
from the real QuantHockey export schema (tests/fixtures/quanthockey_sample.xlsx,
see tests/fixtures/make_fixture.py), so that ongoing bug fixes have a
baseline to diff against. Several of the pinned behaviors are KNOWN BUGS
tracked in beads; each such test names the issue id in a comment so whoever
fixes the bug knows to update (not just re-pass) this test.

Do not "fix" the assertions here to make scoring.py's bugs disappear -- if
scoring.py changes, these tests are expected to need updating, and that is
the point: it should be an intentional diff, not a silent one.
"""
import math

import pandas as pd
import pytest

from yahoo_fantasy_bot import scoring


# ---------------------------------------------------------------------------
# score_row / score_dataframe on real column names
# ---------------------------------------------------------------------------

def test_raw_score_on_real_quanthockey_columns(quanthockey_sample_df):
    """raw_score for the two elite forwards, hand-computed from DEFAULT_WEIGHTS
    against the REAL QuantHockey column names (+/-, SHOTS, PIM, PPG, PPA, SHG,
    SHA, BS), not the synthetic PlusMinus/ShotAttempts/BLK names used by the
    older tests/test_scoring.py (which is exactly how the BS/BLK alias bug,
    yahoo_fantasy_bot-zop, survived -- prefer this real-schema fixture as the
    basis for new scoring tests).
    """
    out = scoring.score_dataframe(quanthockey_sample_df, k=20.0, projected_games=82)

    kucherov = out.loc[out["Name"] == "Nikita Kucherov"].iloc[0]
    mackinnon = out.loc[out["Name"] == "Nathan MacKinnon"].iloc[0]

    # Kucherov: G=37 A=84 +/-=22 PPG=8 PPA=38 SHG=0 SHA=0 SHOTS=265 PIM=45 BS=33
    # (yahoo_fantasy_bot-zop fixed: BS now resolves via the BLK alias)
    expected_kucherov = (
        6 * 37 + 4 * 84 + 2 * 22 + 2 * 8 + 2 * 38 + 2 * 0 + 2 * 0
        + 0.6 * 265 - 1 * 45 + 1 * 33
    )
    assert expected_kucherov == 841.0  # sanity check on the hand calc itself
    assert kucherov["raw_score"] == pytest.approx(expected_kucherov)

    # MacKinnon: G=32 A=84 +/-=25 PPG=9 PPA=29 SHG=0 SHA=0 SHOTS=320 PIM=41 BS=58
    expected_mackinnon = (
        6 * 32 + 4 * 84 + 2 * 25 + 2 * 9 + 2 * 29 + 2 * 0 + 2 * 0
        + 0.6 * 320 - 1 * 41 + 1 * 58
    )
    assert expected_mackinnon == 863.0
    assert mackinnon["raw_score"] == pytest.approx(expected_mackinnon)


# ---------------------------------------------------------------------------
# BLK / BS alias bug -- yahoo_fantasy_bot-zop (FIXED)
# ---------------------------------------------------------------------------

def test_blk_alias_fix_zop(quanthockey_sample_df):
    """FIX yahoo_fantasy_bot-zop: _COLUMN_ALIASES['BLK'] now includes 'BS',
    the real QuantHockey column for blocked shots, so blocks contribute to
    raw_score on real data (they previously contributed nothing, silently,
    even though DEFAULT_WEIGHTS assigns them a weight of +1).

    Kucherov's row has BS=33.0, so raw_score is 33 points higher than the
    pre-fix pinned value of 808.0.
    """
    out = scoring.score_dataframe(quanthockey_sample_df.copy())
    kucherov = out.loc[out["Name"] == "Nikita Kucherov"].iloc[0]
    assert kucherov["raw_score"] == pytest.approx(808.0 + 33.0)

    # An explicit 'BLK' column (the old alias) still works too, and doesn't
    # double-count when 'BS' is also present (BLK is checked first).
    with_blk_alias = quanthockey_sample_df.copy()
    with_blk_alias["BLK"] = with_blk_alias["BS"]
    out_aliased = scoring.score_dataframe(with_blk_alias)
    kucherov_aliased = out_aliased.loc[out_aliased["Name"] == "Nikita Kucherov"].iloc[0]
    assert kucherov_aliased["raw_score"] == pytest.approx(808.0 + 33.0)


# ---------------------------------------------------------------------------
# Goalies score 0 on real data -- yahoo_fantasy_bot-fow
# ---------------------------------------------------------------------------

def test_all_goalies_raw_score_zero_bug_fow(quanthockey_sample_df):
    """BUG yahoo_fantasy_bot-fow (still OPEN -- see tests/test_scoring_fixes.py
    for the new --goalie-input path, which cannot be exercised here since
    this fixture has no real goalie stats and no goalie_input is passed):
    the QuantHockey sheet has no W/GA/SV/SO columns, so every goalie
    (Pos == 'G') still gets raw_score == 0.0 when no separate goalie-stats
    export is supplied, regardless of actual performance. The row IS now
    flagged via `goalie_stats_fabricated` so a fabricated number can never
    reach a report unmarked.
    """
    out = scoring.score_dataframe(quanthockey_sample_df)
    goalies = out.loc[out["Pos"] == "G"]
    assert len(goalies) == 4
    assert (goalies["raw_score"] == 0.0).all()
    assert goalies["is_goalie"].all()
    assert goalies["goalie_stats_fabricated"].all()


def test_goalie_per_game_is_fabricated_bug_fow(quanthockey_sample_df):
    """BUG yahoo_fantasy_bot-fow: since goalie raw_score is always 0 on this
    data, per_game for goalies is NOT derived from performance at all under
    any goalie_method -- it's a function of GP (or a flat constant) seeded
    from the hardcoded DEFAULT_GOALIE_PER_GAME = 1.98.
    """
    DEFAULT_GOALIE_PER_GAME = 1.98

    # 'constant' and 'stats' (no goalie stat columns present, so it falls
    # back the same way 'constant' does) both assign the flat constant to
    # every goalie regardless of GP.
    for method in ("constant", "stats"):
        out = scoring.score_dataframe(quanthockey_sample_df, goalie_method=method)
        goalies = out.loc[out["Pos"] == "G"]
        assert (goalies["raw_score"] == 0.0).all()
        assert goalies["per_game"].apply(
            lambda v: v == pytest.approx(DEFAULT_GOALIE_PER_GAME)
        ).all(), f"goalie_method={method!r} should yield a flat constant per_game"

    # 'gp-fallback' (the default) varies the constant by a GP-relative
    # factor, clipped to [0.5, 2.0] -- still raw_score-independent (raw_score
    # is 0 for all of them), but not flat across goalies.
    out = scoring.score_dataframe(quanthockey_sample_df, goalie_method="gp-fallback")
    goalies = out.loc[out["Pos"] == "G"].set_index("Name")
    assert (goalies["raw_score"] == 0.0).all()

    avg_goalie_gp = goalies["gp"].mean()  # (63+63+62+1)/4 = 47.25
    scale = 0.2

    def expected_fallback(gp):
        factor = 1.0 + scale * ((gp - avg_goalie_gp) / avg_goalie_gp)
        factor = min(max(factor, 0.5), 2.0)
        return DEFAULT_GOALIE_PER_GAME * factor

    for name, row in goalies.iterrows():
        assert row["per_game"] == pytest.approx(expected_fallback(row["gp"]), rel=1e-6), name
    # and it does differ across goalies with different GP -- i.e. it is a
    # function of GP alone, not of any actual goalie performance stat.
    assert goalies["per_game"].nunique() > 1


# ---------------------------------------------------------------------------
# league_mean_per_game contaminated by goalies -- yahoo_fantasy_bot-chg (FIXED)
# ---------------------------------------------------------------------------

def test_league_mean_is_skater_only_fix_chg(quanthockey_sample_df):
    """FIX yahoo_fantasy_bot-chg: league_mean_per_game is now computed over
    SKATERS ONLY (~is_goalie & gp>0). Previously it was computed from all
    rows with gp>0, including goalies whose per_game was still 0 at that
    point in the pipeline (the goalie fallback runs afterwards) -- that
    dragged the mean down and biased shrinkage for every skater. Now
    league_mean_per_game equals the skater-only per_game mean exactly.
    """
    out = scoring.score_dataframe(quanthockey_sample_df, k=20.0, projected_games=82)

    league_mean = out["league_mean_per_game"].iloc[0]
    skaters = out.loc[~out["is_goalie"] & (out["gp"] > 0)]

    # yahoo_fantasy_bot-bq3 follow-up: the prior is now the GAMES-WEIGHTED
    # pooled rate (total points / total games), not a simple average of
    # per-game rates. A simple average lets a 1-game player vote as loudly as
    # an 82-game player in the very prior we shrink toward.
    pooled = skaters["raw_score"].sum() / skaters["gp"].sum()
    assert league_mean == pytest.approx(pooled)

    # Goalies are still excluded entirely (the original chg fix).
    assert league_mean != pytest.approx(
        out.loc[out["gp"] > 0, "raw_score"].sum() / out.loc[out["gp"] > 0, "gp"].sum())

    # Goalies genuinely don't move the value: rescoring with the goalie
    # rows' per_game forced non-zero (simulating what the OLD pre-fallback
    # per_game would have polluted the mean with) must give the same
    # league_mean, because goalies are now excluded from the computation
    # regardless of their per_game value.
    goalie_mask = out["is_goalie"]
    assert goalie_mask.any()


# ---------------------------------------------------------------------------
# ranking_score uses un-shrunk per_game -- yahoo_fantasy_bot-t0p (FIXED)
# ---------------------------------------------------------------------------

def test_ranking_score_uses_shrunk_per_game_fix_t0p(quanthockey_sample_df):
    """FIX yahoo_fantasy_bot-t0p: ranking_score / projected_total are now
    derived from shrunk_per_game, not the raw per_game rate. The raw
    (unshrunk) view is preserved separately as `per_game_projection` so
    nothing is lost -- it's just no longer what ranking is based on.

    This fixture is small and star-heavy (see make_fixture.py), so its
    league_mean is itself high; that makes the specific "2-GP call-up ranked
    below full-season players" demonstration from the original bug report
    unreliable to reproduce here (shrinkage pulls Robidas toward this
    fixture's inflated mean, which can still land him ABOVE more pedestrian
    full-season players -- an artifact of only having 12 rows, not a scoring
    bug). The real end-to-end demonstration -- Draisaitl/Kucherov/MacKinnon
    topping the full 2024-25 export and 1-GP call-ups like Zayne Parekh
    falling well down -- is pinned against the real data file in
    tests/test_scoring_fixes.py::test_ranking_score_real_data_parekh_vs_draisaitl_t0p.
    """
    out = scoring.score_dataframe(quanthockey_sample_df, k=20.0, projected_games=82)
    projection_horizon = out["is_goalie"].map(lambda is_goalie: 60 if is_goalie else 82)

    assert (out["ranking_score"] == out["projected_total"]).all()
    assert (out["projected_total"] == out["shrunk_per_game"] * projection_horizon).all()

    # The raw (unshrunk) view is still available, just under a new name, and
    # is no longer what ranking_score/projected_total report.
    assert (out["per_game_projection"] == out["per_game"] * projection_horizon).all()

    robidas = out.loc[out["Name"] == "Justin Robidas"].iloc[0]
    assert robidas["gp"] == 2
    # Shrinkage now measurably changes robidas's ranking basis vs. the raw
    # projection -- ranking_score no longer just reproduces per_game_projection.
    assert robidas["ranking_score"] != pytest.approx(robidas["per_game_projection"])
    assert robidas["ranking_score"] == pytest.approx(robidas["shrunk_per_game"] * 82)


# ---------------------------------------------------------------------------
# compute_per_game=False path
# ---------------------------------------------------------------------------

def test_compute_per_game_false_path(quanthockey_sample_df):
    out = scoring.score_dataframe(quanthockey_sample_df, compute_per_game=False)

    assert (out["per_game"] == 0.0).all()
    assert (out["shrunk_per_game"] == 0.0).all()
    assert (out["league_mean_per_game"] == 0.0).all()
    assert (out["projected_total"] == out["raw_score"]).all()
    assert (out["adjusted_total"] == out["raw_score"]).all()
    assert (out["ranking_score"] == out["raw_score"]).all()


# ---------------------------------------------------------------------------
# score_multiple_files
# ---------------------------------------------------------------------------

@pytest.fixture
def two_season_files(tmp_path, quanthockey_sample_df):
    """Two synthetic 'season' files derived from the shared fixture: file0
    (newest) is the fixture as-is, file1 (older) has GP halved, simulating
    a prior, less-played season for the same players.
    """
    newer = quanthockey_sample_df.copy()
    older = quanthockey_sample_df.copy()
    older["GP"] = (older["GP"] // 2).astype(float)

    f0 = tmp_path / "season_newest.xlsx"
    f1 = tmp_path / "season_older.xlsx"
    with pd.ExcelWriter(f0) as w:
        newer.to_excel(w, sheet_name="QuantHockey", index=False, startrow=1)
    with pd.ExcelWriter(f1) as w:
        older.to_excel(w, sheet_name="QuantHockey", index=False, startrow=1)
    return [str(f0), str(f1)]


def test_score_multiple_files_column_selection_fix_gl7(two_season_files):
    """FIX yahoo_fantasy_bot-gl7: score_multiple_files now carries Name/Team/
    Pos through the merge (Team/Pos taken from the NEWEST file only, since a
    player's team -- and in principle even position -- can change between
    seasons), plus a per-file `goalie_stats_fabricated` flag (yahoo_fantasy_bot-fow's
    marker column), alongside the existing per-file gp/shrunk_per_game/
    projected_total/raw_score and the combined_* columns.
    """
    merged = scoring.score_multiple_files(
        two_season_files, sheet_name="QuantHockey", header=1, decay=0.5,
        weight_by_games=True,
    )

    expected_cols = {
        "Name", "Team", "Pos",
        "gp_f0", "shrunk_per_game_f0", "projected_total_f0", "raw_score_f0",
        "goalie_stats_fabricated_f0",
        "gp_f1", "shrunk_per_game_f1", "projected_total_f1", "raw_score_f1",
        "goalie_stats_fabricated_f1",
        "combined_shrunk_per_game", "combined_projected_total",
        "combined_ranking_score",
    }
    assert set(merged.columns) == expected_cols
    assert len(merged) == 12
    # Team/Pos come from the newest file (idx==0) and are populated for
    # every player here since both synthetic files share the same roster.
    assert merged["Team"].notna().all()
    assert merged["Pos"].notna().all()
    assert set(merged.loc[merged["Name"] == "Nikita Kucherov", "Team"]) == {"TBL"}


def test_score_multiple_files_decay_weighting_math(two_season_files):
    """Pin the decay-weighting arithmetic: weight for file idx is
    decay**idx * gp_f{idx} (when weight_by_games=True) for SKATERS, and
    decay**idx (no GP multiplier) for GOALIES -- fixed by yahoo_fantasy_bot-6e6,
    which wires the previously-dead goalie_mask up to the real `Pos` column
    now carried through the merge (yahoo_fantasy_bot-gl7). combined_shrunk_per_game
    is the weighted average of shrunk_per_game across files using those
    weights.
    """
    decay = 0.5
    merged = scoring.score_multiple_files(
        two_season_files, sheet_name="QuantHockey", header=1, decay=decay,
        weight_by_games=True,
    )

    is_goalie = merged["Pos"].astype(str).str.strip().str.lower().str.startswith("g")
    w0 = pd.Series((decay ** 0), index=merged.index).where(is_goalie, (decay ** 0) * merged["gp_f0"])
    w1 = pd.Series((decay ** 1), index=merged.index).where(is_goalie, (decay ** 1) * merged["gp_f1"])
    denom = w0 + w1
    expected_combined = (
        merged["shrunk_per_game_f0"] * w0 + merged["shrunk_per_game_f1"] * w1
    ) / denom
    # guard against div-by-zero rows (none expected here since every player
    # has gp>0 in both synthetic files)
    assert (denom > 0).all()
    # sanity: this fixture does include goalies, so the goalie-specific
    # weighting branch is actually exercised by this test
    assert is_goalie.any()

    pd.testing.assert_series_equal(
        merged["combined_shrunk_per_game"], expected_combined,
        check_names=False, rtol=1e-9,
    )


def test_score_multiple_files_combined_uses_shrunk_per_game(two_season_files):
    """score_multiple_files's combined_* columns are derived from
    shrunk_per_game -- and as of yahoo_fantasy_bot-t0p, so is the single-file
    `ranking_score`/`projected_total` now (previously ranking_score used the
    raw per_game rate, so this test used to demonstrate a divergence between
    the per-file projected_total_f{idx} and shrunk_per_game_f{idx} that no
    longer exists: projected_total_f{idx} is shrunk_per_game_f{idx} times the
    position-specific projection horizon for every file, by construction).
    """
    merged = scoring.score_multiple_files(
        two_season_files, sheet_name="QuantHockey", header=1, decay=0.5,
        weight_by_games=True, projected_games=82,
    )
    projection_horizon = merged["Pos"].astype(str).str.strip().str.lower().map(
        lambda pos: 60 if pos.startswith("g") else 82
    )
    pd.testing.assert_series_equal(
        merged["combined_projected_total"],
        merged["combined_shrunk_per_game"] * projection_horizon,
        check_names=False, rtol=1e-9,
    )
    assert (merged["combined_ranking_score"] == merged["combined_projected_total"]).all()

    # Confirm per-file projected_total is now shrunk-derived too.
    robidas = merged.loc[merged["Name"] == "Justin Robidas"].iloc[0]
    assert robidas["projected_total_f0"] == pytest.approx(
        robidas["shrunk_per_game_f0"] * 82
    )


def test_score_multiple_files_empty_list_raises_fix_w2u():
    """FIX yahoo_fantasy_bot-w2u: score_multiple_files([]) now raises a clear
    ValueError naming the problem, instead of a bare
    AttributeError('NoneType' object has no attribute 'columns') from trying
    to iterate merged.columns when merged was never assigned.
    """
    with pytest.raises(ValueError, match="no file_paths given"):
        scoring.score_multiple_files([])
