"""Regression tests for yahoo_fantasy_bot-soj and yahoo_fantasy_bot-7vr.

yahoo_fantasy_bot-soj: rank_players' resolve_input_files() globbed data/*.xlsx
without distinguishing skater exports (Pos column, 50ish columns) from goalie
exports (W/GA/SV/SO columns, no Pos column). A goalie export's season-shaped
filename (QuantHockey-Goalies_2025-2026.xlsx) sorted it as the newest
"skater" file, so a 98-row goalie sheet became the primary input to every
ranking.

yahoo_fantasy_bot-7vr: --goalie-input took a single path; goalies now have
two real seasons (2024-2025, 2025-2026) and should get the same decay
weighting skaters get across multiple season files.

Fixtures here are hand-built (per the make_fixture.py/make_goalie_fixture.py
convention: real column names, header row 1, 'info' sheet + 'QuantHockey'
sheet) rather than depending on data/, which is large and may not be
committed.
"""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pandas as pd
import pytest

from yahoo_fantasy_bot import scoring

REPO_ROOT = Path(__file__).resolve().parent.parent
RANK_PLAYERS_PATH = REPO_ROOT / 'scripts' / 'rank_players.py'


def _load_rank_players_module():
    loader = SourceFileLoader('rank_players_under_test_goalie_ms', str(RANK_PLAYERS_PATH))
    spec = importlib.util.spec_from_loader('rank_players_under_test_goalie_ms', loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _args(mod, argv):
    return mod.build_parser().parse_args(argv)


def _write_quanthockey_xlsx(path, rows):
    """Write rows (list of dicts) to `path` mirroring the real QuantHockey
    export shape: an 'info' banner sheet, then 'QuantHockey' with a blank
    banner row above the real header row (startrow=1, i.e. header=1 on
    read)."""
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([["Source:", "test fixture"]]).to_excel(
            writer, sheet_name="info", index=False, header=False)
        df.to_excel(writer, sheet_name="QuantHockey", index=False, startrow=1)


def _skater_rows():
    return [
        {"Name": "Test Forward", "Team": "TOR", "Pos": "C", "GP": 82,
         "G": 40, "A": 40, "P": 80, "PIM": 20, "+/-": 10, "SHOTS": 250, "BS": 10},
    ]


def _goalie_rows(**overrides):
    row = {
        "Name": "Test Goalie", "Team": "WPG", "Age": 30, "GP": 60,
        "GAA": 2.20, "SV%": 0.915, "W": 40, "L": 15, "GA": 130, "SV": 1500,
        "SOG": 1630, "SO": 5, "TIME": "3540:00", "G": 0, "A": 0, "P": 0, "PIM": 2,
    }
    row.update(overrides)
    return [row]


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-soj: discovery must separate skater from goalie exports
# ---------------------------------------------------------------------------

def test_goalie_export_excluded_from_skater_discovery(tmp_path):
    """The exact bug scenario: a goalie export's season-shaped filename
    (QuantHockey-Goalies_2025-2026.xlsx) must NOT be treated as the newest
    skater file."""
    mod = _load_rank_players_module()
    data_dir = tmp_path / "data"
    _write_quanthockey_xlsx(data_dir / "QuantHockey_2024-2025.xlsx", _skater_rows())
    _write_quanthockey_xlsx(data_dir / "QuantHockey-Goalies_2025-2026.xlsx", _goalie_rows())

    files = mod.resolve_input_files(_args(mod, []), data_dir=data_dir)

    assert [f.name for f in files] == ["QuantHockey_2024-2025.xlsx"]


def test_goalie_export_routed_to_goalie_discovery(tmp_path):
    mod = _load_rank_players_module()
    data_dir = tmp_path / "data"
    _write_quanthockey_xlsx(data_dir / "QuantHockey_2024-2025.xlsx", _skater_rows())
    _write_quanthockey_xlsx(data_dir / "QuantHockey-Goalies_2025-2026.xlsx", _goalie_rows())

    goalie_files = mod.resolve_goalie_files(_args(mod, []), data_dir=data_dir)

    assert [f.name for f in goalie_files] == ["QuantHockey-Goalies_2025-2026.xlsx"]


def test_real_five_file_bug_scenario_resolves_correctly(tmp_path):
    """Reproduce the exact reported bug: 3 skater seasons + 2 goalie seasons
    in data/. The resolved skater list must contain ONLY the 3 skater files,
    newest first, and the goalie files must resolve separately (also
    newest-first)."""
    mod = _load_rank_players_module()
    data_dir = tmp_path / "data"
    _write_quanthockey_xlsx(data_dir / "QuantHockey_2023-2024.xlsx", _skater_rows())
    _write_quanthockey_xlsx(data_dir / "QuantHockey_2024-2025.xlsx", _skater_rows())
    _write_quanthockey_xlsx(data_dir / "QuantHockey_2025-2026.xlsx", _skater_rows())
    _write_quanthockey_xlsx(data_dir / "QuantHockey-Goalies_2024-2025.xlsx", _goalie_rows())
    _write_quanthockey_xlsx(data_dir / "QuantHockey-Goalies_2025-2026.xlsx", _goalie_rows())

    args = _args(mod, [])
    skater_files = mod.resolve_input_files(args, data_dir=data_dir)
    goalie_files = mod.resolve_goalie_files(args, data_dir=data_dir)

    assert [f.name for f in skater_files] == [
        "QuantHockey_2025-2026.xlsx",
        "QuantHockey_2024-2025.xlsx",
        "QuantHockey_2023-2024.xlsx",
    ]
    assert [f.name for f in goalie_files] == [
        "QuantHockey-Goalies_2025-2026.xlsx",
        "QuantHockey-Goalies_2024-2025.xlsx",
    ]


def test_schema_overrides_misleading_filename(tmp_path):
    """yahoo_fantasy_bot-soj explicitly requires BOTH filename AND schema
    checks -- filename alone is too fragile. A file with a skater-shaped
    filename but a goalie-shaped schema (W/GA/SV/SO, no Pos) must still be
    classified as a goalie file (and vice versa)."""
    mod = _load_rank_players_module()
    data_dir = tmp_path / "data"
    # Named like a skater file, but its actual columns are goalie-shaped.
    _write_quanthockey_xlsx(data_dir / "QuantHockey_2025-2026.xlsx", _goalie_rows())

    skater_files = mod.resolve_input_files(_args(mod, []), data_dir=data_dir)
    goalie_files = mod.resolve_goalie_files(_args(mod, []), data_dir=data_dir)

    assert skater_files == []
    assert [f.name for f in goalie_files] == ["QuantHockey_2025-2026.xlsx"]


def test_file_matching_neither_shape_is_a_clear_error(tmp_path):
    """A file that is neither skater-shaped nor goalie-shaped, and whose
    name doesn't match either known pattern, must raise a clear error naming
    the file -- not be silently included in either list."""
    mod = _load_rank_players_module()
    data_dir = tmp_path / "data"
    _write_quanthockey_xlsx(
        data_dir / "SomeOtherExport.xlsx",
        [{"Name": "Whoever", "Team": "XXX", "RandomStat": 1}],
    )

    with pytest.raises(SystemExit) as exc_info:
        mod.resolve_input_files(_args(mod, []), data_dir=data_dir)

    msg = str(exc_info.value.code)
    assert "SomeOtherExport.xlsx" in msg
    assert "skater" in msg.lower() and "goalie" in msg.lower()


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-7vr: multi-season goalie decay weighting
# ---------------------------------------------------------------------------

def test_goalie_input_cli_flag_is_repeatable():
    mod = _load_rank_players_module()
    args = _args(mod, [
        "--goalie-input", "a.xlsx",
        "--goalie-input", "b.xlsx",
    ])
    assert args.goalie_input == ["a.xlsx", "b.xlsx"]


def test_resolve_goalie_files_explicit_multi_input_ordered_by_season(tmp_path):
    mod = _load_rank_players_module()
    data_dir = tmp_path / "data"
    older = tmp_path / "QuantHockey-Goalies_2024-2025.xlsx"
    newer = tmp_path / "QuantHockey-Goalies_2025-2026.xlsx"
    _write_quanthockey_xlsx(older, _goalie_rows())
    _write_quanthockey_xlsx(newer, _goalie_rows())

    args = _args(mod, ["--goalie-input", str(older), "--goalie-input", str(newer)])
    files = mod.resolve_goalie_files(args, data_dir=data_dir)

    # newest-first regardless of the order passed on the command line
    assert [f.name for f in files] == [
        "QuantHockey-Goalies_2025-2026.xlsx",
        "QuantHockey-Goalies_2024-2025.xlsx",
    ]


def test_hellebuyck_sanity_check():
    """Pin the exact formula from the ticket: W=47, GA=125, SV=1539, SO=8
    must score 5*47 - 3*125 + 0.6*1539 + 5*8 == 823.4."""
    row = pd.Series({"Name": "Connor Hellebuyck", "W": 47, "GA": 125, "SV": 1539, "SO": 8})
    assert scoring.score_row(row) == pytest.approx(823.4)


def test_combine_goalie_files_applies_decay_weighting(tmp_path):
    """Two goalie seasons for the same goalie must combine with decay
    weighting (newest=1, next=decay), not just use the newest file."""
    newest = tmp_path / "QuantHockey-Goalies_2025-2026.xlsx"
    older = tmp_path / "QuantHockey-Goalies_2024-2025.xlsx"
    _write_quanthockey_xlsx(newest, _goalie_rows(
        Name="Multi Season Goalie", GP=60, W=40, GA=130, SV=1500, SO=5))
    _write_quanthockey_xlsx(older, _goalie_rows(
        Name="Multi Season Goalie", GP=50, W=20, GA=140, SV=1300, SO=2))

    combined = scoring.combine_goalie_files(
        [str(newest), str(older)], sheet_name="QuantHockey", header=1, decay=0.5,
    )
    row = combined.set_index("Name").loc["Multi Season Goalie"]

    assert row["W"] == pytest.approx((40 * 1.0 + 20 * 0.5) / 1.5)
    assert row["GA"] == pytest.approx((130 * 1.0 + 140 * 0.5) / 1.5)
    assert row["SV"] == pytest.approx((1500 * 1.0 + 1300 * 0.5) / 1.5)
    assert row["SO"] == pytest.approx((5 * 1.0 + 2 * 0.5) / 1.5)
    assert row["GP"] == pytest.approx((60 * 1.0 + 50 * 0.5) / 1.5)


def test_score_multiple_files_blends_both_goalie_seasons(tmp_path):
    """End-to-end: a single skater-season file (with a goalie row) combined
    with TWO real goalie-season files must use the decay-blended raw_score
    across seasons -- not just the newest goalie file's numbers, and not the
    fabricated GP-based estimate."""
    skater_file = tmp_path / "QuantHockey_2025-2026.xlsx"
    _write_quanthockey_xlsx(skater_file, [
        {"Name": "Multi Season Goalie", "Team": "WPG", "Pos": "G", "GP": 60,
         "G": 0, "A": 0, "PIM": 0},
    ])
    goalie_newest = tmp_path / "QuantHockey-Goalies_2025-2026.xlsx"
    goalie_older = tmp_path / "QuantHockey-Goalies_2024-2025.xlsx"
    _write_quanthockey_xlsx(goalie_newest, _goalie_rows(
        Name="Multi Season Goalie", GP=60, W=40, GA=130, SV=1500, SO=5))
    _write_quanthockey_xlsx(goalie_older, _goalie_rows(
        Name="Multi Season Goalie", GP=50, W=20, GA=140, SV=1300, SO=2))

    merged = scoring.score_multiple_files(
        [str(skater_file)], sheet_name="QuantHockey", header=1, decay=0.5,
        goalie_input=[str(goalie_newest), str(goalie_older)],
    )
    goalie = merged.set_index("Name").loc["Multi Season Goalie"]

    expected_w = (40 * 1.0 + 20 * 0.5) / 1.5
    expected_ga = (130 * 1.0 + 140 * 0.5) / 1.5
    expected_sv = (1500 * 1.0 + 1300 * 0.5) / 1.5
    expected_so = (5 * 1.0 + 2 * 0.5) / 1.5
    expected_raw = 5 * expected_w - 3 * expected_ga + 0.6 * expected_sv + 5 * expected_so

    assert goalie["raw_score_f0"] == pytest.approx(expected_raw)
    assert not goalie["goalie_stats_fabricated_f0"]
    # Must NOT equal the newest-file-only raw score (proves both seasons
    # actually contributed, not just the newest goalie file).
    newest_only_raw = 5 * 40 - 3 * 130 + 0.6 * 1500 + 5 * 5
    assert goalie["raw_score_f0"] != pytest.approx(newest_only_raw)


def test_single_goalie_season_still_works_unblended(tmp_path):
    """A single goalie file (no multi-season blending needed) must behave
    exactly as before: goalie_input as a 1-item list is equivalent to the
    old single-path behavior."""
    skater_file = tmp_path / "QuantHockey_2024-2025.xlsx"
    _write_quanthockey_xlsx(skater_file, [
        {"Name": "Connor Hellebuyck", "Team": "WPG", "Pos": "G", "GP": 60,
         "G": 0, "A": 0, "PIM": 0},
    ])
    goalie_file = tmp_path / "QuantHockey-Goalies_2024-2025.xlsx"
    _write_quanthockey_xlsx(goalie_file, _goalie_rows(
        Name="Connor Hellebuyck", GP=60, W=47, GA=125, SV=1539, SO=8))

    merged = scoring.score_multiple_files(
        [str(skater_file)], sheet_name="QuantHockey", header=1, decay=0.5,
        goalie_input=[str(goalie_file)],
    )
    goalie = merged.set_index("Name").loc["Connor Hellebuyck"]
    assert goalie["raw_score_f0"] == pytest.approx(823.4)
    assert not goalie["goalie_stats_fabricated_f0"]


# ---------------------------------------------------------------------------
# Schema-drift regressions: SHOTS/SOG alias pinning, and the SOG name clash
# between the skater SATT alias and the goalie shots-against column.
# ---------------------------------------------------------------------------

def test_shots_and_sog_column_names_resolve_to_the_same_satt_stat():
    """The 2025-2026 skater export renamed 'SHOTS' (used in 2024-2025) to
    'SOG'. Both must resolve to the identical SATT-weighted contribution --
    this is exactly the class of silent schema drift that caused the
    BS/BLK bug (yahoo_fantasy_bot-zop)."""
    shots_row = pd.Series({"Name": "Player A", "Pos": "C", "G": 10, "A": 10, "SHOTS": 200})
    sog_row = pd.Series({"Name": "Player A", "Pos": "C", "G": 10, "A": 10, "SOG": 200})
    assert scoring.score_row(shots_row) == pytest.approx(scoring.score_row(sog_row))


def test_goalie_row_with_sog_never_scored_through_skater_branch():
    """A real goalie row has a 'SOG' column meaning shots-AGAINST, not the
    skater 'shots on goal' stat -- but the skater SATT alias list also
    contains 'SOG'. A goalie row must be detected and scored via the goalie
    branch (DEFAULT_GOALIE_WEIGHTS), never fall through to the skater
    formula using SOG as shot attempts."""
    goalie_row = pd.Series({
        "Name": "Test Goalie", "Team": "WPG", "GP": 60, "W": 40, "GA": 130,
        "SV": 1500, "SOG": 1630, "SO": 5, "G": 0, "A": 0, "P": 0, "PIM": 2,
    })
    score = scoring.score_row(goalie_row)
    expected_goalie_score = (
        scoring.DEFAULT_GOALIE_WEIGHTS["W"] * 40
        + scoring.DEFAULT_GOALIE_WEIGHTS["GA"] * 130
        + scoring.DEFAULT_GOALIE_WEIGHTS["SV"] * 1500
        + scoring.DEFAULT_GOALIE_WEIGHTS["SO"] * 5
    )
    # If this had fallen through to the skater branch, SATT would resolve to
    # SOG=1630 and contribute 0.6*1630=978 alone, which would blow way past
    # the real goalie formula's ~823-ish range for stats like these.
    wrong_skater_style_score = scoring.DEFAULT_WEIGHTS["SATT"] * 1630
    assert score == pytest.approx(expected_goalie_score)
    assert score != pytest.approx(wrong_skater_style_score)


def test_goalie_dataframe_row_flagged_is_goalie_despite_sog_column(tmp_path):
    """Same check at the score_dataframe level (is_goalie detection over a
    full sheet read from disk), since that's the real code path used by
    score_multiple_files against actual QuantHockey-Goalies_*.xlsx files."""
    goalie_file = tmp_path / "QuantHockey-Goalies_2025-2026.xlsx"
    _write_quanthockey_xlsx(goalie_file, _goalie_rows(
        Name="Test Goalie", W=40, GA=130, SV=1500, SOG=1630, SO=5))
    df = pd.read_excel(goalie_file, sheet_name="QuantHockey", header=1)

    out = scoring.score_dataframe(df)
    row = out.set_index("Name").loc["Test Goalie"]

    assert bool(row["is_goalie"])
    expected = (
        scoring.DEFAULT_GOALIE_WEIGHTS["W"] * 40
        + scoring.DEFAULT_GOALIE_WEIGHTS["GA"] * 130
        + scoring.DEFAULT_GOALIE_WEIGHTS["SV"] * 1500
        + scoring.DEFAULT_GOALIE_WEIGHTS["SO"] * 5
    )
    assert row["raw_score"] == pytest.approx(expected)
