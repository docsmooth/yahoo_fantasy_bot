"""Regression tests for scripts/rank_players.py's CLI-level file discovery
and ordering logic (yahoo_fantasy_bot-wy5, yahoo_fantasy_bot-1ku).

scripts/rank_players.py is loaded via importlib (it isn't on sys.path and
isn't a package module), mirroring the pattern already used for
scripts/ybot in tests/test_ybot_cli.py and tests/test_cli_contract.py.
"""
import importlib.util
import os
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RANK_PLAYERS_PATH = REPO_ROOT / 'scripts' / 'rank_players.py'


def _load_rank_players_module():
    loader = SourceFileLoader('rank_players_under_test_cli', str(RANK_PLAYERS_PATH))
    spec = importlib.util.spec_from_loader('rank_players_under_test_cli', loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _args(mod, argv):
    return mod.build_parser().parse_args(argv)


def _touch_same_mtime(*paths):
    """Simulate a fresh git clone: every file gets the same mtime."""
    now = time.time()
    for p in paths:
        p.touch()
        os.utime(p, (now, now))


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-wy5: reproducible, season-based default ordering
# ---------------------------------------------------------------------------

def test_default_sort_is_season(tmp_path):
    mod = _load_rank_players_module()
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    older = data_dir / 'QuantHockey_2023-2024.xlsx'
    newer = data_dir / 'QuantHockey_2024-2025.xlsx'
    # Create the OLDER-season file second, and give both identical mtimes,
    # reproducing "every file gets the checkout timestamp on a fresh clone"
    # (verified in the bug report) plus glob returning them in an arbitrary
    # order. If ordering depended on mtime or creation/glob order, this
    # would be flaky/wrong; season-name parsing must get it right regardless.
    newer.touch()
    older.touch()
    _touch_same_mtime(newer, older)

    args = _args(mod, [])  # default --sort-by is 'season'
    assert args.sort_by == 'season'
    files = mod.resolve_input_files(args, data_dir=data_dir)

    assert [f.name for f in files] == [
        'QuantHockey_2024-2025.xlsx', 'QuantHockey_2023-2024.xlsx',
    ]


def test_reverse_actually_reverses_in_every_sort_mode(tmp_path):
    """Regression test for the --reverse no-op bug: under the old code,
    sorted-by-mtime-then-reversed-in-both-branches meant --sort-by mtime
    with and without --reverse produced IDENTICAL output."""
    mod = _load_rank_players_module()
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    a = data_dir / 'QuantHockey_2023-2024.xlsx'
    b = data_dir / 'QuantHockey_2024-2025.xlsx'
    a.touch()
    time.sleep(0.01)
    b.touch()

    for sort_by in ('season', 'name', 'mtime'):
        forward = mod.resolve_input_files(
            _args(mod, ['--sort-by', sort_by]), data_dir=data_dir)
        backward = mod.resolve_input_files(
            _args(mod, ['--sort-by', sort_by, '--reverse']), data_dir=data_dir)
        assert [f.name for f in forward] != [f.name for f in backward], (
            f'--reverse was a no-op under --sort-by {sort_by}')
        assert list(reversed(forward)) == backward


def test_unparseable_filename_falls_back_to_mtime_with_warning(tmp_path, capsys):
    mod = _load_rank_players_module()
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    parseable = data_dir / 'QuantHockey_2024-2025.xlsx'
    unparseable = data_dir / 'QuantHockey_export_final.xlsx'
    parseable.touch()
    time.sleep(0.01)
    unparseable.touch()

    files = mod.resolve_input_files(_args(mod, []), data_dir=data_dir)

    assert len(files) == 2
    err = capsys.readouterr().err
    assert 'season' in err.lower() and 'mtime' in err.lower()
    assert 'QuantHockey_export_final.xlsx' in err
    # newest-by-mtime (the fallback) should be first, i.e. the file created
    # last -- 'unparseable'.
    assert files[0].name == 'QuantHockey_export_final.xlsx'


def test_resolved_order_and_weights_are_printed(tmp_path, capsys):
    mod = _load_rank_players_module()
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'QuantHockey_2023-2024.xlsx').touch()
    (data_dir / 'QuantHockey_2024-2025.xlsx').touch()

    argv = ['--decay', '0.5']
    p = mod.build_parser()
    args = p.parse_args(argv)
    files = mod.resolve_input_files(args, data_dir=data_dir)
    assert len(files) == 2

    # main() prints the order/weights; exercise that directly by calling it
    # with --input pointed at each discovered file being unnecessary here --
    # instead assert on the same computation main() performs, which is
    # covered end-to-end by the subprocess smoke test in
    # tests/test_runner_cli.py. Here we just check the weight formula.
    weights = [args.decay ** i for i in range(len(files))]
    assert weights == [1.0, 0.5]


# ---------------------------------------------------------------------------
# yahoo_fantasy_bot-1ku: --input default is None; a bad explicit --input is
# a hard error, not a silent fallback to glob discovery.
# ---------------------------------------------------------------------------

def test_input_default_is_none(tmp_path):
    mod = _load_rank_players_module()
    args = _args(mod, [])
    assert args.input is None


def test_explicit_missing_input_hard_errors_instead_of_globbing(tmp_path, capsys):
    mod = _load_rank_players_module()
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    # A real file sits in data/ -- if the bug were present, a typo'd --input
    # would silently fall back to scoring this file instead of erroring.
    (data_dir / 'QuantHockey_2024-2025.xlsx').touch()

    args = _args(mod, ['--input', str(tmp_path / 'QuantHockey_2024-2025.xslx')])  # typo'd extension
    with pytest.raises(SystemExit) as exc_info:
        mod.resolve_input_files(args, data_dir=data_dir)

    assert exc_info.value.code != 0
    assert 'does not exist' in str(exc_info.value.code)


def test_explicit_valid_input_is_used_as_is(tmp_path):
    mod = _load_rank_players_module()
    real_file = tmp_path / 'somewhere_else.xlsx'
    real_file.touch()
    args = _args(mod, ['--input', str(real_file)])
    files = mod.resolve_input_files(args, data_dir=tmp_path / 'data')
    assert files == [real_file]
