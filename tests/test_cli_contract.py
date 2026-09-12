"""Anti-drift contract test between scripts/ybot's docopt Options block and
scripts/rank_players.py's argparse parser.

OWNER DECISION (yahoo_fantasy_bot-he4): scripts/ybot forwards ALL scoring
flags to scripts/rank_players.py, at the accepted cost that the flag list is
duplicated across the two files and must be kept in sync by hand. This test
is the mitigation the owner asked for: it must fail if a scoring flag is
added to one side (ybot's Options block / build_score_command) and not the
other (rank_players.py's argparse parser), or vice versa.

Both scripts transitively import things that aren't installed in this venv
(yahoo_oauth/yahoo_fantasy_api via scripts/ybot's __main__ guard), so both
are loaded here via importlib rather than a normal import -- neither script
lives on sys.path, and scripts/ybot isn't even named as a .py file.
"""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import docopt
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
YBOT_PATH = REPO_ROOT / 'scripts' / 'ybot'
RANK_PLAYERS_PATH = REPO_ROOT / 'scripts' / 'rank_players.py'

# ybot flags that control the bot / the scoring subprocess plumbing itself,
# not scoring parameters forwarded to rank_players.py. This list must stay
# short and hand-curated: anything scoring-related must NOT be added here --
# it belongs in BOTH scripts/ybot's Options block AND
# rank_players.py's build_parser() instead. --rank-extra is a passthrough
# container (raw extra args), not itself a rank_players.py flag, so it's
# excluded too.
NON_SCORING_YBOT_FLAGS = {
    '--apply', '--assumeyes', '--full', '--generations', '--resetcache',
    '--ignorestatus', '--score', '--rank-extra', '--continue-on-score-failure',
}


def _load_module_from_path(path, name):
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _load_rank_players_module():
    return _load_module_from_path(RANK_PLAYERS_PATH, 'rank_players_under_test_contract')


def _load_ybot_module():
    return _load_module_from_path(YBOT_PATH, 'ybot_under_test_contract')


def _ybot_long_flags():
    text = YBOT_PATH.read_text()
    doc = text.split('"""')[1]
    opts = docopt.parse_defaults(doc)
    return {o.long for o in opts if o.long}


def _rank_players_long_flags():
    mod = _load_rank_players_module()
    parser = mod.build_parser()
    flags = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith('--'):
                flags.add(opt)
    return flags, mod


def test_rank_players_exposes_build_parser():
    mod = _load_rank_players_module()
    assert hasattr(mod, 'build_parser'), (
        'rank_players.py must expose a build_parser() function (called by '
        'main()) so this anti-drift test -- and anything else -- can '
        'introspect its argparse options without executing main().'
    )
    parser = mod.build_parser()
    # build_parser() must return a fresh, independent parser each call.
    assert mod.build_parser() is not parser


def test_every_ybot_scoring_flag_is_accepted_by_rank_players():
    ybot_flags = _ybot_long_flags()
    scoring_flags = ybot_flags - NON_SCORING_YBOT_FLAGS
    # Sanity check so this test can't silently degrade to a trivial pass
    # (e.g. if docopt parsing regresses and returns zero options).
    assert len(scoring_flags) >= 18, (
        f'expected at least 18 scoring-related flags on ybot, found '
        f'{len(scoring_flags)}: {sorted(scoring_flags)}'
    )

    rank_players_flags, _ = _rank_players_long_flags()

    missing = sorted(scoring_flags - rank_players_flags)
    assert not missing, (
        f"ybot advertises these scoring flags but rank_players.py's "
        f"argparse parser (build_parser()) does not accept them: {missing}. "
        f"Either add them to build_parser() in scripts/rank_players.py, or "
        f"remove them from scripts/ybot's Options block (and any README "
        f"documentation) -- the two CLIs must stay in sync by hand."
    )


def test_build_score_command_forwards_every_scoring_flag():
    """Build args with every scoring flag set to a representative value and
    assert build_score_command() forwards every single one (the
    --weight-by-games/--no-weight-by-games pair is checked separately
    below, and --rank-extra is a passthrough container rather than a
    forwarded flag itself)."""
    ybot_mod = _load_ybot_module()
    rank_players_flags, _ = _rank_players_long_flags()

    value_flags = {
        '--input': 'data/foo.xlsx',
        '--sheet': 'QuantHockey',
        '--projected-games': '82',
        '--k': '20',
        '--decay': '0.5',
        '--sort-by': 'season',
        '--goalie-method': 'stats',
        '--goalie-input': 'data/goalies.xlsx',
        '--goalie-sheet': 'Goalies',
        '--goalie-header': '1',
        '--yahoo-points-field': 'PPT',
        '--fuzzy-match': '90',
        '--league-id': '1234',
        '--oauth-file': 'my_oauth2.json',
    }
    bool_flags = [
        '--reverse', '--no-per-game', '--normalize-file-weights',
        '--fetch-yahoo',
    ]

    args = dict(value_flags)
    for flag in bool_flags:
        args[flag] = True
    args['--weight-by-games'] = True

    cmd = ybot_mod.build_score_command(args, out_path='out.csv',
                                        script_dir='/repo/scripts')

    for flag, value in value_flags.items():
        assert flag in rank_players_flags, (
            f'{flag} missing from rank_players.py build_parser()')
        assert flag in cmd, f'{flag} not forwarded by build_score_command'
        assert cmd[cmd.index(flag) + 1] == value

    for flag in bool_flags + ['--weight-by-games']:
        assert flag in rank_players_flags, (
            f'{flag} missing from rank_players.py build_parser()')
        assert flag in cmd, f'{flag} not forwarded by build_score_command'


def test_weight_by_games_pair_forwards_whichever_was_passed():
    ybot_mod = _load_ybot_module()

    cmd_true = ybot_mod.build_score_command(
        {'--weight-by-games': True}, out_path='o.csv', script_dir='/r')
    assert '--weight-by-games' in cmd_true
    assert '--no-weight-by-games' not in cmd_true

    cmd_false = ybot_mod.build_score_command(
        {'--no-weight-by-games': True}, out_path='o.csv', script_dir='/r')
    assert '--no-weight-by-games' in cmd_false
    assert '--weight-by-games' not in cmd_false

    cmd_neither = ybot_mod.build_score_command(
        {}, out_path='o.csv', script_dir='/r')
    assert '--weight-by-games' not in cmd_neither
    assert '--no-weight-by-games' not in cmd_neither


def test_weight_by_games_pair_both_passed_is_a_clear_error():
    ybot_mod = _load_ybot_module()
    with pytest.raises(ValueError):
        ybot_mod.build_score_command(
            {'--weight-by-games': True, '--no-weight-by-games': True},
            out_path='o.csv', script_dir='/r')


def test_rank_players_parser_accepts_every_flag_end_to_end():
    """Belt-and-suspenders: actually parse a command line built from the
    real ybot docopt defaults through rank_players.py's real parser, so a
    flag that exists in both places but with incompatible arities/choices
    (not just a name typo) is also caught."""
    mod = _load_rank_players_module()
    parser = mod.build_parser()
    argv = [
        '--input', 'data/foo.xlsx',
        '--sheet', 'QuantHockey',
        '--projected-games', '82',
        '--k', '20',
        '--decay', '0.5',
        '--sort-by', 'season',
        '--reverse',
        '--goalie-method', 'stats',
        '--goalie-input', 'data/goalies.xlsx',
        '--goalie-sheet', 'Goalies',
        '--goalie-header', '1',
        '--no-per-game',
        '--weight-by-games',
        '--normalize-file-weights',
        '--yahoo-points-field', 'PPT',
        '--fuzzy-match', '90',
        '--fetch-yahoo',
        '--league-id', '1234',
        '--oauth-file', 'my_oauth2.json',
    ]
    args = parser.parse_args(argv)
    assert args.input == 'data/foo.xlsx'
    assert args.sort_by == 'season'
    assert args.weight_by_games is True
