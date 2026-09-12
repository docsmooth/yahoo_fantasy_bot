"""Tests for scripts/ybot's docopt Options block and command-building.

scripts/ybot transitively imports yahoo_oauth/yahoo_fantasy_api via
yahoo_fantasy_bot.automation, which are not installed in this environment.
To avoid that dependency, these tests either:

  * parse the docstring directly out of the file with docopt, or
  * load scripts/ybot as a module via importlib.  Because the automation
    import lives inside the `if __name__ == '__main__':` guard (and is not
    executed when the file is merely imported/exec'd as a module), this
    works without yahoo_oauth/yahoo_fantasy_api being present.
"""
import importlib.util
import re
from importlib.machinery import SourceFileLoader
from pathlib import Path

import docopt

REPO_ROOT = Path(__file__).resolve().parent.parent
YBOT_PATH = REPO_ROOT / 'scripts' / 'ybot'


def _load_doc():
    text = YBOT_PATH.read_text()
    return text.split('"""')[1]


def _load_ybot_module():
    loader = SourceFileLoader('ybot_cli_under_test', str(YBOT_PATH))
    spec = importlib.util.spec_from_loader('ybot_cli_under_test', loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_score_with_cfg_file_parses():
    doc = _load_doc()
    args = docopt.docopt(doc, argv=['--score', 'my.cfg'])
    assert args['<cfg_file>'] == 'my.cfg'
    assert args['--score'] is True


def test_goalie_method_is_registered_and_takes_one_arg():
    doc = _load_doc()
    opts = docopt.parse_defaults(doc)
    goalie_opts = [o for o in opts if o.long == '--goalie-method']
    assert len(goalie_opts) == 1
    assert goalie_opts[0].argcount == 1

    # And it actually works when parsed end to end.
    args = docopt.docopt(
        doc, argv=['--score', '--goalie-method', 'stats', 'my.cfg'])
    assert args['--goalie-method'] == 'stats'


def test_score_registered_exactly_once_with_no_argument():
    doc = _load_doc()
    opts = docopt.parse_defaults(doc)
    score_opts = [o for o in opts if o.long == '--score']
    assert len(score_opts) == 1
    assert score_opts[0].argcount == 0


def _docopt_option_chunks(doc):
    """Reproduce docopt.parse_defaults' own chunking so this test checks the
    exact same boundaries docopt itself uses, rather than a re-guessed
    heuristic."""
    split = re.split(r'\n *(<\S+?>|-\S+?)', doc)[1:]
    chunks = [s1 + s2 for s1, s2 in zip(split[::2], split[1::2])]
    return [c for c in chunks if c.startswith('-')]


def test_every_option_line_is_well_formed():
    """Guard against the docopt mis-split regression (yahoo_fantasy_bot-ulo).

    docopt.Option.parse() splits each option chunk on the FIRST run of two
    or more spaces: everything before that is the "spec" (scanned token by
    token for -x/--flag forms), everything after is free-text description.
    If a line only has a single space before its description, the spec
    swallows the whole description too -- so any English words (or another
    '--flag' mentioned in the prose) get parsed as if they were part of the
    spec. This test asserts that never happens, for every option in the
    Options block.
    """
    doc = _load_doc()
    chunks = _docopt_option_chunks(doc)
    assert len(chunks) >= 20, 'sanity check: expected many options'

    for chunk in chunks:
        prefix, _, _description = chunk.partition('  ')
        tokens = prefix.replace(',', ' ').replace('=', ' ').split()
        assert tokens, f'empty option spec parsed from chunk: {chunk!r}'
        # At most one non-flag token is allowed: the argument placeholder
        # (e.g. '<m>' or a bare 'x'), which is how docopt decides the option
        # takes a value. Two or more means prose bled into the spec because
        # of a missing double-space before the description.
        non_flag_tokens = [t for t in tokens if not t.startswith('-')]
        assert len(non_flag_tokens) <= 1, (
            f'option spec has {len(non_flag_tokens)} non-flag tokens '
            f'{non_flag_tokens!r} -- description text bled into the spec '
            f'because of a missing double-space before it. Offending '
            f'chunk: {chunk!r}'
        )
        long_flags = [t for t in tokens if t.startswith('--')]
        assert len(long_flags) <= 1, (
            f'more than one long flag parsed out of a single option spec '
            f'(docopt silently keeps only the last one): {long_flags!r} '
            f'from chunk {chunk!r}'
        )


def test_build_score_command_without_no_per_game():
    mod = _load_ybot_module()
    args = {
        '--goalie-method': 'gp-fallback',
        '--sort-by': 'mtime',
        '--reverse': True,
        '--decay': '0.5',
        '--no-per-game': False,
        '--rank-extra': '--top 50',
    }
    cmd = mod.build_score_command(args, out_path='scored.csv',
                                   script_dir='/repo/scripts')

    assert cmd[0]  # sys.executable, non-empty
    assert cmd[1] == '/repo/scripts/rank_players.py'
    assert '--out' in cmd and cmd[cmd.index('--out') + 1] == 'scored.csv'
    assert '--goalie-method' in cmd
    assert cmd[cmd.index('--goalie-method') + 1] == 'gp-fallback'
    assert '--sort-by' in cmd
    assert cmd[cmd.index('--sort-by') + 1] == 'mtime'
    assert '--reverse' in cmd
    assert '--decay' in cmd
    assert cmd[cmd.index('--decay') + 1] == '0.5'
    # --no-per-game was False: must NOT be forwarded, and the command must
    # still be built/populated without it (regression test for
    # yahoo_fantasy_bot-gor, where the whole block only ran under
    # --no-per-game).
    assert '--no-per-game' not in cmd
    assert '--top' in cmd and '50' in cmd


def test_build_score_command_defaults_use_sys_executable_and_script_dir(monkeypatch):
    import sys
    mod = _load_ybot_module()
    cmd = mod.build_score_command({}, out_path='out.csv')
    assert cmd[0] == sys.executable
    assert cmd[1] == str(YBOT_PATH.parent / 'rank_players.py')


def test_ybot_module_importable_without_score_flag_triggering_automation():
    # Regression guard: importing scripts/ybot must not require
    # yahoo_oauth/yahoo_fantasy_api. If someone moves the automation import
    # back to module scope, this import will start raising ImportError in
    # this environment.
    mod = _load_ybot_module()
    assert hasattr(mod, 'build_score_command')
