import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from yahoo_fantasy_bot import bot, read_only


REPO_ROOT = Path(__file__).resolve().parent.parent
YBOT_PATH = REPO_ROOT / 'scripts' / 'ybot'


def test_ybot_apply_is_rejected_before_config_or_oauth_work():
    result = subprocess.run(
        [sys.executable, str(YBOT_PATH), '--apply', 'missing.cfg'],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert 'write operations are disabled' in result.stderr
    assert 'Config file does not exist' not in result.stderr


def test_manager_blocks_non_dry_run_roster_changes_before_reading_roster():
    manager = bot.ManagerBot.__new__(bot.ManagerBot)
    manager._get_orig_roster = lambda: pytest.fail('should not read roster')

    with pytest.raises(read_only.ReadOnlyOperationError,
                       match='write operations are disabled'):
        manager.apply_roster_moves(dry_run=False, prompt=False)


def test_manager_blocks_non_dry_run_trade_evaluation_before_api_call():
    manager = bot.ManagerBot.__new__(bot.ManagerBot)
    manager.tm = SimpleNamespace(
        proposed_trades=lambda: pytest.fail('should not fetch trades'))

    with pytest.raises(read_only.ReadOnlyOperationError,
                       match='write operations are disabled'):
        manager.evaluate_trades(dry_run=False, verbose=False)


def test_roster_changer_blocks_non_dry_run_before_calculating_moves():
    league = SimpleNamespace(
        team_key=lambda: 'nhl.l.1.t.1',
        to_team=lambda _team_key: SimpleNamespace(),
    )
    changer = bot.RosterChanger(league, False, [], [], [], [], 'IR', False)
    changer._calc_player_drops = lambda: pytest.fail('should not plan writes')

    with pytest.raises(read_only.ReadOnlyOperationError,
                       match='write operations are disabled'):
        changer.apply()


def test_yahoo_handler_mutations_are_blocked_but_reads_are_preserved():
    class Handler:
        def get(self, uri):
            return {'uri': uri}

        def post(self, uri, data):
            raise AssertionError('original post should be replaced')

        def put(self, uri, data):
            raise AssertionError('original put should be replaced')

        def post_transactions(self):
            return self.post('league/test/transactions', '<xml />')

        def put_roster(self):
            return self.put('team/test/roster', '<xml />')

    handler = read_only.protect_yahoo_handler(Handler())
    assert handler.get('league/test') == {'uri': 'league/test'}

    with pytest.raises(read_only.ReadOnlyOperationError):
        handler.put('team/test/roster', '<xml />')
    with pytest.raises(read_only.ReadOnlyOperationError):
        handler.post('league/test/transactions', '<xml />')
    with pytest.raises(read_only.ReadOnlyOperationError):
        handler.post_transactions()
    with pytest.raises(read_only.ReadOnlyOperationError):
        handler.put_roster()
