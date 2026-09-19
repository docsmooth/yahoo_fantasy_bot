import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace


SETUP_SCRIPT = Path(__file__).parents[1] / "scripts" / "ybot_setup"
setup_loader = SourceFileLoader("ybot_setup_test", str(SETUP_SCRIPT))
setup_spec = importlib.util.spec_from_loader("ybot_setup_test", setup_loader)
ybot_setup = importlib.util.module_from_spec(setup_spec)
setup_spec.loader.exec_module(ybot_setup)


def test_fetch_leagues_uses_documented_filtered_endpoint(monkeypatch):
    wizard = ybot_setup.Wizard.__new__(ybot_setup.Wizard)
    wizard.leagues = None
    wizard.sport_code = "nhl"
    wizard.year = 2026
    wizard.gm = SimpleNamespace(
        yhandler=SimpleNamespace(
            get=lambda uri: _capture_league_request(wizard, uri),
        ),
    )
    wizard.sc = object()
    monkeypatch.setattr(ybot_setup.objectpath, "Tree", _EmptyLeagueTree)

    leagues = wizard.fetch_leagues()

    assert wizard.league_request == (
        "users;use_login=1/games;game_codes=nhl;seasons=2026/leagues"
    )
    assert leagues == []


def _capture_league_request(wizard, uri):
    wizard.league_request = uri
    return {}


class _EmptyLeagueTree:
    def __init__(self, response):
        self.response = response

    def execute(self, query):
        assert query == "$..league_key"
        return []
