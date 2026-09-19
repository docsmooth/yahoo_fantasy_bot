import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace


SETUP_SCRIPT = Path(__file__).parents[1] / "scripts" / "ybot_setup"
setup_loader = SourceFileLoader("ybot_setup_test", str(SETUP_SCRIPT))
setup_spec = importlib.util.spec_from_loader("ybot_setup_test", setup_loader)
ybot_setup = importlib.util.module_from_spec(setup_spec)
setup_spec.loader.exec_module(ybot_setup)


def test_fetch_leagues_uses_supported_filtered_endpoint():
    wizard = ybot_setup.Wizard.__new__(ybot_setup.Wizard)
    wizard.leagues = None
    wizard.sport_code = "nhl"
    wizard.year = 2026
    wizard.gm = SimpleNamespace(
        league_ids=lambda **kwargs: _capture_league_ids(wizard, kwargs),
    )
    wizard.sc = object()

    leagues = wizard.fetch_leagues()

    assert wizard.league_id_arguments == {
        "game_codes": ["nhl"],
        "seasons": ["2026"],
    }
    assert leagues == []


def _capture_league_ids(wizard, kwargs):
    wizard.league_id_arguments = kwargs
    return []
