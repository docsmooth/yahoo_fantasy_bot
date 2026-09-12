#!/usr/bin/python

import configparser
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from yahoo_fantasy_bot import roster

RBLDR_COLS = ["player_id", "name", "eligible_positions", "selected_position"]
RSEL_COLS = ["player_id", "name", "HR", "OBP", "W", "ERA"]

FIXTURES_DIR = Path(__file__).parent / "fixtures"
QUANTHOCKEY_SAMPLE_PATH = FIXTURES_DIR / "quanthockey_sample.xlsx"
QUANTHOCKEY_GOALIE_SAMPLE_PATH = FIXTURES_DIR / "quanthockey_goalie_sample.xlsx"


@pytest.fixture
def quanthockey_sample_path() -> Path:
    """Path to the committed real-schema QuantHockey sample workbook.

    See tests/fixtures/make_fixture.py for how it was generated and what
    it contains (4 forwards, 2 defencemen, 2 low-GP call-ups, 4 goalies).
    """
    return QUANTHOCKEY_SAMPLE_PATH


@pytest.fixture
def quanthockey_sample_df(quanthockey_sample_path) -> pd.DataFrame:
    """The QuantHockey sample workbook loaded as a DataFrame.

    Loaded exactly as production code loads real QuantHockey exports:
    sheet_name='QuantHockey', header=1 (one banner row above the real
    column header row).
    """
    return pd.read_excel(quanthockey_sample_path, sheet_name="QuantHockey", header=1)


@pytest.fixture
def quanthockey_goalie_sample_path() -> Path:
    """Path to a hand-built goalie-stats export (W/GA/SV/SO columns).

    There is no real QuantHockey goalie export in data/ yet
    (yahoo_fantasy_bot-nr7 is the blocker), so this fixture stands in for
    one to fixture-test the yahoo_fantasy_bot-fow --goalie-input path. See
    tests/fixtures/make_goalie_fixture.py for how it was generated and what
    it contains (the same four goalies as quanthockey_sample.xlsx, plus one
    goalie absent from the skater sample to exercise the unmatched path).
    """
    return QUANTHOCKEY_GOALIE_SAMPLE_PATH


@pytest.fixture
def quanthockey_goalie_sample_df(quanthockey_goalie_sample_path) -> pd.DataFrame:
    """The goalie-stats sample workbook loaded as a DataFrame."""
    return pd.read_excel(quanthockey_goalie_sample_path, sheet_name="QuantHockey", header=1)


def _minimal_roster_cfg():
    """Build the smallest configparser that roster.Container() needs today.

    roster.Container.__init__ resolves a StatAccumulator class via
    cfg['ScoreAccumulator'] and constructs it. The real implementation
    (yahoo_fantasy_bot.nhl.StatAccumulator) pulls in the nhl_scraper
    package, which isn't installed in this test environment and isn't
    needed to exercise roster.Builder/Container position-fitting logic, so
    we point at the no-op tests/fake_score_accumulator.py stub instead.
    """
    cfg = configparser.RawConfigParser(
        converters={'list': lambda x: [i.strip() for i in x.split(',')]})
    cfg['League'] = {'predictedStatCategories': 'G,A,P'}
    cfg['Scorer'] = {'useWeeklySchedule': 'false'}
    cfg['ScoreAccumulator'] = {
        'package': '',
        'module': 'fake_score_accumulator',
        'class': 'StatAccumulator',
    }
    return cfg


@pytest.fixture
def empty_roster():
    rcont = roster.Container(_minimal_roster_cfg())
    yield rcont


@pytest.fixture
def bldr():
    b = roster.Builder(["C", "1B", "2B", "SS", "3B", "LF", "CF", "RF", "Util",
                        "SP", "SP", "SP", "SP", "SP",
                        "RP", "RP", "RP", "RP", "RP"])
    yield b


@pytest.fixture
def fake_player_selector():
    player_pool = pd.DataFrame(
        [[1, "Borders", 15, 0.319, np.nan, np.nan],
         [2, "Lee", 6, 0.288, np.nan, np.nan],
         [3, "McGriff", 35, 0.400, np.nan, np.nan],
         [4, "Fernandez", 4, 0.352, np.nan, np.nan],
         [5, "Gruber", 31, 0.330, np.nan, np.nan],
         [6, "Bell", 21, 0.303, np.nan, np.nan],
         [7, "Wilson", 3, 0.300, np.nan, np.nan],
         [8, "Felix", 15, 0.318, np.nan, np.nan],
         [9, "Olerud", 14, 0.364, np.nan, np.nan],
         [10, "Hill", 12, 0.281, np.nan, np.nan],
         [11, "Steib", np.nan, np.nan, 18, 2.93],
         [12, "Stottlemyre", np.nan, np.nan, 13, 4.34],
         [13, "Wells", np.nan, np.nan, 11, 3.14],
         [14, "Key", np.nan, np.nan, 13, 4.25],
         [15, "Cerutti", np.nan, np.nan, 9, 4.76]], columns=RSEL_COLS)
    plyr_sel = roster.PlayerSelector(player_pool)
    yield plyr_sel
