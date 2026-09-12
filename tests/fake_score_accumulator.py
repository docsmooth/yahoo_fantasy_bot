"""Minimal stand-in for a ScoreAccumulator implementation, used by
tests/conftest.py's `empty_roster` fixture.

roster.Container(cfg) resolves its StatAccumulator class dynamically via
importlib, pointed at cfg['ScoreAccumulator']['module']/['package']/['class'].
The real implementation (yahoo_fantasy_bot.nhl.StatAccumulator) pulls in the
nhl_scraper package, which is not part of this project's test dependencies
and isn't needed to exercise the position-fitting logic under test in
tests/test_roster.py. This module implements the same no-op interface
(add_player/remove_player/get_summary) without that dependency.
"""
import pandas as pd


class StatAccumulator:
    def __init__(self, cfg):
        self.cfg = cfg

    def add_player(self, plyr):
        pass

    def remove_player(self, plyr):
        pass

    def get_summary(self, roster):
        return pd.DataFrame(data=roster)
