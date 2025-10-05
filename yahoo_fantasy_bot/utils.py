#!/usr/bin/python

import unicodedata
import os
import logging
import pickle
import datetime


def normalized(name):
    """Normalize a name to remove any accents

    :param name: Input name to normalize
    :type name: str
    :return: Normalized name
    :rtype: str
    """
    return unicodedata.normalize('NFD', name).encode(
        'ascii', 'ignore').decode('utf-8')


class CacheBase(object):
    def __init__(self, cfg, cache_dir):
        self.logger = logging.getLogger()
        self.cfg = cfg
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def run_loader(self, fn, expiry, loader):
        cached_data = None

        if os.path.exists(fn):
            with open(fn, "rb") as f:
                cached_data = pickle.load(f)
            if type(cached_data) != dict or "expiry" not in cached_data or \
                    "payload" not in cached_data:
                cached_data = None
            elif cached_data["expiry"] is not None:
                if datetime.datetime.now() > cached_data["expiry"]:
                    self.logger.info(
                        "{} file is stale.  Expired at {}".
                        format(fn, cached_data["expiry"]))
                    cached_data = None

        if cached_data is None:
            self.logger.info("Building new {} file".format(fn))
            cached_data = {}
            cached_data["payload"] = loader()
            if expiry is not None:
                cached_data["expiry"] = datetime.datetime.now() + expiry
            else:
                cached_data["expiry"] = None
            with open(fn, "wb") as f:
                pickle.dump(cached_data, f)
            self.logger.info("Finished building {} file".format(fn))

        return cached_data["payload"]


class TeamCache(CacheBase):
    def __init__(self, cfg, team_key):
        super(TeamCache, self).__init__(
            cfg, "{}/{}/{}".format(cfg['Cache']['dir'], cfg['League']['id'],
                                   team_key))

    def prediction_builder_file(self):
        return "{}/pred_builder.pkl".format(self.cache_dir)

    def load_prediction_builder(self, expiry, loader):
        return self.run_loader(self.prediction_builder_file(), expiry, loader)

    def league_lineup_file(self):
        return "{}/lg_lineups.pkl".format(self.cache_dir)

    def load_league_lineup(self, expiry, loader):
        return self.run_loader(self.league_lineup_file(), expiry, loader)

    def free_agents_cache_file(self):
        return "{}/free_agents.pkl".format(self.cache_dir)

    def load_free_agents(self, expiry, loader):
        return self.run_loader(self.free_agents_cache_file(), expiry, loader)

    def remove(self):
        for fn in [self.prediction_builder_file(),
                   self.league_lineup_file(), self.free_agents_cache_file()]:
            if os.path.exists(fn):
                os.remove(fn)


class LeagueCache(CacheBase):
    def __init__(self, cfg):
        super(LeagueCache, self).__init__(
            cfg, "{}/{}".format(cfg['Cache']['dir'], cfg['League']['id']))

    def statics(self):
        return "{}/league_statics.pkl".format(self.cache_dir)

    def load_statics(self, loader):
        return self.run_loader(self.statics(), None, loader)

    def remove(self):
        for fn in [self.statics()]:
            if os.path.exists(fn):
                os.remove(fn)


class PlayerDetailsCache(object):
    """Simple file-backed cache for player details keyed by player_id.

    Stores a dict {player_id: details} in a pickle file. Provides get() and
    update_many() helpers to reduce API calls.
    """
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                self.data = {}
        except Exception:
            self.data = {}

    def get(self, player_id):
        return self.data.get(int(player_id))

    def set(self, player_id, details):
        self.data[int(player_id)] = details

    def update_many(self, mapping):
        """mapping: dict(player_id -> details)
        """
        for k, v in mapping.items():
            self.data[int(k)] = v

    def save(self):
        try:
            ddir = os.path.dirname(self.cache_file)
            if ddir and not os.path.exists(ddir):
                os.makedirs(ddir, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data, f)
        except Exception:
            pass
