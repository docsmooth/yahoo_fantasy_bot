"""Tests for yahoo_fantasy_bot.utils.PlayerDetailsCache -- previously
untested (yahoo_fantasy_bot-05f).
"""
import os
import pickle

from yahoo_fantasy_bot import utils


def test_get_missing_returns_none(tmp_path):
    cache = utils.PlayerDetailsCache(str(tmp_path / "cache.pkl"))
    assert cache.get(123) is None


def test_set_and_get(tmp_path):
    cache = utils.PlayerDetailsCache(str(tmp_path / "cache.pkl"))
    cache.set(123, {"name": "Nikita Kucherov"})
    assert cache.get(123) == {"name": "Nikita Kucherov"}
    # get() coerces its argument to int, so a string id matching an int key
    # round-trips too.
    assert cache.get("123") == {"name": "Nikita Kucherov"}


def test_update_many(tmp_path):
    cache = utils.PlayerDetailsCache(str(tmp_path / "cache.pkl"))
    cache.update_many({1: {"name": "A"}, 2: {"name": "B"}})
    assert cache.get(1) == {"name": "A"}
    assert cache.get(2) == {"name": "B"}

    # update_many merges into existing data rather than replacing it.
    cache.update_many({2: {"name": "B2"}, 3: {"name": "C"}})
    assert cache.get(1) == {"name": "A"}
    assert cache.get(2) == {"name": "B2"}
    assert cache.get(3) == {"name": "C"}


def test_save_creates_missing_parent_dir(tmp_path):
    nested = tmp_path / "nested" / "dir" / "cache.pkl"
    cache = utils.PlayerDetailsCache(str(nested))
    cache.set(1, {"name": "A"})
    cache.save()

    assert nested.exists()
    with open(nested, "rb") as f:
        on_disk = pickle.load(f)
    assert on_disk == {1: {"name": "A"}}


def test_save_and_load_round_trip(tmp_path):
    cache_file = str(tmp_path / "cache.pkl")

    cache = utils.PlayerDetailsCache(cache_file)
    cache.update_many({10: {"name": "P1"}, 20: {"name": "P2"}})
    cache.save()

    reloaded = utils.PlayerDetailsCache(cache_file)
    assert reloaded.get(10) == {"name": "P1"}
    assert reloaded.get(20) == {"name": "P2"}
    assert reloaded.data == cache.data


def test_load_with_no_existing_file_starts_empty(tmp_path):
    cache_file = str(tmp_path / "does_not_exist.pkl")
    assert not os.path.exists(cache_file)
    cache = utils.PlayerDetailsCache(cache_file)
    assert cache.data == {}


def test_load_with_corrupt_file_starts_empty(tmp_path):
    cache_file = tmp_path / "corrupt.pkl"
    cache_file.write_bytes(b"not a pickle")
    cache = utils.PlayerDetailsCache(str(cache_file))
    assert cache.data == {}
