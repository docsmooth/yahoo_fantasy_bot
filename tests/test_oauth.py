import json

import pytest

from yahoo_fantasy_bot.oauth import OAuthCredentialsError, validate_oauth_file


def _write_credentials(path, **overrides):
    credentials = {
        "access_token": "access-token",
        "consumer_key": "consumer-key",
        "consumer_secret": "consumer-secret",
        "refresh_token": "refresh-token",
        "token_type": "bearer",
    }
    credentials.update(overrides)
    path.write_text(json.dumps(credentials), encoding="utf-8")


def test_validate_oauth_file_accepts_complete_credentials(tmp_path):
    oauth_file = tmp_path / "oauth2.json"
    _write_credentials(oauth_file)

    credentials = validate_oauth_file(str(oauth_file))

    assert credentials["token_type"] == "bearer"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"access_token": ""}, "access_token"),
        ({"refresh_token": "REPLACE_ME_REFRESH_TOKEN"}, "refresh_token"),
        ({"consumer_key": None}, "consumer_key"),
        ({"consumer_secret": "  "}, "consumer_secret"),
        ({"token_type": ""}, "token_type"),
    ],
)
def test_validate_oauth_file_reports_bad_field_without_echoing_value(
    tmp_path, overrides, field
):
    oauth_file = tmp_path / "oauth2.json"
    _write_credentials(oauth_file, **overrides)

    with pytest.raises(OAuthCredentialsError) as error:
        validate_oauth_file(str(oauth_file))

    message = str(error.value)
    assert field in message
    assert "consumer-secret" not in message
    assert "REPLACE_ME_REFRESH_TOKEN" not in message


def test_validate_oauth_file_allows_setup_bootstrap_without_tokens(tmp_path):
    oauth_file = tmp_path / "oauth2.json"
    _write_credentials(
        oauth_file,
        access_token=None,
        refresh_token=None,
        token_type=None,
    )

    validate_oauth_file(str(oauth_file), require_tokens=False)


def test_validate_oauth_file_reports_invalid_json(tmp_path):
    oauth_file = tmp_path / "oauth2.json"
    oauth_file.write_text("{not valid JSON", encoding="utf-8")

    with pytest.raises(OAuthCredentialsError) as error:
        validate_oauth_file(str(oauth_file))

    assert "not valid JSON" in str(error.value)
