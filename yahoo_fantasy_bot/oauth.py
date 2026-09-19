"""Safe validation helpers for Yahoo OAuth credential files."""
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional


class OAuthCredentialsError(ValueError):
    """Raised when an OAuth file cannot safely be used for a Yahoo request."""


class YahooFantasyReadAccessError(RuntimeError):
    """Raised when Yahoo rejects an authenticated Fantasy Sports read."""


_BOOTSTRAP_FIELDS = ("consumer_key", "consumer_secret")
_TOKEN_FIELDS = ("access_token", "refresh_token", "token_type")
_YAHOO_READ_ACCESS_DENIALS = (
    "this application is not authorized to perform this action",
    "additional_authorization_required",
)
_YAHOO_READ_ACCESS_REMEDY = (
    "Yahoo denied this Fantasy Sports read request. The OAuth token is valid, "
    "but the app has not been granted usable Fantasy Sports read access. "
    "Confirm that Fantasy Sports - Read is enabled for this app, then "
    "re-authorize with a new OAuth file using the exact registered callback "
    "URI. If Yahoo still denies access, contact Yahoo Developer support with "
    "the app's consumer key and this authorization error; adding an OAuth "
    "scope to this client will not grant the missing entitlement."
)


def yahoo_fantasy_read_access_error(
    error: BaseException,
) -> Optional[YahooFantasyReadAccessError]:
    """Return a safe diagnostic for Yahoo's known Fantasy-read denials."""
    response = str(error).lower()
    if any(marker in response for marker in _YAHOO_READ_ACCESS_DENIALS):
        return YahooFantasyReadAccessError(_YAHOO_READ_ACCESS_REMEDY)
    return None


def validate_oauth_file(
    path: str,
    *,
    require_tokens: bool = True,
) -> Mapping[str, object]:
    """Load and validate an OAuth JSON file without ever exposing secrets."""
    credential_path = Path(path)
    try:
        with credential_path.open(encoding="utf-8") as credential_file:
            credentials = json.load(credential_file)
    except FileNotFoundError:
        raise OAuthCredentialsError(
            f"OAuth credential file does not exist: {credential_path}. "
            "Run ybot_setup to create and authorize it."
        )
    except json.JSONDecodeError as error:
        raise OAuthCredentialsError(
            f"OAuth credential file is not valid JSON: {credential_path} "
            f"(line {error.lineno}, column {error.colno}). "
            "Recreate it with ybot_setup."
        )
    except OSError as error:
        raise OAuthCredentialsError(
            f"Could not read OAuth credential file {credential_path}: {error}. "
            "Check its permissions and path."
        )

    if not isinstance(credentials, dict):
        raise OAuthCredentialsError(
            f"OAuth credential file must contain a JSON object: {credential_path}. "
            "Recreate it with ybot_setup."
        )

    required_fields: Iterable[str] = _BOOTSTRAP_FIELDS
    if require_tokens:
        required_fields = (*_BOOTSTRAP_FIELDS, *_TOKEN_FIELDS)

    invalid_fields = [
        field
        for field in required_fields
        if not isinstance(credentials.get(field), str)
        or not credentials[field].strip()
        or credentials[field].strip().startswith("REPLACE_ME")
    ]
    if invalid_fields:
        fields = ", ".join(invalid_fields)
        raise OAuthCredentialsError(
            f"OAuth credential file has missing, empty, or placeholder field(s): "
            f"{fields}. Run ybot_setup to create or re-authorize it."
        )

    return credentials
