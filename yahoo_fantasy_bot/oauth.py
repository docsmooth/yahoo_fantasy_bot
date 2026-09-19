"""Safe validation helpers for Yahoo OAuth credential files."""
import json
from pathlib import Path
from typing import Iterable, Mapping


class OAuthCredentialsError(ValueError):
    """Raised when an OAuth file cannot safely be used for a Yahoo request."""


_BOOTSTRAP_FIELDS = ("consumer_key", "consumer_secret")
_TOKEN_FIELDS = ("access_token", "refresh_token", "token_type")


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
