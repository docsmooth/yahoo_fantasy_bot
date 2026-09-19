"""Guards that enforce the bot's read-only Yahoo Fantasy mode."""

READ_ONLY_MESSAGE = (
    "Yahoo Fantasy write operations are disabled: this application runs in "
    "read-only mode."
)


class ReadOnlyOperationError(RuntimeError):
    """Raised when code attempts to change Yahoo Fantasy state."""


def reject_write(*_args, **_kwargs):
    """Reject a Yahoo API request that would change Fantasy state."""
    raise ReadOnlyOperationError(READ_ONLY_MESSAGE)


def protect_yahoo_handler(handler):
    """Disable PUT and POST on a Yahoo API handler used by this bot.

    The library's higher-level transaction helpers dispatch through these
    instance methods, so this protects normal bot flow even if a caller
    bypasses the roster and trade guards.
    """
    handler.put = reject_write
    handler.post = reject_write
    return handler
