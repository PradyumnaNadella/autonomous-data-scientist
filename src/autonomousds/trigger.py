"""Where a run was started from: the command line, the web API, or GitHub Actions.

Set by the AUTONOMOUSDS_TRIGGER environment variable; defaults to "local".
Child processes inherit it, so the API and the Actions workflow set it once.
"""
import os

TRIGGER_ENV = "AUTONOMOUSDS_TRIGGER"
TRIGGERS = ("local", "api", "actions")
RUN_ID_PREFIXES = {"actions": "gha-"}      # only Actions runs get a visible prefix


def current_trigger() -> str:
    """Return where the current run was started from.

    Returns:
        "local", "api" or "actions" (unknown values count as "local").
    """
    value = os.getenv(TRIGGER_ENV, "local")
    return value if value in TRIGGERS else "local"


def run_id_prefix() -> str:
    """Return the run id prefix for the current trigger.

    Returns:
        "gha-" for GitHub Actions runs, otherwise an empty string.
    """
    return RUN_ID_PREFIXES.get(current_trigger(), "")