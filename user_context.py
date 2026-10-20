"""Per-session identity scope for user-owned AETHER data."""

from contextvars import ContextVar
import hashlib
from pathlib import Path

_USER_ID = ContextVar("aether_user_id", default=None)
DATA_DIR = Path(__file__).resolve().parent / "data"


def set_user_id(user_id: str) -> None:
    _USER_ID.set(str(user_id))


def get_user_id() -> str | None:
    return _USER_ID.get()


def get_user_data_dir() -> Path:
    user_id = get_user_id()
    if not user_id:
        return DATA_DIR
    # Never use provider-controlled identifiers directly as filesystem paths.
    scope = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]
    return DATA_DIR / "users" / scope
