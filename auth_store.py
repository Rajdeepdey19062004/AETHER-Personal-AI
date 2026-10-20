"""SQLite accounts with salted scrypt password hashes."""

import base64
import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DB_PATH = Path(__file__).resolve().parent / "data" / "aether.sqlite3"
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DUMMY_SALT = b"aether-login-timing-salt"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                must_change_password INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chats (
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                chat_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, chat_id)
            );
            DROP TABLE IF EXISTS invitations;
        """)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
        if "is_active" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
        if "must_change_password" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0")


def user_count() -> int:
    with connect() as connection:
        return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])


def account_status(user_id: str) -> dict | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT is_active, must_change_password FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N,
                            r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return "$".join(("scrypt", str(_SCRYPT_N), str(_SCRYPT_R), str(_SCRYPT_P),
                     base64.urlsafe_b64encode(salt).decode("ascii"),
                     base64.urlsafe_b64encode(digest).decode("ascii")))


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, n, r, p, salt_text, digest_text = stored.split("$", 5)
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n),
                                r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def create_user(username: str, password: str) -> dict:
    username = username.strip()
    if not 3 <= len(username) <= 64 or any(ord(char) < 33 for char in username):
        raise ValueError("Username must be 3–64 characters with no spaces.")
    if not 10 <= len(password) <= 128:
        raise ValueError("Password must be 10–128 characters.")
    password_hash = _hash_password(password)
    now = datetime.now(timezone.utc)
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        first_user = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        user = {"id": uuid4().hex, "username": username, "is_admin": first_user}
        try:
            connection.execute(
                "INSERT INTO users (id, username, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?, ?)",
                (user["id"], username, password_hash, int(first_user), now.isoformat()),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("That username is already taken.") from error
    return user


def authenticate(username: str, password: str) -> dict | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT id, username, password_hash, is_admin, is_active, must_change_password "
            "FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
    if row is None:
        hashlib.scrypt(password.encode("utf-8"), salt=_DUMMY_SALT, n=_SCRYPT_N,
                        r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
        return None
    if not _verify_password(password, row["password_hash"]) or not row["is_active"]:
        return None
    return {"id": row["id"], "username": row["username"],
            "is_admin": bool(row["is_admin"]),
            "must_change_password": bool(row["must_change_password"])}


def list_users(admin_id: str) -> list[dict]:
    with connect() as connection:
        admin = connection.execute("SELECT is_admin FROM users WHERE id = ? AND is_active = 1", (admin_id,)).fetchone()
        if not admin or not admin["is_admin"]:
            raise PermissionError("Admin access is required.")
        rows = connection.execute(
            "SELECT id, username, is_admin, is_active, created_at FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(row) for row in rows]


def set_user_active(admin_id: str, user_id: str, active: bool) -> None:
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        admin = connection.execute("SELECT is_admin FROM users WHERE id = ? AND is_active = 1", (admin_id,)).fetchone()
        target = connection.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
        if not admin or not admin["is_admin"]:
            raise PermissionError("Admin access is required.")
        if not target or target["is_admin"] or user_id == admin_id:
            raise ValueError("The owner account cannot be deactivated here.")
        connection.execute("UPDATE users SET is_active = ? WHERE id = ?", (int(active), user_id))


def set_temporary_password(admin_id: str, user_id: str) -> str:
    temporary_password = secrets.token_urlsafe(15)
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        admin = connection.execute("SELECT is_admin FROM users WHERE id = ? AND is_active = 1", (admin_id,)).fetchone()
        target = connection.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
        if not admin or not admin["is_admin"]:
            raise PermissionError("Admin access is required.")
        if not target or target["is_admin"] or user_id == admin_id:
            raise ValueError("Choose a regular user account to reset.")
        connection.execute(
            "UPDATE users SET password_hash = ?, must_change_password = 1 WHERE id = ?",
            (_hash_password(temporary_password), user_id),
        )
    return temporary_password


def change_password(user_id: str, current_password: str, new_password: str) -> None:
    if not 10 <= len(new_password) <= 128:
        raise ValueError("New password must be 10–128 characters.")
    with connect() as connection:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE id = ? AND is_active = 1", (user_id,)
        ).fetchone()
        if not row or not _verify_password(current_password, row["password_hash"]):
            raise ValueError("Current password is incorrect.")
        connection.execute(
            "UPDATE users SET password_hash = ?, must_change_password = 0 WHERE id = ?",
            (_hash_password(new_password), user_id),
        )
