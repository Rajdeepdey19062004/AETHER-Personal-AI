"""Small JSON-backed store for local chat history and per-chat attachments."""

import json
import os
from pathlib import Path

from auth_store import connect, init_db
from user_context import get_user_data_dir, get_user_id

CHATS_PATH = Path(__file__).resolve().parent / "data" / "chats.json"


def _chats_path() -> Path:
    data_dir = get_user_data_dir()
    return CHATS_PATH if data_dir == CHATS_PATH.parent else data_dir / "chats.json"


def load_chats() -> list[dict]:
    user_id = get_user_id()
    if user_id:
        init_db()
        with connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM chats WHERE user_id = ? ORDER BY updated_at, chat_id",
                (user_id,),
            ).fetchall()
        chats = []
        for row in rows:
            try:
                item = json.loads(row["payload"])
                if isinstance(item, dict):
                    chats.append(item)
            except json.JSONDecodeError:
                continue
        return chats
    path = _chats_path()
    if not path.is_file():
        return []
    try:
        chats = json.loads(path.read_text(encoding="utf-8"))
        return chats if isinstance(chats, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_chats(chats: list[dict]) -> None:
    user_id = get_user_id()
    if user_id:
        init_db()
        with connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for chat in chats:
                chat_id = str(chat.get("id", ""))
                if not chat_id:
                    continue
                connection.execute(
                    "INSERT INTO chats (user_id, chat_id, payload, updated_at) VALUES (?, ?, ?, datetime('now')) "
                    "ON CONFLICT(user_id, chat_id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at",
                    (user_id, chat_id, json.dumps(chat, ensure_ascii=False)),
                )
            ids = {str(chat.get("id", "")) for chat in chats if chat.get("id")}
            existing_ids = connection.execute(
                "SELECT chat_id FROM chats WHERE user_id = ?", (user_id,)
            ).fetchall()
            stale_ids = [row["chat_id"] for row in existing_ids if row["chat_id"] not in ids]
            connection.executemany(
                "DELETE FROM chats WHERE user_id = ? AND chat_id = ?",
                [(user_id, chat_id) for chat_id in stale_ids],
            )
        return
    path = _chats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(chats, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
