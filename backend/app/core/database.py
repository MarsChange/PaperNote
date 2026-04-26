import aiosqlite
from pathlib import Path

from app.core.config import settings

DB_PATH = str(settings.db_path)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    markdown_path TEXT,
    content_list_path TEXT,
    assets_dir TEXT,
    status TEXT NOT NULL DEFAULT 'uploading',
    summary TEXT,
    keywords TEXT,
    page_count INTEGER,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS annotations (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    color TEXT NOT NULL DEFAULT '#fef08a',
    start_offset INTEGER,
    end_offset INTEGER,
    note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conversations_paper ON conversations(paper_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_annotations_paper ON annotations(paper_id);
"""


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    try:
        await db.executescript(SCHEMA_SQL)
        await _run_migrations(db)
        await db.commit()
    finally:
        await db.close()


async def _run_migrations(db: aiosqlite.Connection):
    table_columns = {
        "papers": {
            "content_list_path": "TEXT",
            "assets_dir": "TEXT",
            "metadata_json": "TEXT",
        },
        "messages": {
            "metadata_json": "TEXT",
        },
        "annotations": {
            "note": "TEXT",
        },
    }

    for table_name, columns in table_columns.items():
        cursor = await db.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in await cursor.fetchall()}
        for column_name, column_type in columns.items():
            if column_name not in existing_columns:
                await db.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                )
