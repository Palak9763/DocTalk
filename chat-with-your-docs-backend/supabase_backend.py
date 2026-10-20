"""Supabase Cloud Backend Adapter for DocTalk.

Provides seamless integration with:
1. Supabase PostgreSQL: `users` table
2. Supabase Storage: `doctalk-documents` bucket for uploaded files
3. Supabase pgvector: `document_chunks` table and `match_document_chunks` RPC search

If SUPABASE_URL or SUPABASE_KEY are unset or invalid, operations smoothly fallback to local SQLite.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "doctalk-documents").strip()

_supabase_client = None


def is_supabase_configured() -> bool:
    """Check if Supabase credentials are validly supplied."""
    return bool(
        SUPABASE_URL
        and SUPABASE_KEY
        and not SUPABASE_KEY.startswith("PASTE_YOUR")
        and not SUPABASE_KEY.startswith("your-")
    )


def get_supabase():
    """Return an active Supabase Client, or None if unconfigured."""
    global _supabase_client
    if not is_supabase_configured():
        return None

    if _supabase_client is None:
        try:
            from supabase import create_client
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Connected to Supabase project at %s", SUPABASE_URL)
        except Exception as exc:
            logger.error("Failed to initialize Supabase client: %s", exc)
            return None
    return _supabase_client


# ---------------------------------------------------------------------------
# Database: Users Operations
# ---------------------------------------------------------------------------
def sb_get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    client = get_supabase()
    if not client:
        return None
    try:
        res = client.table("users").select("*").eq("email", email.strip().lower()).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as exc:
        logger.warning("Supabase get_user_by_email error: %s", exc)
        return None


def sb_get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    client = get_supabase()
    if not client:
        return None
    try:
        res = client.table("users").select("*").eq("id", user_id).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as exc:
        logger.warning("Supabase get_user_by_id error: %s", exc)
        return None


def sb_create_user(user_data: dict[str, Any]) -> bool:
    client = get_supabase()
    if not client:
        return False
    try:
        client.table("users").insert(user_data).execute()
        return True
    except Exception as exc:
        logger.warning("Supabase create_user error: %s", exc)
        return False


def sb_update_user_password(user_id: str, new_hash: str) -> bool:
    client = get_supabase()
    if not client:
        return False
    try:
        client.table("users").update({"password_hash": new_hash}).eq("id", user_id).execute()
        return True
    except Exception as exc:
        logger.warning("Supabase update_user_password error: %s", exc)
        return False


def sb_update_last_login(user_id: str, timestamp: str) -> None:
    client = get_supabase()
    if not client:
        return
    try:
        client.table("users").update({"last_login": timestamp}).eq("id", user_id).execute()
    except Exception as exc:
        logger.warning("Supabase update_last_login error: %s", exc)


def sb_list_users() -> list[dict[str, Any]]:
    client = get_supabase()
    if not client:
        return []
    try:
        res = client.table("users").select("id, email, name, role, created_at, last_login").order("created_at", desc=True).execute()
        return res.data or []
    except Exception as exc:
        logger.warning("Supabase list_users error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Storage: Document Uploads & Retrieval
# ---------------------------------------------------------------------------
def sb_upload_file(user_id: str, filename: str, content: bytes, content_type: str = "application/octet-stream") -> bool:
    client = get_supabase()
    if not client:
        return False
    storage_path = f"{user_id}/{filename}"
    try:
        # Upsert file into the user's folder in Supabase Storage
        client.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path,
            file=content,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        return True
    except Exception as exc:
        logger.warning("Supabase upload_file error: %s", exc)
        return False


def sb_delete_file(user_id: str, filename: str) -> bool:
    client = get_supabase()
    if not client:
        return False
    storage_path = f"{user_id}/{filename}"
    try:
        client.storage.from_(SUPABASE_BUCKET).remove([storage_path])
        return True
    except Exception as exc:
        logger.warning("Supabase delete_file error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Vector Search: Document Chunks & Cosine Similarity
# ---------------------------------------------------------------------------
def sb_save_chunks(user_id: str, doc_name: str, chunks_data: list[dict[str, Any]]) -> bool:
    client = get_supabase()
    if not client or not chunks_data:
        return False
    try:
        # Delete existing chunks for this file if re-indexing
        client.table("document_chunks").delete().eq("user_id", user_id).eq("doc_name", doc_name).execute()
        # Batch insert chunks with their dense embeddings
        client.table("document_chunks").insert(chunks_data).execute()
        return True
    except Exception as exc:
        logger.warning("Supabase save_chunks error: %s", exc)
        return False


def sb_match_chunks(user_id: str, query_embedding: list[float], limit: int = 30) -> list[dict[str, Any]]:
    client = get_supabase()
    if not client:
        return []
    try:
        res = client.rpc("match_document_chunks", {
            "p_user_id": user_id,
            "query_embedding": query_embedding,
            "match_threshold": 0.0,
            "match_count": limit
        }).execute()
        return res.data or []
    except Exception as exc:
        logger.warning("Supabase match_document_chunks RPC error: %s", exc)
        return []
