import pathlib
import json
from supabase import create_client, Client

# ────────────────────── TOKEN & LOGGING ──────────────────────────────
TOKENS_FILE = pathlib.Path(__file__).with_name("tokens.json")
try:
    TOKEN: str = json.loads(TOKENS_FILE.read_text())["discord"]
    TOGETHER_TOKEN: str = json.loads(TOKENS_FILE.read_text())["together"]
    SUPABASE_ANON_KEY: str = json.loads(TOKENS_FILE.read_text())["supabase_anon_key"]
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        f"❌ Could not read Discord or Together token from {TOKENS_FILE}"
    ) from exc

# ────────────────────── SUPABASE CLIENT ──────────────────────────────
SUPABASE_URL = "https://ytxivxdkceidiownvyhj.supabase.co"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ────────────────────── SUPABASE TABLES ──────────────────────────────

# Table: users
# Stores user information and violation counts
"""
create table users (
    id bigint primary key,  -- Discord user ID
    username text not null,
    violation_count int default 0,
    last_violation_at timestamp with time zone,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);
"""

# Table: reports
# Stores all moderation reports
"""
create table reports (
    id bigint primary key generated always as identity,
    reporter_id bigint not null,  -- Discord user ID of reporter
    target_user_id bigint not null references users(id),
    guild_id bigint not null,
    channel_id bigint not null,
    message_id bigint not null,
    reason text not null,  -- e.g., 'SEXUAL_EXPLOITATION', 'GROOMING', etc.
    subcategory text,
    confidence float,  -- For automated detection
    evidence_text text,
    attachment_urls text[],  -- Array of URLs
    reporter_wants_block boolean default false,
    outcome text,  -- e.g., 'REMOVE_MESSAGE', 'WARN_USER', etc.
    resolved_by bigint,  -- Discord user ID of moderator who resolved
    resolved_at timestamp with time zone,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);
"""

# Helper functions for database operations
async def ensure_user_exists(user_id: int, username: str):
    """Ensure a user exists in the database, create if they don't."""
    result = supabase.table("users").select("id").eq("id", user_id).execute()
    if not result.data:
        supabase.table("users").insert({
            "id": user_id,
            "username": username,
            "violation_count": 0
        }).execute()

async def get_user_violation_count(user_id: int) -> int:
    """Get the number of violations for a user."""
    result = supabase.table("users").select("violation_count").eq("id", user_id).execute()
    return result.data[0]["violation_count"] if result.data else 0

async def increment_user_violation_count(user_id: int, username: str):
    """Increment violation count for a user, creating user if doesn't exist."""
    await ensure_user_exists(user_id, username)
    # First get current count
    result = supabase.table("users").select("violation_count").eq("id", user_id).execute()
    current_count = result.data[0]["violation_count"] if result.data else 0
    
    # Then update with incremented count
    supabase.table("users").upsert({
        "id": user_id,
        "username": username,
        "violation_count": current_count + 1,
        "last_violation_at": "now()"
    }).execute()

async def create_report(
    reporter_id: int,
    reporter_name: str,
    target_user_id: int,
    target_username: str,
    guild_id: int,
    channel_id: int,
    message_id: int,
    reason: str,
    subcategory: str = None,
    confidence: float = None,
    evidence_text: str = None,
    attachment_urls: list = None,
    reporter_wants_block: bool = False
) -> dict:
    """Create a new report in the database."""
    # Ensure both users exist
    await ensure_user_exists(reporter_id, reporter_name)
    await ensure_user_exists(target_user_id, target_username)
    
    report_data = {
        "reporter_id": reporter_id,
        "target_user_id": target_user_id,
        "guild_id": guild_id,
        "channel_id": channel_id,
        "message_id": message_id,
        "reason": reason,
        "subcategory": subcategory,
        "confidence": confidence,
        "evidence_text": evidence_text,
        "attachment_urls": attachment_urls,
        "reporter_wants_block": reporter_wants_block
    }
    result = supabase.table("reports").insert(report_data).execute()
    return result.data[0]

async def resolve_report(report_id: int, outcome: str, resolved_by: int, resolved_by_name: str):
    """Mark a report as resolved with the given outcome."""
    await ensure_user_exists(resolved_by, resolved_by_name)
    supabase.table("reports").update({
        "outcome": outcome,
        "resolved_by": resolved_by,
        "resolved_at": "now()"
    }).eq("id", report_id).execute()

async def get_top_offenders(limit: int = 10, exclude_bot_id: int = None) -> list:
    """Get top offenders by violation count."""
    query = supabase.table("users").select("id,username,violation_count,last_violation_at").order("violation_count", desc=True)
    
    if exclude_bot_id:
        query = query.neq("id", exclude_bot_id)
    
    result = query.limit(limit).execute()
    return result.data


