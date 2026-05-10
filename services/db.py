from services.supabase_client import admin_supabase
from services.user_context import get_auth_user

def _attach_user_metadata(payload: dict):
    data = dict(payload)
    user = get_auth_user()

    if user:
        data.setdefault("app_user_id", user.get("id"))
        data.setdefault("app_user_email", user.get("email"))

    return data

def insert_data(table, data):
    return admin_supabase.table(table).insert(_attach_user_metadata(data)).execute()

def fetch_my_rows(table, limit=10):
    query = admin_supabase.table(table).select("*")
    user = get_auth_user()

    if user and user.get("email"):
        query = query.eq("app_user_email", user["email"])

    return query.order("created_at", desc=True).limit(limit).execute()

def fetch_all_rows(table):
    return admin_supabase.table(table).select("*").execute()

def count_rows(table):
    response = admin_supabase.table(table).select("*", count="exact").execute()
    return response.count if hasattr(response, "count") else 0