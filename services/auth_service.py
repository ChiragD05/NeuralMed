from services.supabase_client import auth_supabase

def _to_user_dict(user):
    if not user:
        return None
    metadata = getattr(user, "user_metadata", None) or {}
    return {
        "id": getattr(user, "id", None),
        "email": getattr(user, "email", None),
        "full_name": metadata.get("full_name", ""),
    }

def sign_up(email: str, password: str, full_name: str):
    return auth_supabase.auth.sign_up(
        {
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name,
                }
            },
        }
    )

def sign_in(email: str, password: str):
    return auth_supabase.auth.sign_in_with_password(
        {
            "email": email,
            "password": password,
        }
    )

def sign_out():
    return auth_supabase.auth.sign_out()

def restore_user_from_session():
    try:
        session_resp = auth_supabase.auth.get_session()
        session = getattr(session_resp, "session", None)
        if not session:
            return None

        user_resp = auth_supabase.auth.get_user()
        user = getattr(user_resp, "user", None)
        return _to_user_dict(user)
    except Exception:
        return None

def response_to_user_dict(response):
    user = getattr(response, "user", None)
    return _to_user_dict(user)