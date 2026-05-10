import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL missing in .env")

if not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_ANON_KEY missing in .env")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY missing in .env")

auth_supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
admin_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)