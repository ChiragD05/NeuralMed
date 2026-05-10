from services.supabase_client import supabase

def insert_data(table, data):

    return (
        supabase
        .table(table)
        .insert(data)
        .execute()
    )

def fetch_data(table):

    return (
        supabase
        .table(table)
        .select("*")
        .execute()
    )