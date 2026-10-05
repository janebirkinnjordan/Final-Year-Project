from supabase import Client
from .supabase_client import get_supabase
from fastapi import Depends

def get_db(db: Client = Depends(get_supabase)) -> Client:
    return db