import uuid
from supabase import Client


class ConversationStore:
    def create_session(self, db: Client, title: str) -> dict:
        session = {
            "public_id": uuid.uuid4().hex[:16],
            "title": title
        }
        result = db.table("conversations").insert(session).execute()
        return result.data[0]

    def list_sessions(self, db: Client) -> list:
        result = (
            db.table("conversations")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return result.data

    def get_session(self, db: Client, session_id: str) -> dict | None:
        result = (
            db.table("conversations")
            .select("*")
            .eq("public_id", session_id)
            .single()
            .execute()
        )
        return result.data

    def add_message(self, db: Client, session_id: str, role: str, content: str) -> dict:
        message = {
            "session_id": session_id,
            "role": role,
            "content": content
        }
        result = db.table("messages").insert(message).execute()
        return result.data[0]

    def delete_session(self, db: Client, session_id: str) -> None:
        db.table("messages").delete().eq("session_id", session_id).execute()
        db.table("conversations").delete().eq("public_id", session_id).execute()

    def get_messages(self, db: Client, session_id: str) -> list:
        result = (
            db.table("messages")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at")
            .execute()
        )
        return result.data
        
