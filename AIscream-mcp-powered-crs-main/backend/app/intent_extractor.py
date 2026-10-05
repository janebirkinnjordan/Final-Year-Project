import json
from typing import Any


INTENT_SYSTEM_PROMPT = """
You are an intent parser for a multi-domain conversational recommender system.

Convert the user request into a JSON object for API retrieval.

Supported target domains:
- movies
- books
- music

Rules:
- Do not recommend items.
- Do not invent titles.
- Extract only retrieval intent.
- If the user asks for latest/new/recent/current/released recently, set recent=true.
- If the user says "before 2020", "2020 and before", or "up to 2020", set year_max=2020.
- If the user says "after 2020", "since 2020", or "from 2020", set year_min=2020.
- For country/language requests, infer ISO codes when obvious.
  Examples:
  Malay/Malaysian/Malaysia -> country MY, language ms if language is relevant.
  Korean/Korea -> country KR, language ko.
  Japanese/Japan -> country JP, language ja.
- For movie person queries, use person_reference, actor_reference, or director_reference.
- For book author queries, use author_reference.
- For music artist/singer/band queries, use artist_reference.
- For cross-domain vibe queries, create a clean query using transferable style/mood/theme terms.
- Keep limit between 1 and 10.
- Return valid JSON only.

JSON fields:
{
  "target_domain": "",
  "source_domain": "",
  "query": "",
  "limit": 5,
  "recent": false,
  "sort_by": "",
  "year_min": 0,
  "year_max": 0,
  "decade": "",
  "genre": "",
  "mood": "",
  "theme": "",
  "audience": "",
  "country": "",
  "language": "",
  "person_reference": "",
  "actor_reference": "",
  "director_reference": "",
  "reference_movie": "",
  "author_reference": "",
  "publisher": "",
  "book_subject": "",
  "isbn": "",
  "artist_reference": "",
  "album_reference": "",
  "track_reference": "",
  "music_tag": "",
  "reference_title": "",
  "reference_creator": "",
  "transfer_terms": [],
  "exclude": [],
  "include_adult": false
}
"""


class IntentExtractor:
    def __init__(self, openai_service: Any) -> None:
        self.openai_service = openai_service

    async def extract(self, user_query: str, history_summary: str = "") -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": INTENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Conversation context:\n{history_summary}\n\n"
                    f"User request:\n{user_query}"
                ),
            },
        ]

        # Use your OpenAIService method here.
        # If your service does not have create_json(), add one.
        raw = await self.openai_service.create_json(messages)

        if isinstance(raw, dict):
            return self._normalize(raw)

        try:
            parsed = json.loads(raw)
            return self._normalize(parsed)
        except Exception:
            return self._fallback(user_query)

    def _normalize(self, intent: dict[str, Any]) -> dict[str, Any]:
        cleaned = {}

        for key, value in intent.items():
            if value in (None, "", [], {}):
                continue
            cleaned[key] = value

        cleaned["limit"] = max(1, min(int(cleaned.get("limit", 5)), 10))

        if "recent" not in cleaned:
            cleaned["recent"] = False

        if cleaned.get("year_min") in ("", None):
            cleaned.pop("year_min", None)

        if cleaned.get("year_max") in ("", None):
            cleaned.pop("year_max", None)

        if cleaned.get("country"):
            cleaned["country"] = str(cleaned["country"]).upper()

        if cleaned.get("language"):
            cleaned["language"] = str(cleaned["language"]).lower()

        return cleaned

    def _fallback(self, user_query: str) -> dict[str, Any]:
        lower = user_query.lower()

        if any(word in lower for word in ["book", "books", "novel", "read"]):
            domain = "books"
        elif any(word in lower for word in ["song", "songs", "music", "album", "artist", "singer"]):
            domain = "music"
        else:
            domain = "movies"

        return {
            "target_domain": domain,
            "query": user_query,
            "limit": 5,
            "recent": any(
                word in lower
                for word in ["latest", "recent", "new", "newest", "released recently"]
            ),
        }
