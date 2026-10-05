from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from .mcp_registry import MCPRegistry
from .openai_service import OpenAIService
from .preference_profile import (
    build_preference_profile,
    detect_target_domains,
    preference_context,
)
from .schemas import ExperimentMode


@dataclass(frozen=True)
class ExperimentConfig:
    use_mcp: bool
    use_history: bool
    cross_domain: bool
    grounded_only: bool


SESSION_RECOMMENDATION_MEMORY: dict[str, dict[str, list[dict[str, Any]]]] = {}


EXPERIMENTS: dict[ExperimentMode, ExperimentConfig] = {
    "full": ExperimentConfig(
        use_mcp=True,
        use_history=True,
        cross_domain=True,
        grounded_only=True,
    ),
    "pure_gpt": ExperimentConfig(
        use_mcp=False,
        use_history=True,
        cross_domain=True,
        grounded_only=False,
    ),
    "no_history": ExperimentConfig(
        use_mcp=True,
        use_history=False,
        cross_domain=True,
        grounded_only=True,
    ),
    "no_cross_domain": ExperimentConfig(
        use_mcp=True,
        use_history=True,
        cross_domain=False,
        grounded_only=True,
    ),
}


TOOL_BY_DOMAIN = {
    "movies": "recommend_movies",
    "movie": "recommend_movies",
    "books": "recommend_books",
    "book": "recommend_books",
    "music": "recommend_music",
}


DOMAIN_BY_TOOL = {
    "recommend_movies": "movies",
    "recommend_books": "books",
    "recommend_music": "music",
}


RECENT_WORDS = (
    "latest",
    "recent",
    "recently",
    "new",
    "newest",
    "current",
    "now playing",
    "release",
    "released",
    "new release",
    "new releases",
    "newly released",
    "released recently",
    "trending",
    "published recently",
)


GENRE_WORDS = {
    "action",
    "adventure",
    "animation",
    "biography",
    "business",
    "children",
    "classical",
    "comedy",
    "country",
    "crime",
    "documentary",
    "drama",
    "electronic",
    "family",
    "fantasy",
    "fiction",
    "folk",
    "hip hop",
    "hip-hop",
    "history",
    "horror",
    "indie",
    "jazz",
    "k-pop",
    "kpop",
    "metal",
    "mystery",
    "poetry",
    "pop",
    "rap",
    "r&b",
    "rock",
    "romance",
    "science fiction",
    "sci-fi",
    "scifi",
    "self help",
    "soundtrack",
    "teen",
    "teenage",
    "thriller",
    "war",
    "young adult",
}


MOOD_WORDS = {
    "adventurous",
    "atmospheric",
    "bittersweet",
    "calm",
    "cinematic",
    "creepy",
    "dark",
    "dreamy",
    "emotional",
    "energetic",
    "exciting",
    "feel good",
    "feel-good",
    "focused",
    "fun",
    "funny",
    "gentle",
    "happy",
    "intimate",
    "light",
    "lighthearted",
    "lonely",
    "melancholic",
    "melancholy",
    "mind bending",
    "mind-bending",
    "nostalgic",
    "philosophical",
    "relaxing",
    "romantic",
    "sad",
    "scary",
    "sentimental",
    "slow",
    "soft",
    "study",
    "uplifting",
}


INTENT_SYSTEM_PROMPT = """
You are the intent extraction layer for an MCP-powered conversational recommender system.

Return JSON only. No markdown. No explanation.

Architecture:
- The LLM understands the user's meaning, conversation history, follow-ups, mood, vibe, constraints, and preferences.
- The backend validates the JSON and passes only safe fields to MCP tools.
- MCP tools retrieve grounded candidates from external APIs.
- Do not recommend items and do not invent titles here.

Allowed JSON fields:
{
  "target_domain": "movies" | "books" | "music",
  "target_domains": ["movies" | "books" | "music"],
  "source_domain": "movies" | "books" | "music",
  "query": string,
  "limit": number,
  "recent": boolean,
  "sort_by": string,

  "year_min": number,
  "year_max": number,
  "decade": string,

  "genre": string,
  "mood": string,
  "theme": string,
  "audience": string,

  "page_count": number,
  "page_count_min": number,
  "page_count_max": number,

  "country": string,
  "language": string,

  "person_reference": string,
  "actor_reference": string,
  "director_reference": string,
  "reference_movie": string,

  "author_reference": string,
  "publisher": string,
  "book_subject": string,
  "isbn": string,

  "artist_reference": string,
  "album_reference": string,
  "track_reference": string,
  "music_tag": string,

  "reference_title": string,
  "reference_creator": string,
  "transfer_terms": [string],

  "exclude": string,
  "include_adult": boolean,

  "relation_to_previous": "related" | "unrelated",
  "relation_reason": string
}

Core behavior:
1. First decide whether the latest user message is RELATED or UNRELATED to the previous recommendation request.
   - Set relation_to_previous="related" when the latest message is a follow-up, modifier, correction, continuation, comparison, refinement, or request for more/another/different results.
   - Treat short filter/sort-only messages as related when they lack a new independent topic/entity, e.g. "oldest movies", "newest ones", "more underrated", "make it Korean", "not these", "same but books".
   - Set relation_to_previous="unrelated" when the latest message is a complete new request with a new domain/topic/entity/mood that should replace previous context, e.g. "latest action books" after a movie-actor request.
2. If relation_to_previous="related", resolve the latest message into a complete standalone intent by inheriting still-relevant previous context. Preserve previous domain, genre, mood, theme, audience, language, country, year, creator/person, reference item, and exclusions unless the user clearly changes them.
3. If relation_to_previous="unrelated", do not inherit previous domain/entity/genre/mood/theme. Use only the latest message and stable user preferences.
4. Use the full conversation when history is provided. The latest message may be a follow-up, modifier, correction, or added constraint.
5. Understand arbitrary natural language. Do not merely copy the user phrase when it is situational, emotional, abstract, or conversational.
6. Rewrite query into retrieval-friendly terms for the target domain.
5. Convert mood, vibe, emotion, situation, occasion, reading/listening/watching environment, relationship context, audience context, and constraints into the structured fields when relevant.
6. Similarity/vibe requests should extract reference_title/reference_creator when present and also produce mood/theme/genre/transfer_terms suitable for the target domain.
7. Cross-domain requests should infer the source domain from the reference and the target domain from the requested output.
8. Time intent must be explicit. If the user asks for latest/new/newest/recent/current/upcoming/released recently/newly published/newly released, set recent=true and sort_by="newest".
9. Never use sort_by="relevance" when recent=true unless the user explicitly asks for relevance/search match.
10. If the user mentions one specific year, set both year_min and year_max to that year.
11. If the user asks for rare/underrated/hidden gems/not popular/less mainstream, set sort_by="underrated".
12. If the user asks for oldest/earliest/first, set sort_by="oldest".
13. If the user asks for most famous/popular/top/mainstream, set sort_by="popular".
14. For normal recommendations without a requested order, use sort_by="relevance" for movies, books, and music.
14. For children/family-safe content, set audience appropriately and include_adult=false.
15. For broad regions, preserve the region meaning in query/theme unless a specific country/language is named.
16. For books, use author_reference only when a person is clearly the author. Use page_count/page_count_min/page_count_max only when the user's intent implies length constraints.
17. For music, use artist_reference when a person/band is clearly the artist.
18. For movies, use person_reference/actor_reference/director_reference when a person is involved.
19. Return only fields you are confident about. Leave uncertain fields absent.

Domain-specific guidance:
- Books: Convert reading context into retrieval variables. If the user describes a situation, environment, attention level, commute, bedtime, travel, study, portability, or available time, infer appropriate query/mood/theme/audience/page_count filters. Avoid accidental literal keyword searches when a word is only part of the situation.
- Movies: Convert emotional and situational viewing needs into query/mood/theme/genre/audience. Avoid passing vague natural phrases as the only query.
- Music: Convert vibe, mood, occasion, activity, artist/track references, and similarity requests into artist_reference, track_reference, music_tag, mood, theme, reference_title, reference_creator, and transfer_terms as appropriate.
"""


class RecommendationOrchestrator:
    def __init__(self, registry: MCPRegistry, openai_service: OpenAIService) -> None:
        self.registry = registry
        self.openai_service = openai_service

    async def stream_chat(
        self,
        history: list[dict[str, str]],
        mode: ExperimentMode = "full",
        session_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        config = EXPERIMENTS.get(
            mode,
            EXPERIMENTS["full"],
        )

        effective_history = history if config.use_history else history[-1:]

        latest_user_message = next(
            (m["content"] for m in reversed(history) if m.get("role") == "user"),
            "",
        )

        profile = build_preference_profile(
            history if config.use_history else history[-1:]
        )

        target_domains = detect_target_domains(latest_user_message)
        if not target_domains:
            target_domains = self._infer_target_domains(latest_user_message)

        profile_text = preference_context(
            profile,
            target_domains,
            cross_domain=config.cross_domain,
        )

        profile_message = {
            "role": "user",
            "content": profile_text,
        }

        base_system_message = {
            "role": "system",
            "content": (
                "You are a grounded conversational recommender for movies, books, and music. "
                "For MCP modes, final recommendations must only use returned MCP/API candidates. "
                "Do not invent unavailable titles. Explain why each returned item fits the user."
            ),
        }

        model_messages = [
            base_system_message,
            profile_message,
            *effective_history,
        ]

        tool_results: list[dict[str, Any]] = []

        if config.use_mcp:
            intent = await self._extract_intent_with_llm(
                user_query=latest_user_message,
                history=effective_history,
                history_summary=profile_text,
                cross_domain=config.cross_domain,
                use_history=config.use_history,
            )

            print("\n================ LLM INTENT ================", flush=True)
            print(json.dumps(intent, ensure_ascii=False, indent=2), flush=True)
            print("============================================\n", flush=True)

            intent_domains = self._domains_from_intent(intent)
            if intent_domains:
                target_domains = intent_domains

            for domain in target_domains:
                domain_intent = dict(intent)
                domain_intent["target_domain"] = domain

                tool_name, args = self._intent_to_tool_call(domain_intent)

                # The user-facing limit is how many items we display.
                # The MCP limit is how many grounded candidates we retrieve.
                # This lets repeated same-query requests show different items without hallucinating.
                display_limit = max(1, min(int(args.get("limit") or 5), 10))
                candidate_limit = max(display_limit * 6, 30)

                memory_key = self._intent_memory_key(tool_name, args)
                seen_items = self._get_seen_items_for_intent(session_id, memory_key)

                # Also extract titles already shown in persisted chat history.
                previous_titles = self._extract_previously_recommended_titles(effective_history)
                previous_items = [{"title": title} for title in previous_titles]
                combined_seen = seen_items + previous_items

                args_for_mcp = dict(args)
                args_for_mcp["limit"] = candidate_limit
                args_for_mcp["exclude"] = self._merge_exclude_titles(
                    args_for_mcp.get("exclude"),
                    combined_seen,
                )

                print("\n================ INTENT TO TOOL CALL ================", flush=True)
                print("TOOL:", tool_name, flush=True)
                print("DISPLAY LIMIT:", display_limit, flush=True)
                print("CANDIDATE LIMIT:", candidate_limit, flush=True)
                print("MEMORY KEY:", memory_key, flush=True)
                print("PREVIOUSLY SHOWN:", [item.get("title") for item in combined_seen if item.get("title")], flush=True)
                print("ARGS:", json.dumps(args_for_mcp, ensure_ascii=False, indent=2), flush=True)
                print("=====================================================\n", flush=True)

                result = await self._call_tool_with_debug(
                    tool_name,
                    args_for_mcp,
                    latest_user_message,
                    mode,
                    source="llm_intent_extraction",
                )

                raw_items = result.get("items") or []
                fresh_items = self._filter_seen_items(raw_items, combined_seen)
                display_items = fresh_items[:display_limit]

                # If every candidate was filtered, show grounded results anyway rather than hallucinating.
                if not display_items:
                    display_items = raw_items[:display_limit]

                result["items"] = display_items
                result["display_limit"] = display_limit
                result["candidate_count"] = len(raw_items)
                result["fresh_count"] = len(fresh_items)

                self._remember_displayed_items(session_id, memory_key, display_items)

                tool_results.append(
                    {
                        "name": tool_name,
                        "arguments": args_for_mcp,
                        "result": result,
                        "call_source": "llm_intent_extraction",
                        "intent": domain_intent,
                        "memory_key": memory_key,
                    }
                )

            if (
                self._looks_like_recommendation_request(latest_user_message)
                and not tool_results
            ):
                fallback_results = []

                for domain in target_domains:
                    tool_name = TOOL_BY_DOMAIN.get(domain, "recommend_movies")

                    args = self._build_fallback_tool_args(
                        latest_user_message,
                        profile.as_dict(),
                        target_domain=domain,
                        cross_domain=config.cross_domain,
                    )

                    result = await self._call_tool_with_debug(
                        tool_name,
                        args,
                        latest_user_message,
                        mode,
                        source="deterministic_fallback",
                    )

                    fallback_results.append(
                        {
                            "name": tool_name,
                            "arguments": args,
                            "result": result,
                            "call_source": "deterministic_fallback",
                            "intent": {},
                        }
                    )

                if self._tool_results_have_items(fallback_results):
                    tool_results = fallback_results
                elif not tool_results:
                    tool_results = fallback_results

        if config.use_mcp and tool_results:
            yield self._sse("tools", tool_results)

            debug_payload = self._build_debug_payload(
                latest_user_message,
                mode,
                target_domains,
                tool_results,
            )
            yield self._sse("debug", debug_payload)

            if not self._tool_results_have_items(tool_results):
                final_text = self._format_no_results_answer(tool_results)
            elif config.grounded_only:
                # Deterministic grounded rendering prevents the final LLM from adding extra titles.
                # If MCP returns 4 items, the user sees exactly 4 items.
                final_text = self._format_grounded_answer(tool_results)
            else:
                final_text = await self.openai_service.create_followup(
                    model_messages,
                    tool_results,
                    cross_domain=config.cross_domain,
                    grounded_only=config.grounded_only,
                    mode_name=mode,
                )

        else:
            response = await self.openai_service.create_response(
                model_messages,
                tools=[],
                use_mcp=False,
                use_history=config.use_history,
                cross_domain=config.cross_domain,
                grounded_only=config.grounded_only,
                mode_name=mode,
            )

            final_text = response.get("output_text", "")

        for chunk in self._chunk_text(final_text):
            yield self._sse("text", chunk)

        yield self._sse(
            "done",
            {
                "ok": True,
                "mode": mode,
                "config": config.__dict__,
            },
        )

    async def _extract_intent_with_llm(
        self,
        user_query: str,
        history: list[dict[str, str]],
        history_summary: str,
        cross_domain: bool,
        use_history: bool,
    ) -> dict[str, Any]:
        conversation_context = self._conversation_text(history if use_history else history[-1:])
        relation_context = self._relation_context(history if use_history else history[-1:], user_query)

        messages = [
            {
                "role": "system",
                "content": INTENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Cross-domain transfer enabled: {cross_domain}\n"
                    f"History enabled: {use_history}\n\n"
                    f"Conversation messages:\n{conversation_context}\n\n"
                    f"Conversation preference summary:\n{history_summary}\n\n"
                    f"Latest user message:\n{user_query}\n\n"
                    "Extract the complete structured recommendation intent for the latest user message. "
                    "If it is a follow-up, resolve it using the conversation messages. Return JSON only."
                ),
            },
        ]

        try:
            if hasattr(self.openai_service, "create_json"):
                raw = await self.openai_service.create_json(messages)
            else:
                response = await self.openai_service.create_response(
                    messages,
                    tools=[],
                    use_mcp=False,
                    use_history=True,
                    cross_domain=cross_domain,
                    grounded_only=False,
                    mode_name="intent_extraction",
                )
                raw = response.get("output_text", "{}")

            if isinstance(raw, dict):
                return self._normalize_intent(raw, user_query)

            parsed = self._parse_json_text(str(raw))
            return self._normalize_intent(parsed, user_query)

        except Exception as exc:
            print(f"[INTENT] LLM extraction failed: {type(exc).__name__}: {exc}")
            return self._fallback_intent(user_query)


    def _conversation_text(self, history: list[dict[str, str]], max_messages: int = 12) -> str:
        recent = history[-max_messages:]
        lines: list[str] = []

        for message in recent:
            role = str(message.get("role", "")).strip() or "unknown"
            content = str(message.get("content", "")).strip()

            if content:
                lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _relation_context(self, history: list[dict[str, str]], latest_user_message: str) -> str:
        """Give the LLM explicit context for deciding follow-up vs new request.

        This does not hardcode recommendation meaning. It surfaces the previous
        user turn and generic linguistic signals so the LLM can decide whether
        to inherit prior constraints or start fresh.
        """
        user_messages = [
            str(message.get("content") or "").strip()
            for message in history
            if message.get("role") == "user" and str(message.get("content") or "").strip()
        ]

        previous_user = ""
        if len(user_messages) >= 2:
            previous_user = user_messages[-2]

        latest = str(latest_user_message or "").strip()
        latest_lower = latest.lower()

        followup_markers = (
            "more",
            "another",
            "again",
            "same",
            "similar",
            "like that",
            "like this",
            "those",
            "these",
            "them",
            "ones",
            "not these",
            "instead",
            "but",
            "also",
            "underrated",
            "rare",
            "oldest",
            "newest",
            "latest",
            "shorter",
            "longer",
            "funnier",
            "scarier",
            "sadder",
        )
        has_followup_marker = any(marker in latest_lower for marker in followup_markers)

        domain_mentions = [
            word
            for word in ("movie", "movies", "film", "films", "book", "books", "novel", "novels", "music", "song", "songs", "track", "tracks")
            if re.search(rf"\b{re.escape(word)}\b", latest_lower)
        ]

        # A request with only a domain plus ordering/filter words is often a continuation.
        # The LLM still makes the final decision.
        non_context_words = set(domain_mentions) | {
            "oldest", "newest", "latest", "recent", "recently", "popular", "underrated",
            "rare", "more", "another", "same", "similar", "top", "best", "worst", "good",
            "suggest", "recommend", "me", "show", "give", "please", "the", "a", "an",
        }
        tokens = re.findall(r"[a-zA-Z0-9']+", latest_lower)
        content_tokens = [token for token in tokens if token not in non_context_words]
        looks_filter_only = bool(has_followup_marker and previous_user and len(content_tokens) <= 1)

        return "\n".join(
            [
                f"Previous user request: {previous_user or '(none)'}",
                f"Latest user request: {latest}",
                f"Has follow-up/modifier marker: {has_followup_marker}",
                f"Mentions domains: {', '.join(domain_mentions) if domain_mentions else '(none)'}",
                f"Looks like short filter/modifier-only follow-up: {looks_filter_only}",
                "Decision rule: related means inherit still-relevant previous constraints; unrelated means do not inherit request-specific constraints.",
            ]
        )

    def _parse_json_text(self, text: str) -> dict[str, Any]:
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return {}

    def _normalize_intent(
        self,
        intent: dict[str, Any],
        user_query: str,
    ) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}

        for key, value in intent.items():
            if value in (None, "", [], {}):
                continue
            cleaned[key] = value

        target_domain = str(cleaned.get("target_domain") or "").lower().strip()
        if target_domain in ("movie", "film", "films"):
            target_domain = "movies"
        elif target_domain in ("book", "novel", "novels"):
            target_domain = "books"
        elif target_domain in ("song", "songs", "track", "tracks", "album", "albums"):
            target_domain = "music"

        if target_domain not in ("movies", "books", "music"):
            inferred = self._infer_target_domains(user_query)
            target_domain = inferred[0] if inferred else "movies"

        cleaned["target_domain"] = target_domain

        target_domains = cleaned.get("target_domains")
        if isinstance(target_domains, list):
            normalized_domains = []
            for domain in target_domains:
                domain_text = str(domain).lower().strip()
                if domain_text in ("movie", "movies", "film", "films"):
                    normalized_domains.append("movies")
                elif domain_text in ("book", "books", "novel", "novels"):
                    normalized_domains.append("books")
                elif domain_text in ("music", "song", "songs", "track", "tracks", "album", "albums"):
                    normalized_domains.append("music")

            if normalized_domains:
                cleaned["target_domains"] = list(dict.fromkeys(normalized_domains))

        try:
            cleaned["limit"] = max(1, min(int(cleaned.get("limit", 5)), 10))
        except Exception:
            cleaned["limit"] = 5

        lower_query = str(user_query or "").lower()

        explicit_recent = any(word in lower_query for word in RECENT_WORDS)
        cleaned["recent"] = bool(cleaned.get("recent", False)) or explicit_recent

        explicit_oldest_request = any(
            phrase in lower_query
            for phrase in ("oldest", "earliest", "first", "classic", "classics")
        )
        explicit_popular_request = any(
            phrase in lower_query
            for phrase in (
                "popular",
                "famous",
                "mainstream",
                "well known",
                "well-known",
                "best known",
                "most known",
            )
        )
        explicit_top_rated_request = any(
            phrase in lower_query
            for phrase in ("top rated", "highest rated", "best rated")
        )
        explicit_underrated_request = any(
            phrase in lower_query
            for phrase in (
                "underrated",
                "hidden gem",
                "hidden gems",
                "rare",
                "less known",
                "less-known",
                "less mainstream",
                "not popular",
            )
        )

        # Hard sort hierarchy. This intentionally overrides the LLM when it
        # returns the bad combination: recent=true + sort_by="relevance".
        if explicit_oldest_request:
            cleaned["sort_by"] = "oldest"
        elif explicit_underrated_request:
            cleaned["sort_by"] = "underrated"
        elif explicit_top_rated_request:
            cleaned["sort_by"] = "top_rated"
        elif explicit_popular_request:
            cleaned["sort_by"] = "popular"
        elif cleaned.get("recent"):
            cleaned["sort_by"] = "newest"
        elif not cleaned.get("sort_by"):
            cleaned["sort_by"] = self._default_sort_by(cleaned.get("target_domain", "movies"))

        if cleaned.get("sort_by") in ("new", "latest", "recent"):
            cleaned["sort_by"] = "newest"

        mood_or_vibe_request = any(
            cleaned.get(key)
            for key in ("mood", "theme", "transfer_terms", "reference_title", "reference_creator")
        )

        if mood_or_vibe_request and cleaned.get("sort_by") == "popular" and not explicit_popular_request:
            cleaned["sort_by"] = "relevance"

        # Music-specific repair: "latest BTS songs" should be an artist query,
        # not a literal Last.fm text search for the phrase "latest BTS songs".
        if cleaned.get("target_domain") == "music" and not cleaned.get("artist_reference"):
            inferred_artist = self._infer_music_artist_reference(user_query)
            if inferred_artist:
                cleaned["artist_reference"] = inferred_artist

        for int_key in ("year_min", "year_max", "page_count", "page_count_min", "page_count_max"):
            value = cleaned.get(int_key)
            try:
                value_int = int(value)
                if value_int > 0:
                    cleaned[int_key] = value_int
                else:
                    cleaned.pop(int_key, None)
            except Exception:
                cleaned.pop(int_key, None)

        standalone_year = self._extract_standalone_year(lower_query)
        if standalone_year and not self._has_relative_year_word(lower_query):
            cleaned["year_min"] = standalone_year
            cleaned["year_max"] = standalone_year

        if any(word in lower_query for word in ("with kids", "for kids", "kid", "kids", "children", "child", "family friendly")):
            cleaned["audience"] = cleaned.get("audience") or "kids"
            cleaned["include_adult"] = False

        if cleaned.get("target_domain") == "books":
            page_filters = self._extract_page_filters(lower_query)
            cleaned.update(page_filters)

        if "asian" in lower_query and not any(word in lower_query for word in ("korean", "japanese", "chinese", "indian", "malay", "malaysian", "indonesian", "thai", "filipino")):
            cleaned.pop("country", None)
            if cleaned.get("target_domain") == "movies" and (not cleaned.get("query") or "asian" not in str(cleaned.get("query")).lower()):
                cleaned["query"] = "Asian movies"

        if cleaned.get("country"):
            cleaned["country"] = str(cleaned["country"]).upper().strip()

        if cleaned.get("language"):
            cleaned["language"] = str(cleaned["language"]).lower().strip()

        if not cleaned.get("query"):
            query_parts = []

            transfer_terms = cleaned.get("transfer_terms")
            if isinstance(transfer_terms, list):
                query_parts.extend(str(term) for term in transfer_terms if term)

            for key in (
                "reference_title",
                "reference_creator",
                "genre",
                "mood",
                "theme",
                "audience",
                "person_reference",
                "author_reference",
                "artist_reference",
            ):
                if cleaned.get(key):
                    query_parts.append(str(cleaned[key]))

            cleaned["query"] = " ".join(query_parts).strip() or user_query

        return cleaned

    def _fallback_intent(self, user_query: str) -> dict[str, Any]:
        lower = user_query.lower()
        domains = self._infer_target_domains(user_query)
        domain = domains[0] if domains else "movies"

        intent: dict[str, Any] = {
            "target_domain": domain,
            "query": user_query,
            "limit": 5,
            "recent": any(word in lower for word in RECENT_WORDS),
        }

        year_min = self._extract_year_min(lower)
        year_max = self._extract_year_max(lower)
        genre = self._extract_first_match(lower, GENRE_WORDS)
        mood = self._extract_first_match(lower, MOOD_WORDS)

        standalone_year = self._extract_standalone_year(lower)

        if year_min:
            intent["year_min"] = year_min

        if standalone_year and not self._has_relative_year_word(lower):
            intent["year_min"] = standalone_year
            intent["year_max"] = standalone_year

        if year_max:
            intent["year_max"] = year_max
            intent.pop("year_min", None)

        if genre:
            intent["genre"] = genre

        if mood:
            intent["mood"] = mood

        if any(word in lower for word in ("with kids", "for kids", "kid", "kids", "children", "child", "family friendly")):
            intent["audience"] = "kids"
            intent["include_adult"] = False

        if "asian" in lower and domain == "movies":
            intent["query"] = "Asian movies"
            intent.pop("country", None)

        if any(word in lower for word in ("oldest", "earliest", "first", "classic", "classics")):
            intent["sort_by"] = "oldest"
        elif any(word in lower for word in ("underrated", "hidden gem", "hidden gems", "rare", "less known", "less-known")):
            intent["sort_by"] = "underrated"
        elif any(word in lower for word in ("highest rated", "top rated", "best rated")):
            intent["sort_by"] = "top_rated"
        elif any(word in lower for word in ("famous", "popular", "best known", "well known", "mainstream")):
            intent["sort_by"] = "popular"
        elif intent.get("recent"):
            intent["sort_by"] = "newest"
        else:
            intent["sort_by"] = self._default_sort_by(domain)

        if domain == "music" and not intent.get("artist_reference"):
            artist = self._infer_music_artist_reference(user_query)
            if artist:
                intent["artist_reference"] = artist

        return intent

    def _domains_from_intent(self, intent: dict[str, Any]) -> list[str]:
        domains: list[str] = []

        raw_domains = intent.get("target_domains")
        if isinstance(raw_domains, list):
            for domain in raw_domains:
                domain_text = str(domain).lower().strip()
                if domain_text in ("movie", "movies", "film", "films"):
                    domains.append("movies")
                elif domain_text in ("book", "books", "novel", "novels"):
                    domains.append("books")
                elif domain_text in ("music", "song", "songs", "album", "albums", "track", "tracks"):
                    domains.append("music")

        target_domain = str(intent.get("target_domain") or "").lower().strip()
        if target_domain in ("movie", "movies", "film", "films"):
            domains.append("movies")
        elif target_domain in ("book", "books", "novel", "novels"):
            domains.append("books")
        elif target_domain in ("music", "song", "songs", "album", "albums", "track", "tracks"):
            domains.append("music")

        return list(dict.fromkeys(domains))

    def _title_key(self, title: Any) -> str:
        text = str(title or "").lower().strip()
        text = re.sub(r"[^a-z0-9\u00c0-\u024f\u3040-\u30ff\u4e00-\u9fff가-힣]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _intent_memory_key(self, tool_name: str, args: dict[str, Any]) -> str:
        """Stable novelty-memory key.

        This avoids repeats across semantically similar requests. It intentionally
        depends mainly on LLM-resolved variables, not the exact sentence wording,
        so "movies that make me cry" and "heartbreak movies" can share
        already-shown memory when their mood/theme/genre are similar.
        """
        query_text = str(args.get("query", "")).lower().strip()

        important = {
            "tool": tool_name,
            "genre": str(args.get("genre", "")).lower().strip(),
            "mood": str(args.get("mood", "")).lower().strip(),
            "theme": str(args.get("theme", "")).lower().strip(),
            "audience": str(args.get("audience", "")).lower().strip(),
            "country": str(args.get("country", "")).upper().strip(),
            "language": str(args.get("language", "")).lower().strip(),
            "year_min": int(args.get("year_min") or 0),
            "year_max": int(args.get("year_max") or 0),
            "recent": bool(args.get("recent") or False),
            "sort_by": str(args.get("sort_by", "")).lower().strip(),
            "author_reference": str(args.get("author_reference", "")).lower().strip(),
            "artist_reference": str(args.get("artist_reference", "")).lower().strip(),
            "actor_reference": str(args.get("actor_reference", "")).lower().strip(),
            "director_reference": str(args.get("director_reference", "")).lower().strip(),
            "reference_title": str(args.get("reference_title", "")).lower().strip(),
            "reference_creator": str(args.get("reference_creator", "")).lower().strip(),
        }

        has_semantic_fields = any(
            important.get(key)
            for key in (
                "genre",
                "mood",
                "theme",
                "audience",
                "country",
                "language",
                "author_reference",
                "artist_reference",
                "actor_reference",
                "director_reference",
                "reference_title",
                "reference_creator",
            )
        )

        if not has_semantic_fields:
            important["query"] = query_text

        raw = json.dumps(important, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_seen_items_for_intent(self, session_id: str, memory_key: str) -> list[dict[str, Any]]:
        return SESSION_RECOMMENDATION_MEMORY.get(session_id, {}).get(memory_key, [])

    def _remember_displayed_items(
        self,
        session_id: str,
        memory_key: str,
        items: list[dict[str, Any]],
    ) -> None:
        if not session_id or not memory_key or not items:
            return

        session_store = SESSION_RECOMMENDATION_MEMORY.setdefault(session_id, {})
        stored = session_store.setdefault(memory_key, [])
        existing = {self._title_key(item.get("title")) for item in stored}

        for item in items:
            key = self._title_key(item.get("title"))
            if key and key not in existing:
                stored.append(item)
                existing.add(key)

        session_store[memory_key] = stored[-100:]

    def _merge_exclude_titles(self, existing_exclude: Any, seen_items: list[dict[str, Any]]) -> str:
        titles: list[str] = []

        if isinstance(existing_exclude, list):
            titles.extend(str(x).strip() for x in existing_exclude if str(x).strip())
        elif existing_exclude:
            titles.extend(
                part.strip()
                for part in re.split(r"[,;\n]+", str(existing_exclude))
                if part.strip()
            )

        for item in seen_items:
            title = str(item.get("title") or "").strip()
            if title:
                titles.append(title)

        deduped: list[str] = []
        seen: set[str] = set()

        for title in titles:
            key = self._title_key(title)
            if key and key not in seen:
                seen.add(key)
                deduped.append(title)

        return ", ".join(deduped)

    def _filter_seen_items(
        self,
        items: list[dict[str, Any]],
        seen_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen_titles = {
            self._title_key(item.get("title"))
            for item in seen_items
            if item.get("title")
        }

        fresh: list[dict[str, Any]] = []
        local_seen: set[str] = set()

        for item in items:
            key = self._title_key(item.get("title"))
            if not key or key in seen_titles or key in local_seen:
                continue
            fresh.append(item)
            local_seen.add(key)

        return fresh

    def _extract_previously_recommended_titles(self, history: list[dict[str, str]]) -> list[str]:
        """Extract titles previously shown to the user so repeated/follow-up queries can get variety.

        This is not intent hardcoding. It only prevents duplicate grounded candidates.
        It looks for the markdown format used by _format_grounded_answer: - **Title** ...
        """
        titles: list[str] = []
        seen: set[str] = set()

        for message in history:
            if message.get("role") != "assistant":
                continue

            content = str(message.get("content") or "")
            for raw in re.findall(r"^\s*[-*]\s+\*\*(.+?)\*\*", content, flags=re.MULTILINE):
                title = raw.strip()
                # Remove common subtitle/year suffixes from the displayed label.
                title = re.sub(r"\s+\((?:19|20)\d{2}\)\s*$", "", title).strip()
                if not title:
                    continue
                key = self._normalize_title(title)
                if key and key not in seen:
                    seen.add(key)
                    titles.append(title)

        return titles

    def _coerce_tool_arg_types(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Match LLM JSON output to MCP/Pydantic tool schemas.

        The LLM may naturally produce arrays for fields such as transfer_terms or exclude.
        Current MCP tools expect those fields as strings, so normalize them here.
        """
        fixed = dict(args)

        for key in ("transfer_terms", "exclude"):
            if key not in fixed:
                continue
            value = fixed.get(key)
            if isinstance(value, list):
                fixed[key] = ", ".join(str(v).strip() for v in value if str(v).strip())
            elif value is None:
                fixed.pop(key, None)
            else:
                fixed[key] = str(value).strip()

        return fixed

    def _intent_to_tool_call(self, intent: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        domain = str(intent.get("target_domain") or "movies").lower().strip()

        if domain in ("movie", "film", "films"):
            domain = "movies"
        elif domain in ("book", "novel", "novels"):
            domain = "books"
        elif domain in ("song", "songs", "track", "tracks", "album", "albums"):
            domain = "music"

        tool_name = TOOL_BY_DOMAIN.get(domain, "recommend_movies")

        allowed_by_tool = {
            "recommend_movies": {
                "query",
                "limit",
                "recent",
                "genre",
                "mood",
                "theme",
                "audience",
                "year_min",
                "year_max",
                "country",
                "language",
                "person_reference",
                "actor_reference",
                "director_reference",
                "reference_movie",
                "source_domain",
                "reference_title",
                "reference_creator",
                "transfer_terms",
                "sort_by",
                "exclude",
                "include_adult",
            },
            "recommend_books": {
                "query",
                "limit",
                "recent",
                "genre",
                "mood",
                "theme",
                "audience",
                "year_min",
                "year_max",
                "language",
                "country",
                "author_reference",
                "publisher",
                "book_subject",
                "isbn",
                "page_count",
                "page_count_min",
                "page_count_max",
                "sort_by",
                "exclude",
                "include_adult",
            },
            "recommend_music": {
                "query",
                "limit",
                "recent",
                "genre",
                "mood",
                "theme",
                "year_min",
                "year_max",
                "country",
                "language",
                "artist_reference",
                "album_reference",
                "track_reference",
                "music_tag",
                "reference_title",
                "reference_creator",
                "transfer_terms",
                "sort_by",
                "exclude",
                "include_adult",
            },
        }

        args: dict[str, Any] = {}

        for key in allowed_by_tool[tool_name]:
            value = intent.get(key)

            if value in (None, "", [], {}):
                continue

            args[key] = value

        if "query" not in args or not str(args.get("query", "")).strip():
            transfer_terms = intent.get("transfer_terms") or []

            if isinstance(transfer_terms, list) and transfer_terms:
                args["query"] = " ".join(str(term) for term in transfer_terms if term)
            else:
                args["query"] = (
                    intent.get("reference_title")
                    or intent.get("reference_movie")
                    or intent.get("person_reference")
                    or intent.get("author_reference")
                    or intent.get("artist_reference")
                    or "popular"
                )

        args["query"] = str(args["query"]).strip()
        args["limit"] = max(1, min(int(args.get("limit", 5)), 10))

        if "recent" not in args:
            args["recent"] = False

        if args.get("recent") and not args.get("sort_by"):
            args["sort_by"] = "newest"
        elif not args.get("sort_by"):
            args["sort_by"] = "relevance"

        for int_key in ("year_min", "year_max", "page_count", "page_count_min", "page_count_max"):
            if int_key in args:
                try:
                    value_int = int(args[int_key])
                    if value_int > 0:
                        args[int_key] = value_int
                    else:
                        args.pop(int_key, None)
                except Exception:
                    args.pop(int_key, None)

        if args.get("country"):
            args["country"] = str(args["country"]).upper().strip()

        if args.get("language"):
            args["language"] = str(args["language"]).lower().strip()

        if args.get("include_adult") is None:
            args.pop("include_adult", None)

        args = self._coerce_tool_arg_types(tool_name, args)

        return tool_name, args

    async def _call_tool_with_debug(
        self,
        tool_name: str,
        args: dict[str, Any],
        user_query: str,
        mode: ExperimentMode,
        *,
        source: str,
    ) -> dict[str, Any]:
        print("\n================ MCP QUERY TRACE ================", flush=True)
        print("TRACE SOURCE:", source, flush=True)
        print("USER QUERY:", user_query, flush=True)
        print("MODE:", mode, flush=True)
        print("TOOL:", tool_name, flush=True)
        print("ARGS:", json.dumps(args, ensure_ascii=False, indent=2), flush=True)

        result = await self.registry.call_tool(tool_name, args)

        items = result.get("items") or []
        print("RESULT SOURCE:", result.get("source") or result.get("provider"), flush=True)
        print("RESULT COUNT:", len(items), flush=True)
        if result.get("error"):
            print("RESULT ERROR:", result.get("error"), flush=True)

        for idx, item in enumerate(items[:10], start=1):
            print(
                f'{idx}. {item.get("title")} | '
                f'{item.get("subtitle")} | '
                f'source={item.get("source") or result.get("source")} | '
                f'url={item.get("url")}',
                flush=True,
            )

        print("=================================================\n", flush=True)
        return result

    def _infer_target_domains(self, text: str) -> list[str]:
        lower = text.lower()
        domains: list[str] = []

        if any(
            word in lower
            for word in ["movie", "movies", "film", "films", "watch", "cinema"]
        ):
            domains.append("movies")

        if any(
            word in lower
            for word in [
                "song",
                "songs",
                "music",
                "musics",
                "track",
                "tracks",
                "album",
                "albums",
                "artist",
                "singer",
                "band",
                "playlist",
            ]
        ):
            domains.append("music")

        if any(
            word in lower
            for word in [
                "book",
                "books",
                "novel",
                "novels",
                "read",
                "author",
                "writer",
                "novelist",
            ]
        ):
            domains.append("books")

        return domains or ["movies"]

    def _looks_like_recommendation_request(self, text: str) -> bool:
        lower = text.lower()

        return any(
            word in lower
            for word in [
                "recommend",
                "suggest",
                "find",
                "give me",
                "what should i watch",
                "what should i read",
                "playlist",
                "songs",
                "tracks",
                "movies",
                "films",
                "books",
                "music",
                "musics",
                "albums",
                "latest",
                "new",
                "recent",
                "vibe",
                "vibes",
                "similar",
                "like",
                "feeling",
                "mood",
                "aesthetic",
                "famous",
                "popular",
                "oldest",
                "best",
            ]
        )

    def _build_fallback_tool_args(
        self,
        latest_user_message: str,
        profile: dict[str, list[str]],
        *,
        target_domain: str,
        cross_domain: bool,
    ) -> dict[str, Any]:
        query = self._build_fallback_retrieval_query(
            latest_user_message,
            profile,
            target_domain=target_domain,
            cross_domain=cross_domain,
        )

        lower = query.lower()

        recent = any(word in lower for word in RECENT_WORDS)
        year_min = self._extract_year_min(lower)
        year_max = self._extract_year_max(lower)
        genre = self._extract_first_match(lower, GENRE_WORDS)
        mood = self._extract_first_match(lower, MOOD_WORDS)

        args: dict[str, Any] = {
            "query": query,
            "limit": 5,
            "recent": bool(recent),
        }

        if genre:
            args["genre"] = genre

        if mood:
            args["mood"] = mood

        standalone_year = self._extract_standalone_year(lower)

        if year_min:
            args["year_min"] = int(year_min)

        if standalone_year and not self._has_relative_year_word(lower):
            args["year_min"] = int(standalone_year)
            args["year_max"] = int(standalone_year)

        if year_max:
            args["year_max"] = int(year_max)
            args.pop("year_min", None)

        if any(word in lower for word in ("oldest", "earliest", "first")):
            args["sort_by"] = "oldest"
        elif any(word in lower for word in ("famous", "popular", "best known", "well known")):
            args["sort_by"] = "popular"
        elif any(word in lower for word in ("highest rated", "top rated", "best rated")):
            args["sort_by"] = "top_rated"
        elif recent:
            args["sort_by"] = "newest"
        else:
            args["sort_by"] = self._default_sort_by(target_domain)

        if any(word in lower for word in ("with kids", "for kids", "kid", "kids", "children", "child", "family friendly")):
            args["audience"] = "kids"
            args["include_adult"] = False

        if "asian" in lower and target_domain == "movies":
            args["query"] = "Asian movies"
            args.pop("country", None)

        if target_domain == "movies":
            person = self._extract_person_reference(latest_user_message)
            if person:
                args["person_reference"] = person

        if target_domain == "music":
            artist = self._extract_artist_reference(latest_user_message)
            if artist:
                args["artist_reference"] = artist

            album = self._extract_album_reference(latest_user_message)
            if album:
                args["album_reference"] = album

        if target_domain == "books":
            author = self._extract_author_reference(latest_user_message)
            if author:
                args["author_reference"] = author
            args.update(self._extract_page_filters(lower))

        return args

    def _build_fallback_retrieval_query(
        self,
        latest_user_message: str,
        profile: dict[str, list[str]],
        *,
        target_domain: str,
        cross_domain: bool,
    ) -> str:
        latest = latest_user_message.strip()
        lower = latest.lower()

        direct_domain_words = {
            "movies": ["movie", "movies", "film", "films", "watch", "cinema"],
            "music": [
                "song",
                "songs",
                "music",
                "musics",
                "track",
                "tracks",
                "album",
                "albums",
                "artist",
                "singer",
                "band",
                "playlist",
            ],
            "books": [
                "book",
                "books",
                "novel",
                "novels",
                "read",
                "author",
                "writer",
                "novelist",
            ],
        }

        is_direct_domain_request = any(
            word in lower for word in direct_domain_words.get(target_domain, [])
        )

        if is_direct_domain_request:
            return latest

        parts = [latest]

        if cross_domain:
            for values in profile.values():
                parts.extend(values[:2])
        else:
            key_by_domain = {
                "movies": "movie_preferences",
                "music": "music_preferences",
                "books": "book_preferences",
            }
            parts.extend(profile.get(key_by_domain.get(target_domain, ""), [])[:2])

        query = " ".join(part for part in parts if part).strip()
        return query or f"popular {target_domain}"

    def _infer_music_artist_reference(self, user_query: str) -> str:
        text = str(user_query or "").strip()

        patterns = (
            r"\b(?:latest|newest|recent|new)\s+(.+?)\s+(?:songs?|tracks?|music|albums?)\b",
            r"\b(?:songs?|tracks?|music|albums?)\s+by\s+(.+?)\b",
            r"\bby\s+(.+?)\b",
        )

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue

            artist = match.group(1).strip()
            artist = re.sub(
                r"\b(latest|newest|recent|new|songs?|tracks?|music|albums?|recommend|suggest|me|please)\b",
                " ",
                artist,
                flags=re.IGNORECASE,
            )
            artist = re.sub(r"\s+", " ", artist).strip(" -:")

            if artist:
                return artist

        return ""

    def _default_sort_by(self, target_domain: str) -> str:
        return "relevance"

    def _extract_page_filters(self, text: str) -> dict[str, int]:
        filters: dict[str, int] = {}

        exact_match = re.search(r"\b(?:with|around|about|exactly)?\s*(\d{1,4})\s+pages?\b", text)
        if exact_match:
            filters["page_count"] = int(exact_match.group(1))

        under_match = re.search(r"\b(?:under|below|less than|max|max(?:imum)?|up to)\s*(\d{1,4})\s+pages?\b", text)
        if under_match:
            filters.pop("page_count", None)
            filters["page_count_max"] = int(under_match.group(1))

        over_match = re.search(r"\b(?:over|above|more than|min|min(?:imum)?|at least)\s*(\d{1,4})\s+pages?\b", text)
        if over_match:
            filters.pop("page_count", None)
            filters["page_count_min"] = int(over_match.group(1))

        if "short book" in text or "short books" in text:
            filters.setdefault("page_count_max", 160)

        return filters

    def _has_relative_year_word(self, text: str) -> bool:
        return bool(re.search(r"\b(before|until|up to|through|and before|or before|after|since|from)\b", text))

    def _extract_standalone_year(self, text: str) -> int | None:
        if self._has_relative_year_word(text):
            return None
        match = re.search(r"\b(20\d{2})\b", text)
        if match:
            return int(match.group(1))
        return None

    def _extract_year_min(self, text: str) -> int | None:
        if re.search(r"\b(before|until|up to|through|and before|or before)\b", text):
            return None

        match = re.search(r"\b(?:after|since|from)\s+(20\d{2})\b", text)
        if match:
            return int(match.group(1))

        match = re.search(r"\b(20\d{2})\b", text)
        if match:
            return int(match.group(1))

        return None

    def _extract_year_max(self, text: str) -> int | None:
        patterns = [
            r"\b(?:before|until|up to|through)\s+(20\d{2})\b",
            r"\b(20\d{2})\s+(?:and before|or before)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))

        return None

    def _extract_first_match(self, text: str, candidates: set[str]) -> str | None:
        for candidate in sorted(candidates, key=len, reverse=True):
            if re.search(rf"\b{re.escape(candidate)}\b", text):
                return candidate
        return None

    def _extract_person_reference(self, text: str) -> str | None:
        patterns = [
            r"(?:actor|actress|director|filmmaker|movie by|film by)\s+([A-Z][\w\s&.'-]{1,60})",
            r"([A-Z][\w\s&.'-]{1,60})['’]s\s+(?:movies|films|oldest|latest|famous|popular|best)",
            r"(?:oldest|latest|famous|popular|best)\s+([A-Z][\w\s&.'-]{1,60})\s+(?:movies|films)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)

            if match:
                person = match.group(1).strip(" .,!?:;'’")
                person = re.sub(
                    r"\b(oldest|latest|famous|popular|best|movies?|films?|actor|actress|director|please|recommend|suggest|me|give)\b",
                    " ",
                    person,
                    flags=re.IGNORECASE,
                )
                person = re.sub(r"\s+", " ", person).strip(" .,!?:;'’")

                if person:
                    return person

        return None

    def _extract_artist_reference(self, text: str) -> str | None:
        patterns = [
            r"(?:by|from)\s+([A-Z][\w\s&.'-]{1,60})",
            r"(?:singer|artist|musician|band)\s+([A-Z][\w\s&.'-]{1,60})",
            r"new\s+([A-Z][\w\s&.'-]{1,60})\s+(?:song|songs|music|musics|track|tracks|album|albums)",
            r"latest\s+([A-Z][\w\s&.'-]{1,60})\s+(?:song|songs|music|musics|track|tracks|album|albums)",
            r"([A-Z][\w\s&.'-]{1,60})\s+(?:new|latest|recent)\s+(?:song|songs|music|musics|track|tracks|album|albums)",
            r"([A-Z][\w\s&.'-]{1,60})['’]s\s+(?:latest|new|recent)\s+(?:song|songs|music|musics|track|tracks|album|albums)",
            r"like\s+([A-Z][\w\s&.'-]{1,60})",
            r"similar to\s+([A-Z][\w\s&.'-]{1,60})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)

            if match:
                artist = match.group(1).strip(" .,!?:;'’")
                artist = re.sub(
                    r"\b(new|latest|recent|released|songs?|musics?|music|tracks?|albums?|please|recommend|suggest|me|give)\b",
                    " ",
                    artist,
                    flags=re.IGNORECASE,
                )
                artist = re.sub(r"\s+", " ", artist).strip(" .,!?:;'’")

                if artist:
                    return artist

        return None

    def _extract_album_reference(self, text: str) -> str | None:
        patterns = [
            r"album\s+(?:called|named|titled)\s+([A-Z0-9][\w\s&.'-]{1,80})",
            r"(?:recommend|suggest|find|give me)?\s*([A-Z0-9][\w\s&.'-]{1,80})\s+album",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)

            if match:
                album = match.group(1).strip(" .,!?:;'’")
                album = re.sub(
                    r"\b(recommend|suggest|find|give me|some|latest|new|recent|released|album|albums|please|me)\b",
                    " ",
                    album,
                    flags=re.IGNORECASE,
                )
                album = re.sub(r"\s+", " ", album).strip(" .,!?:;'’")

                if album and len(album.split()) <= 8:
                    return album

        return None

    def _extract_author_reference(self, text: str) -> str | None:
        patterns = [
            r"(?:by|from)\s+([A-Z][\w\s&.'-]{1,60})",
            r"(?:author|writer|novelist)\s+([A-Z][\w\s&.'-]{1,60})",
            r"new\s+([A-Z][\w\s&.'-]{1,60})\s+(?:book|books|novel|novels)",
            r"latest\s+([A-Z][\w\s&.'-]{1,60})\s+(?:book|books|novel|novels)",
            r"([A-Z][\w\s&.'-]{1,60})['’]s\s+(?:latest|new|recent)\s+(?:book|books|novel|novels)",
            r"books?\s+by\s+([A-Z][\w\s&.'-]{1,60})",
            r"novels?\s+by\s+([A-Z][\w\s&.'-]{1,60})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)

            if match:
                author = match.group(1).strip(" .,!?:;'’")
                author = re.sub(
                    r"\b(new|latest|recent|released|books?|novels?|author|writer|please|recommend|suggest|me|give)\b",
                    " ",
                    author,
                    flags=re.IGNORECASE,
                )
                author = re.sub(r"\s+", " ", author).strip(" .,!?:;'’")

                if author:
                    return author

        return None

    def _tool_results_have_items(self, tool_results: list[dict[str, Any]]) -> bool:
        for call in tool_results:
            result = call.get("result") or {}
            if result.get("items"):
                return True
        return False

    def _allowed_titles(self, tool_results: list[dict[str, Any]]) -> set[str]:
        titles: set[str] = set()

        for call in tool_results:
            result = call.get("result") or {}

            for item in result.get("items") or []:
                title = item.get("title")

                if title:
                    titles.add(self._normalize_title(title))

        return titles

    def _normalize_title(self, text: str) -> str:
        text = re.sub(r"\s*\((?:19|20)\d{2}\)\s*$", "", text)
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
        return text

    def _contains_hallucinated_recommendation(
        self,
        text: str,
        tool_results: list[dict[str, Any]],
    ) -> bool:
        allowed = self._allowed_titles(tool_results)

        if not allowed:
            return False

        bullet_titles = re.findall(
            r"^\s*[-*]\s+\*\*(.+?)\*\*",
            text,
            flags=re.MULTILINE,
        )

        if not bullet_titles:
            return False

        for raw_title in bullet_titles:
            title = raw_title.split(" — ")[0].strip()
            normalized = self._normalize_title(title)

            if normalized and normalized not in allowed:
                return True

        return False

    def _format_grounded_answer(self, tool_results: list[dict[str, Any]]) -> str:
        lines = ["Here are grounded recommendations retrieved from the MCP tools:\n"]
        any_items = False

        for call in tool_results:
            result = call.get("result") or {}
            items = result.get("items") or []

            if not items:
                continue

            domain = result.get("domain") or call.get("name", "").replace("recommend_", "")
            source = result.get("source") or result.get("provider") or "unknown"

            lines.append(f"**{str(domain).title()}** _(Retrieved from: {source})_")

            for item in items[:5]:
                any_items = True
                title = item.get("title") or "Untitled"
                subtitle = item.get("subtitle") or ""
                summary = (item.get("summary") or "").strip()
                item_source = item.get("source") or item.get("provider") or source
                url = item.get("url") or ""

                label = f" ({subtitle})" if subtitle else ""
                desc = f" — {summary}" if summary else ""
                link = f" [View source]({url})" if url else ""

                lines.append(
                    f"- **{title}{label}**{desc} _(Source: {item_source})_{link}"
                )

            lines.append("")

        if not any_items:
            return "I could not find usable recommendations from the MCP tools for this request."

        return "\n".join(lines).strip()

    def _format_no_results_answer(self, tool_results: list[dict[str, Any]]) -> str:
        debug_lines = [
            "I could not find usable recommendations from the MCP tools for this request.",
            "",
            "Debug summary:",
        ]

        for call in tool_results:
            result = call.get("result") or {}
            debug_lines.append(f'- Tool: `{call.get("name")}`')
            debug_lines.append(f'  - Call source: `{call.get("call_source")}`')
            debug_lines.append(
                f'  - Arguments: `{json.dumps(call.get("arguments") or {}, ensure_ascii=False)}`'
            )
            debug_lines.append(
                f'  - Provider/source: `{result.get("source") or result.get("provider")}`'
            )
            debug_lines.append(f'  - Item count: `{len(result.get("items") or [])}`')

            if call.get("intent"):
                debug_lines.append(
                    f'  - Intent: `{json.dumps(call.get("intent") or {}, ensure_ascii=False)}`'
                )

            if result.get("error"):
                debug_lines.append(f'  - Error: `{result.get("error")}`')

        return "\n".join(debug_lines)

    def _build_debug_payload(
        self,
        user_query: str,
        mode: ExperimentMode,
        target_domains: list[str],
        tool_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        calls = []

        for call in tool_results:
            result = call.get("result") or {}
            items = result.get("items") or []

            calls.append(
                {
                    "tool": call.get("name"),
                    "call_source": call.get("call_source"),
                    "arguments": call.get("arguments"),
                    "intent": call.get("intent"),
                    "provider": result.get("source") or result.get("provider"),
                    "count": len(items),
                    "titles": [item.get("title") for item in items[:10]],
                    "error": result.get("error"),
                }
            )

        return {
            "user_query": user_query,
            "mode": mode,
            "target_domains": target_domains,
            "calls": calls,
        }

    def _chunk_text(self, text: str, size: int = 120):
        for i in range(0, len(text), size):
            yield text[i:i + size]

    def _sse(self, event: str, data: Any) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n"
