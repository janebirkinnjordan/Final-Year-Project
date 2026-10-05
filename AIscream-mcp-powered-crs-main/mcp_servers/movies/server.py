import os
import re
import sys
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote_plus

import httpx
from mcp.server.fastmcp import FastMCP

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def _load_project_env() -> None:
    """Load API keys from both project-root .env and backend/.env.

    Backend settings use backend/.env, but MCP servers are separate child
    processes. Loading both locations makes the server work no matter where the
    user put TMDB_API_KEY, OMDB_API_KEY, LASTFM_API_KEY, etc.
    """
    env_paths = (PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env")

    if load_dotenv:
        for env_path in env_paths:
            load_dotenv(env_path, override=False)

    aliases = {
        "TMDB_API_KEY": ("tmdb_api_key", "TMDB_KEY"),
        "OMDB_API_KEY": ("omdb_api_key", "OMDB_KEY"),
        "LASTFM_API_KEY": ("lastfm_api_key", "LAST_FM_API_KEY", "last_fm_api_key"),
        "GOOGLE_BOOKS_API_KEY": ("google_books_api_key", "GOOGLE_BOOKS_KEY"),
    }

    for canonical, names in aliases.items():
        value = os.getenv(canonical, "")
        if not value:
            for name in names:
                value = os.getenv(name, "")
                if value:
                    os.environ[canonical] = value
                    break
        if value:
            for name in names:
                os.environ.setdefault(name, value)


_load_project_env()


mcp = FastMCP("movies")


TMDB_GENRES: Dict[str, int] = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "animated": 16,
    "cartoon": 16,
    "comedy": 35,
    "romantic comedy": 35,
    "romcom": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "historical": 36,
    "horror": 27,
    "music": 10402,
    "musical": 10402,
    "mystery": 9648,
    "romance": 10749,
    "romantic": 10749,
    "science fiction": 878,
    "sci fi": 878,
    "sci-fi": 878,
    "thriller": 53,
    "war": 10752,
    "western": 37,
}

COUNTRY_ALIASES: Dict[str, str] = {
    "malay": "MY",
    "malaysian": "MY",
    "malaysia": "MY",
    "korean": "KR",
    "korea": "KR",
    "south korean": "KR",
    "japanese": "JP",
    "japan": "JP",
    "indonesian": "ID",
    "indonesia": "ID",
    "thai": "TH",
    "thailand": "TH",
    "filipino": "PH",
    "philippines": "PH",
    "indian": "IN",
    "india": "IN",
    "chinese": "CN",
    "china": "CN",
    "hong kong": "HK",
    "taiwanese": "TW",
    "taiwan": "TW",
    "french": "FR",
    "france": "FR",
    "spanish": "ES",
    "spain": "ES",
    "mexican": "MX",
    "mexico": "MX",
}

LANGUAGE_ALIASES: Dict[str, str] = {
    "malay": "ms",
    "malaysian": "ms",
    "bahasa melayu": "ms",
    "korean": "ko",
    "japanese": "ja",
    "indonesian": "id",
    "thai": "th",
    "filipino": "tl",
    "tagalog": "tl",
    "hindi": "hi",
    "tamil": "ta",
    "chinese": "zh",
    "mandarin": "zh",
    "cantonese": "cn",
    "french": "fr",
    "spanish": "es",
    "english": "en",
}


def debug(*args: Any) -> None:
    """
    Safe MCP debug logger.

    MCP servers use stdio transport. On Windows, stderr/stdout can sometimes be
    closed or redirected by the parent process. A normal print(..., file=sys.stderr)
    can raise OSError: [Errno 9] Bad file descriptor and crash the tool.

    This logger must never crash recommendation tools.
    """
    try:
        message = " ".join(str(arg) for arg in args)
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def debug_json(label: str, payload: Any) -> None:
    try:
        debug(f"\n========== {label} ==========")
        debug(json.dumps(payload, ensure_ascii=False, indent=2))
        debug("=" * (20 + len(label)))
    except Exception:
        pass


async def fetch_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data
        return {}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _title_key(title: Any) -> str:
    text = str(title or "").lower().strip()
    text = re.sub(r"[^a-z0-9\u00c0-\u024f\u3040-\u30ff\u4e00-\u9fff가-힣]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_exclude_titles(exclude: Any) -> Set[str]:
    if isinstance(exclude, list):
        parts = [str(x) for x in exclude]
    else:
        parts = re.split(r"[,;\n]+", str(exclude or ""))

    output: Set[str] = set()
    for part in parts:
        key = _title_key(part)
        if key:
            output.add(key)
    return output


def _filter_excluded_items(items: List[Dict[str, Any]], exclude_titles: Set[str]) -> List[Dict[str, Any]]:
    if not exclude_titles:
        return items

    filtered: List[Dict[str, Any]] = []
    for item in items:
        key = _title_key(item.get("title"))
        if key and key not in exclude_titles:
            filtered.append(item)
    return filtered


def _detect_recent(query: str, recent: bool) -> bool:
    if recent:
        return True

    text = _norm(query)
    return any(
        word in text
        for word in [
            "new",
            "newest",
            "latest",
            "recent",
            "recently released",
            "current",
            "upcoming",
            "now playing",
            "2026",
        ]
    )


def _detect_year_min(query: str, provided: int) -> int:
    if provided:
        return provided

    match = re.search(r"\b(19\d{2}|20\d{2})\b", query or "")
    if match:
        return int(match.group(1))

    return 0


def _detect_year_max(query: str, year_min: int, provided: int) -> int:
    if provided:
        return provided

    if year_min:
        return year_min

    return 0


def _detect_country(query: str, provided: str) -> str:
    text = _norm(provided)

    if text:
        if len(text) == 2:
            return text.upper()
        if text in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[text]
        # Common LLM outputs such as "SOUTH KOREA" must become KR,
        # not the invalid TMDB country code "SOUTH KOREA".
        compact_aliases = {
            "south korea": "KR",
            "republic of korea": "KR",
            "united states": "US",
            "usa": "US",
            "america": "US",
            "united kingdom": "GB",
            "uk": "GB",
            "britain": "GB",
        }
        if text in compact_aliases:
            return compact_aliases[text]

    query_text = _norm(query)
    for phrase, code in sorted(COUNTRY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if phrase in query_text:
            return code

    return ""


def _detect_language(query: str, provided: str) -> str:
    text = _norm(provided)

    if text:
        if len(text) == 2:
            return text
        if text in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[text]
        country_to_language = {
            "south korea": "ko",
            "korea": "ko",
            "korean": "ko",
            "japan": "ja",
            "japanese": "ja",
            "china": "zh",
            "chinese": "zh",
            "india": "hi",
            "indian": "hi",
            "thailand": "th",
            "thai": "th",
            "indonesia": "id",
            "indonesian": "id",
        }
        if text in country_to_language:
            return country_to_language[text]

    query_text = _norm(query)
    for phrase, code in sorted(LANGUAGE_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if phrase in query_text:
            return code

    # Country words can imply original language.
    implied = {
        "south korea": "ko",
        "korea": "ko",
        "korean": "ko",
        "japan": "ja",
        "japanese": "ja",
        "china": "zh",
        "chinese": "zh",
    }
    for phrase, code in sorted(implied.items(), key=lambda x: len(x[0]), reverse=True):
        if phrase in query_text:
            return code

    return ""


def _has_strong_movie_constraints(
    genre: str = "",
    country: str = "",
    language: str = "",
    year_min: int = 0,
    year_max: int = 0,
    recent: bool = False,
    sort_by: str = "",
) -> bool:
    """True when global trending/search would likely violate the user filters."""
    return bool(
        genre
        or country
        or language
        or year_min
        or year_max
        or recent
        or _norm(sort_by) in ("new", "newest", "latest", "oldest", "earliest", "first")
    )


def _is_family_audience(query: str, audience: str) -> bool:
    text = f"{query} {audience}".lower()

    return any(
        word in text
        for word in [
            "kids",
            "kid",
            "children",
            "child",
            "family",
            "family-friendly",
            "cartoon",
            "animated",
            "animation",
            "upin",
            "ipin",
        ]
    )


def _detect_genre(query: str, genre: str, mood: str, theme: str, audience: str) -> str:
    given = _norm(genre)
    if given in TMDB_GENRES:
        return given

    if given == "romantic comedy":
        return "romantic comedy"

    text = f"{query} {genre} {mood} {theme} {audience}".lower()

    for name in sorted(TMDB_GENRES.keys(), key=len, reverse=True):
        if name and name in text:
            return name

    # Small generic genre hints. This is API mapping, not arbitrary response hardcoding.
    if any(x in text for x in ["romcom", "romantic comedy"]):
        return "romantic comedy"
    if any(x in text for x in ["cartoon", "animated", "animation"]):
        return "animation"
    if any(x in text for x in ["family-friendly", "kids", "children", "child"]):
        return "family"
    if any(x in text for x in ["heartbreak", "emotional", "cry", "sad", "tearjerker", "melancholic"]):
        return "drama"
    if any(x in text for x in ["love", "romantic", "romance"]):
        return "romance"

    return ""


def _looks_like_person_query(query: str, genre: str) -> bool:
    text = _norm(query)
    if not text:
        return False

    if genre:
        return False

    blocked = [
        "sad",
        "cry",
        "family",
        "romance",
        "horror",
        "comedy",
        "action",
        "drama",
        "cartoon",
        "animation",
        "korean",
        "malay",
        "asian",
    ]

    if any(word in text for word in blocked):
        return False

    words = query.strip().split()
    if 1 <= len(words) <= 4:
        return True

    return False


def _clean_person_candidate(query: str) -> str:
    """Turn a natural request like 'ryan gosling latest movies' into 'ryan gosling'.

    The LLM should normally fill person_reference, but this server keeps a
    defensive repair so MCP still makes the correct TMDB person API call when
    the LLM only sends query.
    """
    text = str(query or "").strip()
    text = re.sub(
        r"\b(suggest|recommend|give|show|me|some|please|latest|newest|recent|new|oldest|earliest|popular|famous|best|top|movies?|films?|cinema|watch)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" -:")
    return text


def _sort_value(sort_by: str, recent: bool, year_min: int) -> str:
    sort_by = _norm(sort_by)

    if sort_by in ("new", "newest", "latest") or recent:
        return "primary_release_date.desc"

    if sort_by in ("oldest", "earliest", "first"):
        return "primary_release_date.asc"

    if sort_by in ("top_rated", "highest_rated", "best_rated", "best rated", "top rated", "rated"):
        return "vote_average.desc"

    if sort_by in ("underrated", "rare", "hidden_gem", "hidden gem", "less_popular", "less popular", "not popular"):
        return "vote_average.desc"

    if sort_by in ("popular", "famous", "well known", "well-known", "classic"):
        return "popularity.desc"

    # Default mixed/relevance should not be pure popularity.
    return "vote_count.desc"


def _apply_sort_filters(params: Dict[str, Any], sort_by: str, country: str, language: str) -> None:
    sort_mode = _norm(sort_by)

    if sort_mode in ("top_rated", "highest_rated", "best_rated", "best rated", "top rated", "rated"):
        params.setdefault("vote_count.gte", 300 if not (country or language) else 30)

    if sort_mode in ("underrated", "rare", "hidden_gem", "hidden gem", "less_popular", "less popular", "not popular"):
        params["vote_count.gte"] = 80 if not (country or language) else 10
        params["vote_count.lte"] = 2500 if not (country or language) else 1200
        params["popularity.lte"] = 80 if not (country or language) else 60

    if sort_mode in ("", "relevance", "mixed", "mix", "balanced"):
        params.setdefault("vote_count.gte", 80 if not (country or language) else 10)


def _clean_query(query: str, genre: str, mood: str, theme: str, audience: str) -> str:
    text = str(query or "").strip()
    text = re.sub(r"\b(suggest|recommend|give me|show me|movies?|films?|watch|to watch)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        text = "movie"

    return text


def _movie_retrieval_text(
    query: str,
    genre: str = "",
    mood: str = "",
    theme: str = "",
    audience: str = "",
    transfer_terms: str = "",
    reference_title: str = "",
    reference_creator: str = "",
) -> str:
    parts = [
        query,
        genre,
        mood,
        theme,
        audience,
        transfer_terms,
        reference_title,
        reference_creator,
    ]

    text = " ".join(str(p).strip() for p in parts if str(p).strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _discover_params(
    api_key: str,
    genre_id: int,
    recent: bool,
    year_min: int,
    year_max: int,
    country: str,
    language: str,
    sort_by: str,
    include_adult: bool,
    family_audience: bool,
    strict_recent_window: bool = True,
) -> Dict[str, Any]:
    today = date.today()

    params: Dict[str, Any] = {
        "api_key": api_key,
        "page": 1,
        "sort_by": _sort_value(sort_by, recent, year_min),
        "include_adult": "true" if include_adult else "false",
        "include_video": "false",
        "language": "en-US",
        "primary_release_date.lte": today.isoformat(),
    }

    if not recent and not year_min and not year_max:
        params["vote_count.gte"] = 10 if (country or language) else 80

    _apply_sort_filters(params, sort_by, country, language)

    if genre_id:
        params["with_genres"] = genre_id

    if country:
        params["with_origin_country"] = country

    if language:
        params["with_original_language"] = language

    if family_audience:
        params["certification_country"] = "US"
        params["certification.lte"] = "PG"

    if year_min:
        params["primary_release_date.gte"] = f"{int(year_min)}-01-01"

    if year_max:
        params["primary_release_date.lte"] = f"{int(year_max)}-12-31"
    elif recent and strict_recent_window:
        params["primary_release_date.gte"] = f"{today.year - 3}-01-01"
        params["primary_release_date.lte"] = today.isoformat()

    return params


def _format_tmdb_results(
    data: Dict[str, Any],
    limit: int,
    source_label: str,
    exclude_titles: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    exclude_titles = exclude_titles or set()
    results = data.get("results", [])
    items: List[Dict[str, Any]] = []

    for raw in results:
        if not isinstance(raw, dict):
            continue

        title = raw.get("title") or raw.get("name") or ""
        if not title:
            continue

        title_key = _title_key(title)
        if title_key in exclude_titles:
            continue

        release_date = raw.get("release_date") or raw.get("first_air_date") or ""
        year = release_date[:4] if release_date else ""
        overview = raw.get("overview") or ""
        rating = raw.get("vote_average")

        summary_parts: List[str] = []
        if rating:
            summary_parts.append(f"TMDB rating: {rating}.")
        if overview:
            summary_parts.append(overview)

        item = {
            "domain": "movie",
            "title": title,
            "subtitle": year,
            "summary": " ".join(summary_parts).strip(),
            "source": source_label,
            "provider": "TMDB",
            "mcp_server": "movies",
            "external_id": raw.get("id"),
            "url": f"https://www.themoviedb.org/movie/{raw.get('id')}" if raw.get("id") else "",
        }

        items.append(item)

        if len(items) >= limit:
            break

    return items


def _filter_items_by_year(items: List[Dict[str, Any]], year_min: int, year_max: int) -> List[Dict[str, Any]]:
    if not year_min and not year_max:
        return items

    filtered: List[Dict[str, Any]] = []

    for item in items:
        year_text = str(item.get("subtitle") or "").strip()
        try:
            year = int(year_text[:4])
        except Exception:
            continue

        if year_min and year < year_min:
            continue
        if year_max and year > year_max:
            continue

        filtered.append(item)

    return filtered


def _interleave_unique_items(groups: List[List[Dict[str, Any]]], limit: int) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    max_len = max((len(group) for group in groups), default=0)

    for index in range(max_len):
        for group in groups:
            if index >= len(group):
                continue

            item = group[index]
            key = _title_key(item.get("title"))

            if not key or key in seen:
                continue

            seen.add(key)
            output.append(item)

            if len(output) >= limit:
                return output

    return output[:limit]


async def _fetch_tmdb_strategy(
    label: str,
    url: str,
    params: Dict[str, Any],
    limit: int,
    exclude_titles: Set[str],
    year_min: int = 0,
    year_max: int = 0,
    apply_year_filter: bool = True,
    max_pages: int = 5,
) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []

    safe_params = dict(params)
    safe_params["api_key"] = "***"
    debug("\n========== TMDB STRATEGY ==========")
    debug("label:", label)
    debug("url:", url)
    debug("params:", safe_params)
    debug("===================================")

    for page in range(1, max_pages + 1):
        page_params = dict(params)
        if "page" in page_params:
            page_params["page"] = page

        try:
            data = await fetch_json(url, page_params)
        except Exception as exc:
            debug("TMDB strategy failed:", label, repr(exc))
            break

        count = len(data.get("results", [])) if isinstance(data, dict) else 0
        first_titles = [x.get("title") for x in data.get("results", [])[:5]] if isinstance(data, dict) else []

        debug("\n========== TMDB RESPONSE ==========")
        debug("label:", label)
        debug("page:", page)
        debug("count:", count)
        debug("first_titles:", first_titles)
        debug("===================================")

        page_items = _format_tmdb_results(
            data,
            limit=20,
            source_label=label,
            exclude_titles=exclude_titles,
        )

        if apply_year_filter:
            page_items = _filter_items_by_year(page_items, year_min, year_max)

        existing_keys = {_title_key(existing.get("title")) for existing in collected}

        for item in page_items:
            key = _title_key(item.get("title"))
            if key and key not in existing_keys:
                collected.append(item)
                existing_keys.add(key)

        if len(collected) >= limit or count == 0:
            break

    return collected[:limit]


async def _tmdb_person_movies(
    api_key: str,
    person_query: str,
    limit: int,
    recent: bool,
    year_min: int,
    year_max: int,
    sort_by: str,
    include_adult: bool,
) -> Dict[str, Any]:
    search_data = await fetch_json(
        "https://api.themoviedb.org/3/search/person",
        {
            "api_key": api_key,
            "query": person_query,
            "include_adult": "true" if include_adult else "false",
            "language": "en-US",
            "page": 1,
        },
    )

    people = search_data.get("results", [])
    if not people:
        return {
            "domain": "movie",
            "items": [],
            "source": "TMDB person search no results",
            "provider": "TMDB",
        }

    person = people[0]
    person_id = person.get("id")
    person_name = person.get("name") or person_query

    if not person_id:
        return {
            "domain": "movie",
            "items": [],
            "source": "TMDB person search no usable id",
            "provider": "TMDB",
        }

    credits = await fetch_json(
        f"https://api.themoviedb.org/3/person/{person_id}/movie_credits",
        {
            "api_key": api_key,
            "language": "en-US",
        },
    )

    raw_movies = []
    for section in ["cast", "crew"]:
        values = credits.get(section, [])
        if isinstance(values, list):
            raw_movies.extend(values)

    seen_ids: Set[str] = set()
    movies: List[Dict[str, Any]] = []

    for raw in raw_movies:
        if not isinstance(raw, dict):
            continue

        movie_id = str(raw.get("id") or "")
        if movie_id and movie_id in seen_ids:
            continue
        if movie_id:
            seen_ids.add(movie_id)

        title = raw.get("title") or raw.get("original_title") or ""
        if not title:
            continue

        release_date = raw.get("release_date") or ""
        year = 0
        if release_date[:4].isdigit():
            year = int(release_date[:4])

        if year_min and year and year < year_min:
            continue
        if year_max and year and year > year_max:
            continue

        movies.append(raw)

    sort_mode = _norm(sort_by)

    if recent or sort_mode in ("new", "newest", "latest"):
        # For latest person requests, use person credits sorted by real release date.
        # Avoid generic /discover because that returns random newest TMDB entries.
        movies = [movie for movie in movies if movie.get("release_date")]
        movies.sort(key=lambda x: x.get("release_date") or "", reverse=True)
        source = "TMDB person newest credits"
    elif sort_mode in ("oldest", "earliest", "first"):
        movies = [movie for movie in movies if movie.get("release_date")]
        movies.sort(key=lambda x: x.get("release_date") or "")
        source = "TMDB person oldest credits"
    elif sort_mode in ("top_rated", "rated", "best"):
        today_text = date.today().isoformat()
        released = [
            movie for movie in movies
            if movie.get("release_date") and movie.get("release_date") <= today_text
        ]
        movies = released or movies
        movies.sort(
            key=lambda x: (
                float(x.get("vote_average") or 0),
                float(x.get("vote_count") or 0),
            ),
            reverse=True,
        )
        source = "TMDB person rated credits"
    else:
        # General person recommendations should not be dominated by unreleased
        # future titles. Prefer released, well-known credits.
        today_text = date.today().isoformat()
        released = [
            movie for movie in movies
            if movie.get("release_date") and movie.get("release_date") <= today_text
        ]
        movies = released or movies
        movies.sort(
            key=lambda x: (
                float(x.get("vote_count") or 0),
                float(x.get("vote_average") or 0),
                float(x.get("popularity") or 0),
            ),
            reverse=True,
        )
        source = "TMDB person released relevance credits"

    data = {"results": movies}
    items = _format_tmdb_results(data, limit=limit, source_label=source)

    return {
        "domain": "movie",
        "items": items,
        "source": source,
        "provider": "TMDB",
        "person": person_name,
    }


async def _tmdb_retrieve(
    query: str,
    limit: int,
    recent: bool,
    genre: str,
    year_min: int,
    year_max: int,
    country: str,
    language: str,
    sort_by: str,
    include_adult: bool,
    family_audience: bool,
    exclude_titles: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    api_key = os.getenv("TMDB_API_KEY", "")

    if not api_key:
        return {
            "domain": "movie",
            "items": [],
            "source": "TMDB",
            "provider": "TMDB",
            "error": "TMDB_API_KEY is missing or not loaded.",
        }

    exclude_titles = exclude_titles or set()
    genre_id = TMDB_GENRES.get(_norm(genre), 0)
    person_query = _looks_like_person_query(query, genre)
    sort_mode = _norm(sort_by)

    debug("### USING NEW _tmdb_retrieve MIXED VERSION ###")

    if person_query:
        person_result = await _tmdb_person_movies(
            api_key=api_key,
            person_query=query,
            limit=limit,
            recent=recent,
            year_min=year_min,
            year_max=year_max,
            sort_by=sort_by,
            include_adult=include_adult,
        )

        if person_result.get("items"):
            person_result["items"] = _filter_excluded_items(
                person_result.get("items") or [],
                exclude_titles,
            )[:limit]
            return person_result

    groups: List[List[Dict[str, Any]]] = []
    strong_constraints = _has_strong_movie_constraints(
        genre=genre,
        country=country,
        language=language,
        year_min=year_min,
        year_max=year_max,
        recent=recent,
        sort_by=sort_by,
    )

    # For constrained requests such as "newly released Korean horror movies",
    # do NOT run broad search/trending first. Those endpoints are global and will
    # pollute the answer with unrelated trending titles.
    if query and not strong_constraints:
        search_items = await _fetch_tmdb_strategy(
            label="TMDB search relevance",
            url="https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": api_key,
                "query": query,
                "page": 1,
                "include_adult": "true" if include_adult else "false",
                "language": "en-US",
            },
            limit=limit,
            exclude_titles=exclude_titles,
            year_min=year_min,
            year_max=year_max,
            apply_year_filter=True,
            max_pages=3,
        )

        if search_items:
            groups.append(search_items)

    discover_params = _discover_params(
        api_key=api_key,
        genre_id=genre_id,
        recent=recent,
        year_min=year_min,
        year_max=year_max,
        country=country,
        language=language,
        sort_by=sort_by or "relevance",
        include_adult=include_adult,
        family_audience=family_audience,
        strict_recent_window=True,
    )

    discover_items = await _fetch_tmdb_strategy(
        label="TMDB discover structured",
        url="https://api.themoviedb.org/3/discover/movie",
        params=discover_params,
        limit=limit,
        exclude_titles=exclude_titles,
        year_min=year_min,
        year_max=year_max,
        apply_year_filter=True,
        max_pages=5,
    )

    if discover_items:
        groups.append(discover_items)

    if sort_mode in ("", "relevance", "mixed", "mix", "balanced") and not strong_constraints:
        top_rated_params = _discover_params(
            api_key=api_key,
            genre_id=genre_id,
            recent=False,
            year_min=year_min,
            year_max=year_max,
            country=country,
            language=language,
            sort_by="top_rated",
            include_adult=include_adult,
            family_audience=family_audience,
            strict_recent_window=False,
        )

        top_rated_items = await _fetch_tmdb_strategy(
            label="TMDB discover top-rated mix",
            url="https://api.themoviedb.org/3/discover/movie",
            params=top_rated_params,
            limit=limit,
            exclude_titles=exclude_titles,
            year_min=year_min,
            year_max=year_max,
            apply_year_filter=True,
            max_pages=3,
        )

        if top_rated_items:
            groups.append(top_rated_items)

        underrated_params = _discover_params(
            api_key=api_key,
            genre_id=genre_id,
            recent=False,
            year_min=year_min,
            year_max=year_max,
            country=country,
            language=language,
            sort_by="underrated",
            include_adult=include_adult,
            family_audience=family_audience,
            strict_recent_window=False,
        )

        underrated_items = await _fetch_tmdb_strategy(
            label="TMDB discover underrated mix",
            url="https://api.themoviedb.org/3/discover/movie",
            params=underrated_params,
            limit=limit,
            exclude_titles=exclude_titles,
            year_min=year_min,
            year_max=year_max,
            apply_year_filter=True,
            max_pages=3,
        )

        if underrated_items:
            groups.append(underrated_items)

    # Trending is allowed only for unconstrained recent requests.
    # For constrained requests, use trending only as a last fallback after
    # discover returned nothing.
    if recent and not strong_constraints:
        trending_items = await _fetch_tmdb_strategy(
            label="TMDB trending week",
            url="https://api.themoviedb.org/3/trending/movie/week",
            params={
                "api_key": api_key,
                "page": 1,
                "language": "en-US",
            },
            limit=limit,
            exclude_titles=exclude_titles,
            year_min=year_min,
            year_max=year_max,
            apply_year_filter=False,
            max_pages=3,
        )

        if trending_items:
            groups.append(trending_items)

    if recent and strong_constraints and not discover_items:
        fallback_items = await _fetch_tmdb_strategy(
            label="TMDB trending week fallback - structured filters had no results",
            url="https://api.themoviedb.org/3/trending/movie/week",
            params={
                "api_key": api_key,
                "page": 1,
                "language": "en-US",
            },
            limit=limit,
            exclude_titles=exclude_titles,
            year_min=year_min,
            year_max=year_max,
            apply_year_filter=False,
            max_pages=1,
        )

        if fallback_items:
            groups.append(fallback_items)

    mixed_items = _interleave_unique_items(groups, limit)

    if mixed_items:
        return {
            "domain": "movie",
            "items": mixed_items[:limit],
            "source": "TMDB mixed retrieval",
            "provider": "TMDB",
        }

    return {
        "domain": "movie",
        "items": [],
        "source": "TMDB no results",
        "provider": "TMDB",
        "error": "TMDB returned no usable movie candidates.",
    }


async def _backup_omdb(query: str, limit: int, year_min: int, year_max: int) -> Dict[str, Any]:
    api_key = os.getenv("OMDB_API_KEY", "")
    if not api_key:
        return {
            "domain": "movie",
            "items": [],
            "source": "OMDb",
            "provider": "OMDb",
            "error": "OMDB_API_KEY is missing or not loaded.",
        }

    try:
        data = await fetch_json(
            "https://www.omdbapi.com/",
            {
                "apikey": api_key,
                "s": query,
                "type": "movie",
                "page": 1,
            },
        )
    except Exception as exc:
        return {
            "domain": "movie",
            "items": [],
            "source": "OMDb",
            "provider": "OMDb",
            "error": str(exc),
        }

    results = data.get("Search") or []
    items: List[Dict[str, Any]] = []

    for raw in results:
        title = raw.get("Title") or ""
        year_text = raw.get("Year") or ""

        year = 0
        match = re.search(r"\b(19\d{2}|20\d{2})\b", year_text)
        if match:
            year = int(match.group(1))

        if year_min and year and year < year_min:
            continue
        if year_max and year and year > year_max:
            continue

        item = {
            "domain": "movie",
            "title": title,
            "subtitle": year_text,
            "summary": f"OMDb result: {raw.get('Type', 'movie')}.",
            "source": "OMDb search",
            "provider": "OMDb",
            "mcp_server": "movies",
            "external_id": raw.get("imdbID"),
            "url": f"https://www.imdb.com/title/{raw.get('imdbID')}/" if raw.get("imdbID") else "",
        }

        items.append(item)

        if len(items) >= limit:
            break

    if items:
        return {
            "domain": "movie",
            "items": items,
            "source": "OMDb search",
            "provider": "OMDb",
        }

    return {
        "domain": "movie",
        "items": [],
        "source": "OMDb no results",
        "provider": "OMDb",
        "error": "OMDb returned no results.",
    }


@mcp.tool(
    description=(
        "Recommend movie candidates from TMDB with OMDb backup. "
        "LLM supplies intent variables. MCP retrieves grounded candidates. "
        "This server uses mixed retrieval, not popularity only."
    )
)
async def recommend_movies(
    query: str,
    limit: int = 5,
    recent: bool = False,
    genre: str = "",
    mood: str = "",
    theme: str = "",
    audience: str = "",
    year_min: int = 0,
    year_max: int = 0,
    country: str = "",
    language: str = "",
    person_reference: str = "",
    actor_reference: str = "",
    director_reference: str = "",
    reference_movie: str = "",
    source_domain: str = "",
    reference_title: str = "",
    reference_creator: str = "",
    transfer_terms: str = "",
    sort_by: str = "",
    include_adult: bool = False,
    exclude: str = "",
) -> Dict[str, Any]:
    safe_limit = max(1, min(int(limit or 5), 40))

    query = (query or "").strip()
    genre = (genre or "").strip()
    mood = (mood or "").strip()
    theme = (theme or "").strip()
    audience = (audience or "").strip().lower()
    country = (country or "").strip()
    language = (language or "").strip()
    person_reference = (person_reference or "").strip()
    actor_reference = (actor_reference or "").strip()
    director_reference = (director_reference or "").strip()
    reference_movie = (reference_movie or "").strip()
    source_domain = (source_domain or "").strip().lower()
    reference_title = (reference_title or "").strip()
    reference_creator = (reference_creator or "").strip()
    transfer_terms = str(transfer_terms or "").strip()
    sort_by = (sort_by or "").strip().lower()
    exclude = str(exclude or "").strip()

    person_query = person_reference or actor_reference or director_reference

    # Defensive MCP-side intent repair: if the LLM did not fill
    # person_reference but the query is clearly a person request like
    # "ryan gosling latest movies", clean it and route to TMDB person credits.
    if not person_query:
        cleaned_person_candidate = _clean_person_candidate(query)
        if cleaned_person_candidate and _looks_like_person_query(cleaned_person_candidate, ""):
            person_query = cleaned_person_candidate

    base_query = person_query or reference_movie or query

    detected_recent = _detect_recent(query, recent)
    detected_year_min = _detect_year_min(query, _int_value(year_min))
    detected_year_max = _detect_year_max(query, detected_year_min, _int_value(year_max))
    detected_country = _detect_country(query, country)
    detected_language = _detect_language(query, language)
    family_audience = _is_family_audience(query, audience)
    detected_genre = _detect_genre(query, genre, mood, theme, audience)

    if detected_recent:
        sort_by = "newest"
    elif not sort_by:
        sort_by = "relevance"

    clean_query = _clean_query(base_query, detected_genre, mood, theme, audience)

    retrieval_genre = ""
    if detected_genre and detected_genre.lower() not in (clean_query or query).lower():
        retrieval_genre = detected_genre

    retrieval_query = _movie_retrieval_text(
        query=clean_query or query,
        genre=retrieval_genre,
        mood=mood,
        theme=theme,
        audience=audience,
        transfer_terms=transfer_terms,
        reference_title=reference_title,
        reference_creator=reference_creator,
    )

    exclude_titles = _split_exclude_titles(exclude)

    debug("\n========== MOVIE MCP TOOL ==========")
    debug("query:", query)
    debug("clean_query:", clean_query)
    debug("retrieval_query:", retrieval_query)
    debug("limit:", safe_limit)
    debug("recent:", detected_recent)
    debug("genre:", detected_genre)
    debug("mood:", mood)
    debug("theme:", theme)
    debug("audience:", audience)
    debug("source_domain:", source_domain)
    debug("reference_title:", reference_title)
    debug("reference_creator:", reference_creator)
    debug("transfer_terms:", transfer_terms)
    debug("family_audience:", family_audience)
    debug("year_min:", detected_year_min)
    debug("year_max:", detected_year_max)
    debug("country:", detected_country)
    debug("language:", detected_language)
    debug("person_reference:", person_query)
    debug("sort_by:", sort_by)
    debug("include_adult:", include_adult)
    debug("exclude:", exclude)
    debug("exclude_titles:", sorted(exclude_titles))
    debug("TMDB key loaded:", bool(os.getenv("TMDB_API_KEY", "")))
    debug("OMDB key loaded:", bool(os.getenv("OMDB_API_KEY", "")))
    debug("====================================")

    retrieve_query = person_query or retrieval_query or clean_query or query

    primary = await _tmdb_retrieve(
        query=retrieve_query,
        limit=safe_limit,
        recent=detected_recent,
        genre=detected_genre,
        year_min=detected_year_min,
        year_max=detected_year_max,
        country=detected_country,
        language=detected_language,
        sort_by=sort_by,
        include_adult=bool(include_adult),
        family_audience=family_audience,
        exclude_titles=exclude_titles,
    )

    detected_payload = {
        "recent": detected_recent,
        "genre": detected_genre,
        "mood": mood,
        "theme": theme,
        "source_domain": source_domain,
        "reference_title": reference_title,
        "reference_creator": reference_creator,
        "transfer_terms": transfer_terms,
        "retrieval_query": retrieval_query,
        "year_min": detected_year_min,
        "year_max": detected_year_max,
        "country": detected_country,
        "language": detected_language,
        "audience": audience,
        "sort_by": sort_by,
        "exclude_count": len(exclude_titles),
    }

    if primary and primary.get("items"):
        primary["detected"] = detected_payload
        debug_json("MOVIES TOOL OUTPUT", {
            "source": primary.get("source"),
            "provider": primary.get("provider"),
            "count": len(primary.get("items") or []),
            "detected": primary.get("detected"),
            "items": [
                {
                    "title": i.get("title"),
                    "subtitle": i.get("subtitle"),
                    "source": i.get("source"),
                    "url": i.get("url"),
                }
                for i in (primary.get("items") or [])[:10]
            ],
        })
        return primary

    backup_query = retrieval_query or detected_genre or clean_query or mood or theme or "movie"
    backup = await _backup_omdb(
        backup_query,
        safe_limit,
        detected_year_min,
        detected_year_max,
    )

    if backup and backup.get("items"):
        backup["items"] = _filter_excluded_items(backup.get("items") or [], exclude_titles)
        backup["detected"] = detected_payload
        return backup

    output = {
        "domain": "movie",
        "items": [],
        "source": "TMDB/OMDb no results",
        "provider": "TMDB/OMDb",
        "detected": detected_payload,
        "error": {
            "tmdb": primary.get("error") if isinstance(primary, dict) else None,
            "omdb": backup.get("error") if isinstance(backup, dict) else None,
        },
    }

    debug_json("MOVIES TOOL OUTPUT", output)
    return output


if __name__ == "__main__":
    mcp.run()
