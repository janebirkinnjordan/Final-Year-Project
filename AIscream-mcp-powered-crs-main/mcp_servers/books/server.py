import os
import re
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Set

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


mcp = FastMCP("books")


LANGUAGE_ALIASES: Dict[str, str] = {
    "english": "en",
    "malay": "ms",
    "malaysian": "ms",
    "korean": "ko",
    "japanese": "ja",
    "indonesian": "id",
    "thai": "th",
    "chinese": "zh",
    "french": "fr",
    "spanish": "es",
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

    return [
        item for item in items
        if _title_key(item.get("title")) not in exclude_titles
    ]


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _normalize_language(language: str) -> str:
    language = (language or "").strip().lower()
    if not language:
        return ""
    if len(language) == 2:
        return language
    return LANGUAGE_ALIASES.get(language, language)


def _detect_recent(query: str, recent: bool) -> bool:
    if recent:
        return True

    text = (query or "").lower()
    return any(word in text for word in ["latest", "new", "newest", "recent", "recently published", "2026"])


def _build_books_query(
    query: str,
    genre: str,
    mood: str,
    theme: str,
    audience: str,
    author_reference: str,
    publisher: str,
    book_subject: str,
    isbn: str,
) -> str:
    query = (query or "").strip()
    genre = (genre or "").strip()
    mood = (mood or "").strip()
    theme = (theme or "").strip()
    audience = (audience or "").strip()
    author_reference = (author_reference or "").strip()
    publisher = (publisher or "").strip()
    book_subject = (book_subject or "").strip()
    isbn = (isbn or "").strip()

    if isbn:
        return f"isbn:{isbn}"

    parts: List[str] = []

    if query:
        parts.append(query)

    if book_subject:
        parts.append(f"subject:{book_subject}")
    elif genre:
        parts.append(f"subject:{genre}")

    if author_reference:
        parts.append(f"inauthor:{author_reference}")

    if publisher:
        parts.append(f"inpublisher:{publisher}")

    # Let LLM variables influence retrieval without hardcoding exact user phrases.
    semantic_parts = [mood, theme, audience]
    semantic_text = " ".join(x for x in semantic_parts if x).strip()
    if semantic_text:
        parts.append(semantic_text)

    result = " ".join(parts).strip()
    result = re.sub(r"\s+", " ", result)

    return result or "books"


def _sort_google_books(sort_by: str, recent: bool) -> str:
    sort_by = (sort_by or "").strip().lower()

    if recent or sort_by in ("new", "newest", "latest"):
        return "newest"

    # Google Books only supports relevance/newest.
    return "relevance"


def _format_google_books_items(
    data: Dict[str, Any],
    page_count: int,
    page_count_min: int,
    page_count_max: int,
) -> List[Dict[str, Any]]:
    raw_items = data.get("items") or []
    items: List[Dict[str, Any]] = []

    for raw in raw_items:
        if not isinstance(raw, dict):
            continue

        volume = raw.get("volumeInfo") or {}

        title = volume.get("title") or ""
        if not title:
            continue

        authors = volume.get("authors") or []
        authors_text = ", ".join(authors) if isinstance(authors, list) else str(authors or "")

        published_date = volume.get("publishedDate") or ""
        year = published_date[:4] if published_date else ""

        categories = volume.get("categories") or []
        category_text = ", ".join(categories[:2]) if isinstance(categories, list) else str(categories or "")

        description = volume.get("description") or ""
        description = re.sub(r"\s+", " ", description).strip()
        if len(description) > 260:
            description = description[:260].rstrip() + "..."

        pages = _int_value(volume.get("pageCount"))

        if page_count and pages and pages != page_count:
            continue
        if page_count_min and pages and pages < page_count_min:
            continue
        if page_count_max and pages and pages > page_count_max:
            continue

        summary_parts: List[str] = []
        if pages:
            summary_parts.append(f"Pages: {pages}.")
        if category_text:
            summary_parts.append(f"Category: {category_text}.")
        if description:
            summary_parts.append(description)

        subtitle_bits: List[str] = []
        if authors_text:
            subtitle_bits.append(authors_text)
        if year:
            subtitle_bits.append(year)

        item = {
            "domain": "book",
            "title": title,
            "subtitle": " | ".join(subtitle_bits),
            "summary": " ".join(summary_parts).strip(),
            "source": "Google Books",
            "provider": "Google Books",
            "mcp_server": "books",
            "external_id": raw.get("id"),
            "url": volume.get("infoLink") or raw.get("selfLink") or "",
            "page_count": pages,
        }

        items.append(item)

    return items


def _clean_openlibrary_query(query: str) -> str:
    text = str(query or "").strip()
    text = re.sub(r"\b(latest|newest|recent|recently published|new books?|books?|book|suggest|recommend|show me|give me)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text or str(query or "").strip() or "books"


def _format_openlibrary_items(
    data: Dict[str, Any],
    page_count: int,
    page_count_min: int,
    page_count_max: int,
) -> List[Dict[str, Any]]:
    docs = data.get("docs") or []
    items: List[Dict[str, Any]] = []

    for raw in docs:
        if not isinstance(raw, dict):
            continue

        title = raw.get("title") or ""
        if not title:
            continue

        authors = raw.get("author_name") or []
        authors_text = ", ".join(authors[:3]) if isinstance(authors, list) else str(authors or "")

        first_publish_year = raw.get("first_publish_year") or ""
        publish_years = raw.get("publish_year") or []
        latest_year = ""
        if isinstance(publish_years, list) and publish_years:
            numeric_years = [int(y) for y in publish_years if str(y).isdigit()]
            if numeric_years:
                latest_year = str(max(numeric_years))

        year_text = latest_year or str(first_publish_year or "")

        pages = _int_value(raw.get("number_of_pages_median"))
        if page_count and pages and pages != page_count:
            continue
        if page_count_min and pages and pages < page_count_min:
            continue
        if page_count_max and pages and pages > page_count_max:
            continue

        subjects = raw.get("subject") or []
        subject_text = ", ".join(subjects[:3]) if isinstance(subjects, list) else str(subjects or "")

        key = raw.get("key") or ""
        olid = raw.get("cover_edition_key") or raw.get("edition_key", [""])
        if isinstance(olid, list):
            olid = olid[0] if olid else ""

        summary_parts: List[str] = []
        if pages:
            summary_parts.append(f"Pages: {pages}.")
        if subject_text:
            summary_parts.append(f"Subjects: {subject_text}.")
        if first_publish_year:
            summary_parts.append(f"First published: {first_publish_year}.")

        subtitle_bits: List[str] = []
        if authors_text:
            subtitle_bits.append(authors_text)
        if year_text:
            subtitle_bits.append(str(year_text))

        items.append(
            {
                "domain": "book",
                "title": title,
                "subtitle": " | ".join(subtitle_bits),
                "summary": " ".join(summary_parts).strip(),
                "source": "OpenLibrary",
                "provider": "OpenLibrary",
                "mcp_server": "books",
                "external_id": key or olid,
                "url": f"https://openlibrary.org{key}" if key else (f"https://openlibrary.org/books/{olid}" if olid else ""),
                "page_count": pages,
            }
        )

    return items


async def _openlibrary_books(
    retrieval_query: str,
    limit: int,
    recent: bool,
    sort_by: str,
    language: str,
    page_count: int,
    page_count_min: int,
    page_count_max: int,
    exclude_titles: Set[str],
) -> List[Dict[str, Any]]:
    q = _clean_openlibrary_query(retrieval_query)

    params: Dict[str, Any] = {
        "q": q,
        "limit": min(max(limit + len(exclude_titles), limit), 100),
    }

    if recent or str(sort_by or "").lower() in ("new", "newest", "latest"):
        params["sort"] = "new"

    if language:
        params["language"] = language

    data = await fetch_json("https://openlibrary.org/search.json", params)
    items = _format_openlibrary_items(
        data,
        page_count=page_count,
        page_count_min=page_count_min,
        page_count_max=page_count_max,
    )
    items = _filter_excluded_items(items, exclude_titles)
    return items[:limit]


@mcp.tool(
    description=(
        "Recommend book candidates from Google Books with OpenLibrary fallback. "
        "LLM supplies semantic intent variables; this MCP maps them to grounded book retrieval."
    )
)
async def recommend_books(
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
    author_reference: str = "",
    publisher: str = "",
    book_subject: str = "",
    isbn: str = "",
    page_count: int = 0,
    page_count_min: int = 0,
    page_count_max: int = 0,
    sort_by: str = "",
    exclude: str = "",
    include_adult: bool = False,
) -> Dict[str, Any]:
    safe_limit = max(1, min(int(limit or 5), 40))

    query = (query or "").strip()
    genre = (genre or "").strip()
    mood = (mood or "").strip()
    theme = (theme or "").strip()
    audience = (audience or "").strip()
    country = (country or "").strip()
    language = _normalize_language(language)
    author_reference = (author_reference or "").strip()
    publisher = (publisher or "").strip()
    book_subject = (book_subject or "").strip()
    isbn = (isbn or "").strip()
    sort_by = (sort_by or "").strip().lower()
    exclude = str(exclude or "").strip()

    detected_recent = _detect_recent(query, recent)

    if detected_recent:
        sort_by = "newest"
    elif not sort_by:
        sort_by = "relevance"

    page_count = _int_value(page_count)
    page_count_min = _int_value(page_count_min)
    page_count_max = _int_value(page_count_max)
    year_min = _int_value(year_min)
    year_max = _int_value(year_max)

    retrieval_query = _build_books_query(
        query=query,
        genre=genre,
        mood=mood,
        theme=theme,
        audience=audience,
        author_reference=author_reference,
        publisher=publisher,
        book_subject=book_subject,
        isbn=isbn,
    )

    order_by = _sort_google_books(sort_by, detected_recent)
    exclude_titles = _split_exclude_titles(exclude)

    params: Dict[str, Any] = {
        "q": retrieval_query,
        "maxResults": min(max(safe_limit + len(exclude_titles), safe_limit), 40),
        "printType": "books",
        "orderBy": order_by,
    }

    if language:
        params["langRestrict"] = language

    debug("\n========== BOOK MCP TOOL ==========")
    debug("query:", query)
    debug("retrieval_query:", retrieval_query)
    debug("limit:", safe_limit)
    debug("recent:", detected_recent)
    debug("genre:", genre)
    debug("mood:", mood)
    debug("theme:", theme)
    debug("audience:", audience)
    debug("book_subject:", book_subject)
    debug("author_reference:", author_reference)
    debug("language:", language)
    debug("sort_by:", sort_by)
    debug("order_by:", order_by)
    debug("page_count:", page_count)
    debug("page_count_min:", page_count_min)
    debug("page_count_max:", page_count_max)
    debug("exclude_titles:", sorted(exclude_titles))
    debug("===================================")

    google_error = ""

    try:
        data = await fetch_json("https://www.googleapis.com/books/v1/volumes", params)
        items = _format_google_books_items(
            data,
            page_count=page_count,
            page_count_min=page_count_min,
            page_count_max=page_count_max,
        )
    except Exception as exc:
        google_error = str(exc)
        debug("Google Books failed; trying OpenLibrary fallback:", google_error)
        items = []

    if not items:
        try:
            items = await _openlibrary_books(
                retrieval_query=retrieval_query,
                limit=safe_limit,
                recent=detected_recent,
                sort_by=sort_by,
                language=language,
                page_count=page_count,
                page_count_min=page_count_min,
                page_count_max=page_count_max,
                exclude_titles=exclude_titles,
            )
            if items:
                output = {
                    "domain": "book",
                    "items": items,
                    "source": "OpenLibrary fallback" if google_error else "OpenLibrary",
                    "provider": "OpenLibrary",
                    "detected": {
                        "recent": detected_recent,
                        "genre": genre,
                        "mood": mood,
                        "theme": theme,
                        "audience": audience,
                        "book_subject": book_subject,
                        "author_reference": author_reference,
                        "year_min": year_min,
                        "year_max": year_max,
                        "sort_by": sort_by or ("newest" if detected_recent else "relevance"),
                        "page_count": page_count,
                        "page_count_min": page_count_min,
                        "page_count_max": page_count_max,
                        "retrieval_query": retrieval_query,
                        "exclude_count": len(exclude_titles),
                    },
                    "google_books_error": google_error or None,
                }
                debug_json("BOOKS TOOL OUTPUT", {
                    "count": len(items),
                    "source": output["source"],
                    "detected": output["detected"],
                    "items": [
                        {
                            "title": item.get("title"),
                            "subtitle": item.get("subtitle"),
                            "source": item.get("source"),
                        }
                        for item in items[:10]
                    ],
                })
                return output
        except Exception as exc:
            openlibrary_error = str(exc)
            if google_error:
                return {
                    "domain": "book",
                    "items": [],
                    "source": "Google Books and OpenLibrary error",
                    "provider": "Google Books/OpenLibrary",
                    "detected": {
                        "recent": detected_recent,
                        "genre": genre,
                        "mood": mood,
                        "theme": theme,
                        "audience": audience,
                        "book_subject": book_subject,
                        "author_reference": author_reference,
                        "year_min": year_min,
                        "year_max": year_max,
                        "sort_by": sort_by or ("newest" if detected_recent else "relevance"),
                        "page_count": page_count,
                        "page_count_min": page_count_min,
                        "page_count_max": page_count_max,
                        "retrieval_query": retrieval_query,
                        "exclude_count": len(exclude_titles),
                    },
                    "error": {
                        "google_books": google_error,
                        "openlibrary": openlibrary_error,
                    },
                }

    if year_min or year_max:
        filtered_by_year: List[Dict[str, Any]] = []
        for item in items:
            subtitle = str(item.get("subtitle") or "")
            match = re.search(r"\b(19\d{2}|20\d{2})\b", subtitle)
            if not match:
                continue

            year = int(match.group(1))
            if year_min and year < year_min:
                continue
            if year_max and year > year_max:
                continue

            filtered_by_year.append(item)

        items = filtered_by_year

    items = _filter_excluded_items(items, exclude_titles)
    items = items[:safe_limit]

    output = {
        "domain": "book",
        "items": items,
        "source": "Google Books",
        "provider": "Google Books",
        "detected": {
            "recent": detected_recent,
            "genre": genre,
            "mood": mood,
            "theme": theme,
            "audience": audience,
            "book_subject": book_subject,
            "author_reference": author_reference,
            "year_min": year_min,
            "year_max": year_max,
            "sort_by": sort_by or ("newest" if detected_recent else "relevance"),
            "page_count": page_count,
            "page_count_min": page_count_min,
            "page_count_max": page_count_max,
            "retrieval_query": retrieval_query,
            "exclude_count": len(exclude_titles),
        },
    }

    debug_json("BOOKS TOOL OUTPUT", {
        "count": len(items),
        "detected": output["detected"],
        "items": [
            {
                "title": item.get("title"),
                "subtitle": item.get("subtitle"),
                "source": item.get("source"),
            }
            for item in items[:10]
        ],
    })

    return output


if __name__ == "__main__":
    mcp.run()
