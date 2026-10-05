import json
import os
import re
import sys
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
    """Load API keys from project root .env and backend/.env.

    MCP servers run as child processes, so they should not assume backend settings
    have already been loaded into their environment.
    """
    env_paths = (
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "backend" / ".env",
    )

    if load_dotenv:
        for env_path in env_paths:
            load_dotenv(env_path, override=False)

    aliases = {
        "LASTFM_API_KEY": (
            "lastfm_api_key",
            "LAST_FM_API_KEY",
            "last_fm_api_key",
            "LASTFM_KEY",
        ),
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

mcp = FastMCP("music")


MUSICBRAINZ_BASE = "https://musicbrainz.org/ws/2"
LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"

USER_AGENT = "AIscream-MCP-CRS/1.0 (local academic recommender project)"

RECENT_WORDS = (
    "latest",
    "newest",
    "recent",
    "recently",
    "new",
    "new release",
    "new releases",
    "recently released",
    "newly released",
    "current",
)

SONG_WORDS = (
    "song",
    "songs",
    "track",
    "tracks",
    "single",
    "singles",
)

ALBUM_WORDS = (
    "album",
    "albums",
    "ep",
    "eps",
    "release",
    "releases",
    "discography",
)

NOISY_RECORDING_WORDS = (
    "interview",
    "commentary",
    "podcast",
    "karaoke",
    "tribute",
    "made famous by",
    "originally performed by",
    "work tape",
    "voice memo",
    "acapella",
    "a cappella",
)

REMIX_WORDS = (
    "remix",
    "edit",
    "extended",
    "vip mix",
    "club mix",
    "radio mix",
    "instrumental",
    "sped up",
    "slowed",
)


def debug(*args: Any) -> None:
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


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _title_key(title: Any) -> str:
    text = str(title or "").lower().strip()
    text = re.sub(
        r"[^a-z0-9\u00c0-\u024f\u3040-\u30ff\u4e00-\u9fff가-힣]+",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_exclude_titles(exclude: Any) -> Set[str]:
    if isinstance(exclude, list):
        raw_parts = [str(item) for item in exclude]
    else:
        raw_parts = re.split(r"[,;\n]+", str(exclude or ""))

    output: Set[str] = set()
    for part in raw_parts:
        key = _title_key(part)
        if key:
            output.add(key)

    return output


def _filter_excluded_items(
    items: List[Dict[str, Any]],
    exclude_titles: Set[str],
) -> List[Dict[str, Any]]:
    if not exclude_titles:
        return items

    output: List[Dict[str, Any]] = []

    for item in items:
        key = _title_key(item.get("title"))
        if key and key not in exclude_titles:
            output.append(item)

    return output


async def _get_json(
    url: str,
    params: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 25.0,
) -> Dict[str, Any]:
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    if headers:
        request_headers.update(headers)

    async with httpx.AsyncClient(timeout=timeout, headers=request_headers) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}


async def _musicbrainz_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params)
    params["fmt"] = "json"

    return await _get_json(
        f"{MUSICBRAINZ_BASE}{path}",
        params=params,
        headers={"User-Agent": USER_AGENT},
    )


async def _lastfm_get(params: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("LASTFM_API_KEY", "")

    if not api_key:
        return {}

    full_params = dict(params)
    full_params.update(
        {
            "api_key": api_key,
            "format": "json",
        }
    )

    return await _get_json(LASTFM_BASE, full_params)


def _detect_recent(query: str, recent: bool) -> bool:
    if recent:
        return True

    text = _norm(query)
    return any(word in text for word in RECENT_WORDS)


def _looks_like_latest_request(query: str, recent: bool, sort_by: str) -> bool:
    text = _norm(query)
    sort_text = _norm(sort_by)

    return bool(
        recent
        or sort_text in ("new", "newest", "latest", "recent")
        or any(word in text for word in RECENT_WORDS)
    )


def _wants_songs(
    query: str,
    track_reference: str = "",
    album_reference: str = "",
) -> bool:
    text = _norm(query)

    if track_reference:
        return True

    if album_reference:
        return False

    if any(word in text for word in SONG_WORDS):
        return True

    if any(word in text for word in ALBUM_WORDS):
        return False

    return False


def _wants_albums(query: str, album_reference: str = "") -> bool:
    text = _norm(query)

    return bool(
        album_reference
        or any(word in text for word in ALBUM_WORDS)
    )


def _clean_artist_text(value: str) -> str:
    artist = str(value or "").strip()

    artist = re.sub(
        r"\b("
        r"suggest|recommend|give|show|find|me|some|please|"
        r"latest|newest|recent|new|current|"
        r"songs?|tracks?|music|singles?|albums?|eps?|releases?"
        r")\b",
        " ",
        artist,
        flags=re.IGNORECASE,
    )

    artist = re.sub(r"\s+", " ", artist).strip(" -:,.!?;\"'’")

    return artist


def _extract_artist_from_latest_music_query(query: str) -> str:
    """Extract full artist name from natural music requests.

    Important cases:
    - "latest songs by Sabrina Carpenter" -> "Sabrina Carpenter"
    - "suggest me sabrina carpenter latest songs" -> "sabrina carpenter"
    - "newest BTS songs" -> "BTS"
    - "Katy Perry latest songs" -> "Katy Perry"

    This function intentionally captures the full phrase after "by".
    The previous bug extracted only "Sabrina".
    """
    text = str(query or "").strip()
    text = re.sub(r"\s+", " ", text)

    patterns = (
        # "latest songs by Sabrina Carpenter"
        r"\b(?:latest|newest|recent|new|current)\s+"
        r"(?:songs?|tracks?|music|singles?)\s+by\s+(.+)$",

        # "songs by Sabrina Carpenter"
        r"\b(?:songs?|tracks?|music|singles?)\s+by\s+(.+)$",

        # "latest Sabrina Carpenter songs"
        r"\b(?:latest|newest|recent|new|current)\s+(.+?)\s+"
        r"(?:songs?|tracks?|music|singles?|albums?|eps?|releases?)\b",

        # "Sabrina Carpenter latest songs"
        r"^(.+?)\s+(?:latest|newest|recent|new|current)\s+"
        r"(?:songs?|tracks?|music|singles?|albums?|eps?|releases?)\b",

        # "by Sabrina Carpenter"
        r"\bby\s+(.+)$",
    )

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        artist = _clean_artist_text(match.group(1))

        if artist:
            return artist

    return ""


def _repair_artist_from_by_phrase(query: str, artist_reference: str) -> str:
    """If extractor returned only first name, repair using the full 'by <artist>' phrase."""
    current = str(artist_reference or "").strip()
    text = str(query or "").strip()

    by_match = re.search(r"\bby\s+(.+)$", text, flags=re.IGNORECASE)
    if not by_match:
        return current

    full_artist = _clean_artist_text(by_match.group(1))

    if not full_artist:
        return current

    if not current:
        return full_artist

    if len(full_artist.split()) > len(current.split()):
        return full_artist

    return current


def _clean_query_for_artist_search(query: str, artist_reference: str) -> str:
    if artist_reference:
        return artist_reference.strip()

    inferred = _extract_artist_from_latest_music_query(query)
    return inferred or str(query or "").strip()


def _is_noisy_recording(title: str) -> bool:
    text = _norm(title)

    if not text:
        return True

    return any(word in text for word in NOISY_RECORDING_WORDS)


def _is_remix_or_alt_version(title: str) -> bool:
    text = _norm(title)
    return any(word in text for word in REMIX_WORDS)


def _safe_date(value: Any) -> str:
    text = str(value or "").strip()

    if re.match(r"^\d{4}(-\d{2})?(-\d{2})?$", text):
        return text

    return ""


def _date_sort_key(value: Any) -> str:
    date_text = _safe_date(value)

    if not date_text:
        return "0000-00-00"

    if re.match(r"^\d{4}$", date_text):
        return f"{date_text}-00-00"

    if re.match(r"^\d{4}-\d{2}$", date_text):
        return f"{date_text}-00"

    return date_text


def _artist_score(candidate_name: str, requested_name: str, score: Any = None) -> tuple[int, int, int]:
    candidate = _norm(candidate_name)
    requested = _norm(requested_name)

    try:
        mb_score = int(score or 0)
    except Exception:
        mb_score = 0

    exact = 1 if candidate == requested else 0
    contains = 1 if requested and requested in candidate else 0

    return exact, contains, mb_score


async def _musicbrainz_search_artist(artist_name: str) -> Optional[Dict[str, Any]]:
    artist_name = str(artist_name or "").strip()

    if not artist_name:
        return None

    queries = [
        f'artist:"{artist_name}"',
        artist_name,
    ]

    best_artist: Optional[Dict[str, Any]] = None
    best_score = (-1, -1, -1)

    for query in queries:
        try:
            data = await _musicbrainz_get(
                "/artist",
                {
                    "query": query,
                    "limit": 10,
                    "offset": 0,
                },
            )
        except Exception as exc:
            debug("MusicBrainz artist search failed:", repr(exc))
            continue

        artists = data.get("artists") or []
        if not isinstance(artists, list):
            continue

        for artist in artists:
            if not isinstance(artist, dict):
                continue

            name = artist.get("name") or ""
            candidate_score = _artist_score(
                name,
                artist_name,
                artist.get("score"),
            )

            if candidate_score > best_score:
                best_score = candidate_score
                best_artist = artist

    return best_artist


async def _musicbrainz_release_groups_for_artist(
    artist_id: str,
    *,
    limit: int = 60,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    offsets = [0, 100]

    for offset in offsets:
        try:
            data = await _musicbrainz_get(
                "/release-group",
                {
                    "artist": artist_id,
                    "limit": 100,
                    "offset": offset,
                },
            )
        except Exception as exc:
            debug("MusicBrainz release-group browse failed:", repr(exc))
            break

        raw_groups = data.get("release-groups") or []
        if not isinstance(raw_groups, list):
            continue

        for group in raw_groups:
            if not isinstance(group, dict):
                continue

            group_id = group.get("id")
            if not group_id or group_id in seen_ids:
                continue

            seen_ids.add(group_id)
            groups.append(group)

        if len(groups) >= limit:
            break

    groups.sort(
        key=lambda item: _date_sort_key(item.get("first-release-date")),
        reverse=True,
    )

    return groups[:limit]


async def _musicbrainz_releases_for_group(
    release_group_id: str,
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    try:
        data = await _musicbrainz_get(
            "/release",
            {
                "release-group": release_group_id,
                "inc": "media+recordings+artist-credits",
                "limit": limit,
                "offset": 0,
            },
        )
    except Exception as exc:
        debug("MusicBrainz releases for group failed:", repr(exc))
        return []

    releases = data.get("releases") or []

    if not isinstance(releases, list):
        return []

    releases.sort(
        key=lambda item: _date_sort_key(item.get("date")),
        reverse=True,
    )

    return releases[:limit]


def _recording_artist_text(recording: Dict[str, Any], fallback_artist: str) -> str:
    credits = recording.get("artist-credit") or recording.get("artist-credit-phrase")

    if isinstance(credits, str) and credits.strip():
        return credits.strip()

    if isinstance(credits, list):
        parts: List[str] = []

        for credit in credits:
            if isinstance(credit, dict):
                artist = credit.get("artist") or {}
                if isinstance(artist, dict) and artist.get("name"):
                    parts.append(str(artist.get("name")))
                elif credit.get("name"):
                    parts.append(str(credit.get("name")))
            elif isinstance(credit, str):
                parts.append(credit)

        if parts:
            return ", ".join(parts)

    return fallback_artist


def _release_artist_text(release: Dict[str, Any], fallback_artist: str) -> str:
    credits = release.get("artist-credit") or release.get("artist-credit-phrase")

    if isinstance(credits, str) and credits.strip():
        return credits.strip()

    if isinstance(credits, list):
        parts: List[str] = []

        for credit in credits:
            if isinstance(credit, dict):
                artist = credit.get("artist") or {}
                if isinstance(artist, dict) and artist.get("name"):
                    parts.append(str(artist.get("name")))
                elif credit.get("name"):
                    parts.append(str(credit.get("name")))
            elif isinstance(credit, str):
                parts.append(credit)

        if parts:
            return ", ".join(parts)

    return fallback_artist


def _extract_recordings_from_release(
    release: Dict[str, Any],
    *,
    requested_artist: str,
    release_group_title: str,
    release_group_type: str,
    release_group_id: str,
    exclude_titles: Set[str],
    include_remixes: bool,
) -> List[Dict[str, Any]]:
    release_title = release.get("title") or release_group_title or ""
    release_date = release.get("date") or ""
    release_id = release.get("id") or release_group_id
    release_artist = _release_artist_text(release, requested_artist)

    media = release.get("media") or []
    output: List[Dict[str, Any]] = []

    if not isinstance(media, list):
        return output

    for medium in media:
        if not isinstance(medium, dict):
            continue

        tracks = medium.get("tracks") or []

        if not isinstance(tracks, list):
            continue

        for track in tracks:
            if not isinstance(track, dict):
                continue

            recording = track.get("recording") or {}
            if not isinstance(recording, dict):
                recording = {}

            title = (
                recording.get("title")
                or track.get("title")
                or release_title
                or ""
            )

            title = str(title or "").strip()

            if not title:
                continue

            title_key = _title_key(title)

            if title_key in exclude_titles:
                continue

            if _is_noisy_recording(title):
                continue

            if not include_remixes and _is_remix_or_alt_version(title):
                continue

            recording_id = recording.get("id") or track.get("id") or release_id
            artist_text = _recording_artist_text(recording, release_artist)

            summary_parts = [
                f"Track from release: {release_title}."
            ]

            if release_date:
                summary_parts.append(f"Release date: {release_date}.")

            if release_group_type:
                summary_parts.append(f"Release group type: {release_group_type}.")

            output.append(
                {
                    "domain": "music",
                    "title": title,
                    "subtitle": artist_text or requested_artist,
                    "summary": " ".join(summary_parts).strip(),
                    "source": "MusicBrainz newest artist recordings",
                    "provider": "MusicBrainz",
                    "mcp_server": "music",
                    "external_id": recording_id,
                    "url": (
                        f"https://musicbrainz.org/recording/{recording_id}"
                        if recording_id
                        else f"https://musicbrainz.org/release/{release_id}"
                    ),
                    "_date": release_date,
                }
            )

    return output


async def _musicbrainz_newest_artist_recordings(
    artist_name: str,
    *,
    limit: int,
    exclude_titles: Set[str],
    include_remixes: bool = False,
) -> List[Dict[str, Any]]:
    artist = await _musicbrainz_search_artist(artist_name)

    if not artist:
        return []

    artist_id = artist.get("id")
    resolved_artist_name = artist.get("name") or artist_name

    if not artist_id:
        return []

    groups = await _musicbrainz_release_groups_for_artist(
        artist_id,
        limit=80,
    )

    items: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()

    for group in groups:
        group_id = group.get("id")
        group_title = group.get("title") or ""
        primary_type = group.get("primary-type") or ""
        first_release_date = group.get("first-release-date") or ""

        if not group_id:
            continue

        # For "songs", prioritize Singles and EPs. Albums can contain tracks,
        # but they are lower priority and can flood the result.
        type_text = _norm(primary_type)
        if type_text not in ("single", "ep", "album"):
            continue

        releases = await _musicbrainz_releases_for_group(
            group_id,
            limit=5,
        )

        if not releases:
            # Fallback: for single release groups, the group title is usually the song.
            if type_text == "single":
                title_key = _title_key(group_title)
                if title_key and title_key not in seen_titles and title_key not in exclude_titles:
                    if not _is_noisy_recording(group_title):
                        if include_remixes or not _is_remix_or_alt_version(group_title):
                            items.append(
                                {
                                    "domain": "music",
                                    "title": group_title,
                                    "subtitle": resolved_artist_name,
                                    "summary": (
                                        f"Single release group. "
                                        f"First release date: {first_release_date}."
                                    ).strip(),
                                    "source": "MusicBrainz newest artist recordings",
                                    "provider": "MusicBrainz",
                                    "mcp_server": "music",
                                    "external_id": group_id,
                                    "url": f"https://musicbrainz.org/release-group/{group_id}",
                                    "_date": first_release_date,
                                }
                            )
                            seen_titles.add(title_key)

            if len(items) >= limit:
                break

            continue

        for release in releases:
            release_items = _extract_recordings_from_release(
                release,
                requested_artist=resolved_artist_name,
                release_group_title=group_title,
                release_group_type=primary_type,
                release_group_id=group_id,
                exclude_titles=exclude_titles,
                include_remixes=include_remixes,
            )

            for item in release_items:
                key = _title_key(item.get("title"))

                if not key or key in seen_titles:
                    continue

                seen_titles.add(key)
                items.append(item)

                if len(items) >= limit:
                    break

            if len(items) >= limit:
                break

        if len(items) >= limit:
            break

    items.sort(
        key=lambda item: _date_sort_key(item.get("_date")),
        reverse=True,
    )

    for item in items:
        item.pop("_date", None)

    return items[:limit]


async def _musicbrainz_newest_artist_releases(
    artist_name: str,
    *,
    limit: int,
    exclude_titles: Set[str],
) -> List[Dict[str, Any]]:
    artist = await _musicbrainz_search_artist(artist_name)

    if not artist:
        return []

    artist_id = artist.get("id")
    resolved_artist_name = artist.get("name") or artist_name

    if not artist_id:
        return []

    groups = await _musicbrainz_release_groups_for_artist(
        artist_id,
        limit=80,
    )

    output: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()

    for group in groups:
        group_id = group.get("id")
        title = group.get("title") or ""
        first_release_date = group.get("first-release-date") or ""
        primary_type = group.get("primary-type") or ""

        key = _title_key(title)

        if not key or key in seen_titles or key in exclude_titles:
            continue

        seen_titles.add(key)

        output.append(
            {
                "domain": "music",
                "title": title,
                "subtitle": resolved_artist_name,
                "summary": (
                    f"Release group type: {primary_type or 'unknown'}. "
                    f"First release date: {first_release_date or 'unknown'}."
                ),
                "source": "MusicBrainz newest artist releases",
                "provider": "MusicBrainz",
                "mcp_server": "music",
                "external_id": group_id,
                "url": f"https://musicbrainz.org/release-group/{group_id}",
                "_date": first_release_date,
            }
        )

        if len(output) >= limit:
            break

    output.sort(
        key=lambda item: _date_sort_key(item.get("_date")),
        reverse=True,
    )

    for item in output:
        item.pop("_date", None)

    return output[:limit]


async def _lastfm_artist_top_tracks(
    artist_name: str,
    *,
    limit: int,
    exclude_titles: Set[str],
) -> List[Dict[str, Any]]:
    api_key = os.getenv("LASTFM_API_KEY", "")

    if not api_key or not artist_name:
        return []

    try:
        data = await _lastfm_get(
            {
                "method": "artist.gettoptracks",
                "artist": artist_name,
                "limit": limit,
                "autocorrect": 1,
            }
        )
    except Exception as exc:
        debug("Last.fm artist.gettoptracks failed:", repr(exc))
        return []

    tracks = (
        data.get("toptracks", {}).get("track", [])
        if isinstance(data, dict)
        else []
    )

    if isinstance(tracks, dict):
        tracks = [tracks]

    output: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()

    for track in tracks:
        if not isinstance(track, dict):
            continue

        title = track.get("name") or ""
        key = _title_key(title)

        if not key or key in seen_titles or key in exclude_titles:
            continue

        seen_titles.add(key)

        artist = track.get("artist") or {}
        artist_text = artist.get("name") if isinstance(artist, dict) else artist_name

        listeners = track.get("listeners")
        playcount = track.get("playcount")
        url = track.get("url") or ""

        summary_parts = []
        if listeners:
            summary_parts.append(f"Listeners: {listeners}.")
        if playcount:
            summary_parts.append(f"Playcount: {playcount}.")

        output.append(
            {
                "domain": "music",
                "title": title,
                "subtitle": artist_text or artist_name,
                "summary": " ".join(summary_parts).strip(),
                "source": "Last.fm artist top tracks",
                "provider": "Last.fm",
                "mcp_server": "music",
                "external_id": "",
                "url": url,
            }
        )

        if len(output) >= limit:
            break

    return output


async def _lastfm_track_search(
    query: str,
    *,
    limit: int,
    exclude_titles: Set[str],
) -> List[Dict[str, Any]]:
    api_key = os.getenv("LASTFM_API_KEY", "")

    if not api_key or not query:
        return []

    try:
        data = await _lastfm_get(
            {
                "method": "track.search",
                "track": query,
                "limit": limit,
            }
        )
    except Exception as exc:
        debug("Last.fm track.search failed:", repr(exc))
        return []

    matches = (
        data.get("results", {})
        .get("trackmatches", {})
        .get("track", [])
        if isinstance(data, dict)
        else []
    )

    if isinstance(matches, dict):
        matches = [matches]

    output: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()

    for track in matches:
        if not isinstance(track, dict):
            continue

        title = track.get("name") or ""
        key = _title_key(title)

        if not key or key in seen_titles or key in exclude_titles:
            continue

        if _is_noisy_recording(title):
            continue

        seen_titles.add(key)

        artist = track.get("artist") or ""
        listeners = track.get("listeners")
        url = track.get("url") or ""

        summary = f"Last.fm text search match. Listeners: {listeners}." if listeners else "Last.fm text search match."

        output.append(
            {
                "domain": "music",
                "title": title,
                "subtitle": artist,
                "summary": summary,
                "source": "Last.fm track.search",
                "provider": "Last.fm",
                "mcp_server": "music",
                "external_id": "",
                "url": url,
            }
        )

        if len(output) >= limit:
            break

    return output


async def _lastfm_tag_top_tracks(
    tag: str,
    *,
    limit: int,
    exclude_titles: Set[str],
) -> List[Dict[str, Any]]:
    api_key = os.getenv("LASTFM_API_KEY", "")

    if not api_key or not tag:
        return []

    try:
        data = await _lastfm_get(
            {
                "method": "tag.gettoptracks",
                "tag": tag,
                "limit": limit,
            }
        )
    except Exception as exc:
        debug("Last.fm tag.gettoptracks failed:", repr(exc))
        return []

    tracks = (
        data.get("tracks", {}).get("track", [])
        if isinstance(data, dict)
        else []
    )

    if isinstance(tracks, dict):
        tracks = [tracks]

    output: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()

    for track in tracks:
        if not isinstance(track, dict):
            continue

        title = track.get("name") or ""
        key = _title_key(title)

        if not key or key in seen_titles or key in exclude_titles:
            continue

        seen_titles.add(key)

        artist = track.get("artist") or {}
        artist_text = artist.get("name") if isinstance(artist, dict) else ""

        output.append(
            {
                "domain": "music",
                "title": title,
                "subtitle": artist_text,
                "summary": f"Top Last.fm track for tag: {tag}.",
                "source": "Last.fm tag top tracks",
                "provider": "Last.fm",
                "mcp_server": "music",
                "external_id": "",
                "url": track.get("url") or "",
            }
        )

        if len(output) >= limit:
            break

    return output


def _interleave_unique_items(
    groups: List[List[Dict[str, Any]]],
    limit: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()

    max_len = max((len(group) for group in groups), default=0)

    for index in range(max_len):
        for group in groups:
            if index >= len(group):
                continue

            item = group[index]
            key = _title_key(item.get("title"))

            if not key or key in seen_titles:
                continue

            seen_titles.add(key)
            output.append(item)

            if len(output) >= limit:
                return output

    return output[:limit]


def _query_tags(
    genre: str = "",
    mood: str = "",
    theme: str = "",
    music_tag: str = "",
) -> List[str]:
    tags: List[str] = []

    for value in (music_tag, genre, mood, theme):
        text = str(value or "").strip()
        if text:
            tags.append(text)

    deduped: List[str] = []
    seen: Set[str] = set()

    for tag in tags:
        key = _norm(tag)
        if key and key not in seen:
            seen.add(key)
            deduped.append(tag)

    return deduped


@mcp.tool(
    description=(
        "Recommend grounded music candidates. "
        "For latest artist songs, this uses MusicBrainz artist search, release groups, "
        "releases, and recordings so final items are tracks/recordings, not album titles."
    )
)
async def recommend_music(
    query: str,
    limit: int = 5,
    recent: bool = False,
    genre: str = "",
    mood: str = "",
    theme: str = "",
    year_min: int = 0,
    year_max: int = 0,
    country: str = "",
    language: str = "",
    artist_reference: str = "",
    album_reference: str = "",
    track_reference: str = "",
    music_tag: str = "",
    reference_title: str = "",
    reference_creator: str = "",
    transfer_terms: str = "",
    sort_by: str = "",
    exclude: str = "",
    include_adult: bool = False,
) -> Dict[str, Any]:
    safe_limit = max(1, min(int(limit or 5), 40))

    query = str(query or "").strip()
    genre = str(genre or "").strip()
    mood = str(mood or "").strip()
    theme = str(theme or "").strip()
    country = str(country or "").strip()
    language = str(language or "").strip()
    artist_reference = str(artist_reference or "").strip()
    album_reference = str(album_reference or "").strip()
    track_reference = str(track_reference or "").strip()
    music_tag = str(music_tag or "").strip()
    reference_title = str(reference_title or "").strip()
    reference_creator = str(reference_creator or "").strip()
    transfer_terms = str(transfer_terms or "").strip()
    sort_by = str(sort_by or "").strip().lower()
    exclude = str(exclude or "").strip()

    detected_recent = _detect_recent(query, recent)

    latest_request = _looks_like_latest_request(
        query=query,
        recent=detected_recent,
        sort_by=sort_by,
    )

    if latest_request:
        detected_recent = True
        sort_by = "newest"
    elif not sort_by:
        sort_by = "relevance"

    if latest_request and not artist_reference:
        artist_reference = _extract_artist_from_latest_music_query(query)

    artist_reference = _repair_artist_from_by_phrase(query, artist_reference)

    # Another repair for LLM behavior:
    # If query says "latest songs by Sabrina Carpenter" but LLM only provided query,
    # make the query become the artist name for retrieval.
    if not artist_reference and any(word in _norm(query) for word in SONG_WORDS + ALBUM_WORDS):
        artist_reference = _extract_artist_from_latest_music_query(query)

    if artist_reference:
        query_for_search = artist_reference
    else:
        query_parts = [
            query,
            reference_title,
            reference_creator,
            transfer_terms,
            genre,
            mood,
            theme,
            music_tag,
        ]
        query_for_search = " ".join(part for part in query_parts if part).strip()

    wants_songs = _wants_songs(
        query=query,
        track_reference=track_reference,
        album_reference=album_reference,
    )
    wants_albums = _wants_albums(
        query=query,
        album_reference=album_reference,
    )

    if artist_reference and latest_request and not wants_albums:
        wants_songs = True

    include_remixes = any(
        word in _norm(query)
        for word in (
            "remix",
            "remixes",
            "version",
            "versions",
            "edit",
            "edits",
        )
    )

    exclude_titles = _split_exclude_titles(exclude)

    debug("\n========== MUSIC MCP TOOL ==========")
    debug("query:", query)
    debug("query_for_search:", query_for_search)
    debug("limit:", safe_limit)
    debug("recent:", detected_recent)
    debug("latest_request:", latest_request)
    debug("wants_songs:", wants_songs)
    debug("wants_albums:", wants_albums)
    debug("genre:", genre)
    debug("mood:", mood)
    debug("theme:", theme)
    debug("country:", country)
    debug("language:", language)
    debug("artist_reference after extraction:", artist_reference)
    debug("album_reference:", album_reference)
    debug("track_reference:", track_reference)
    debug("music_tag:", music_tag)
    debug("reference_title:", reference_title)
    debug("reference_creator:", reference_creator)
    debug("transfer_terms:", transfer_terms)
    debug("sort_by:", sort_by)
    debug("exclude:", exclude)
    debug("exclude_titles:", sorted(exclude_titles))
    debug("LASTFM key loaded:", bool(os.getenv("LASTFM_API_KEY", "")))
    debug("====================================")

    groups: List[List[Dict[str, Any]]] = []

    # 1. Correct path for "latest songs by <artist>":
    #    artist -> MusicBrainz release groups -> releases -> recordings/tracks.
    if artist_reference and latest_request and wants_songs:
        recordings = await _musicbrainz_newest_artist_recordings(
            artist_reference,
            limit=safe_limit,
            exclude_titles=exclude_titles,
            include_remixes=include_remixes,
        )
        if recordings:
            groups.append(recordings)

    # 2. Correct path for latest albums/EPs/releases.
    if artist_reference and latest_request and wants_albums and not wants_songs:
        releases = await _musicbrainz_newest_artist_releases(
            artist_reference,
            limit=safe_limit,
            exclude_titles=exclude_titles,
        )
        if releases:
            groups.append(releases)

    # 3. Normal artist request: Last.fm top tracks.
    if artist_reference and not latest_request:
        artist_tracks = await _lastfm_artist_top_tracks(
            artist_reference,
            limit=safe_limit,
            exclude_titles=exclude_titles,
        )
        if artist_tracks:
            groups.append(artist_tracks)

    # 4. If latest artist request found no MusicBrainz recordings, fall back to releases.
    if artist_reference and latest_request and not groups:
        releases = await _musicbrainz_newest_artist_releases(
            artist_reference,
            limit=safe_limit,
            exclude_titles=exclude_titles,
        )
        if releases:
            groups.append(releases)

    # 5. Tag/mood/genre search. Useful for "sad pop songs", "study music", etc.
    tags = _query_tags(
        genre=genre,
        mood=mood,
        theme=theme,
        music_tag=music_tag,
    )

    if tags and not latest_request:
        for tag in tags[:3]:
            tag_items = await _lastfm_tag_top_tracks(
                tag,
                limit=safe_limit,
                exclude_titles=exclude_titles,
            )
            if tag_items:
                groups.append(tag_items)

    # 6. Track/text search fallback.
    # Avoid literal search for "latest songs by Sabrina Carpenter" because Last.fm
    # text search is relevance/popularity, not newest-by-date.
    should_use_track_search = bool(query_for_search) and not (
        latest_request and artist_reference
    )

    if should_use_track_search:
        searched = await _lastfm_track_search(
            query_for_search,
            limit=safe_limit,
            exclude_titles=exclude_titles,
        )
        if searched:
            groups.append(searched)

    items = _interleave_unique_items(groups, safe_limit)
    items = _filter_excluded_items(items, exclude_titles)[:safe_limit]

    detected_payload = {
        "recent": detected_recent,
        "genre": genre,
        "mood": mood,
        "theme": theme,
        "country": country,
        "language": language,
        "artist_reference": artist_reference,
        "album_reference": album_reference,
        "track_reference": track_reference,
        "reference_title": reference_title,
        "reference_creator": reference_creator,
        "transfer_terms": transfer_terms,
        "tags": tags,
        "sort_by": sort_by or ("newest" if detected_recent else "relevance"),
        "wants_songs": wants_songs,
        "wants_albums": wants_albums,
        "exclude_count": len(exclude_titles),
    }

    if items:
        output = {
            "domain": "music",
            "items": items,
            "source": "Last.fm/MusicBrainz mixed retrieval",
            "provider": "Last.fm/MusicBrainz",
            "detected": detected_payload,
        }

        debug_json(
            "MUSIC TOOL OUTPUT",
            {
                "source": output.get("source"),
                "provider": output.get("provider"),
                "count": len(items),
                "detected": detected_payload,
                "items": [
                    {
                        "title": item.get("title"),
                        "subtitle": item.get("subtitle"),
                        "source": item.get("source"),
                        "url": item.get("url"),
                    }
                    for item in items[:10]
                ],
            },
        )

        return output

    error_messages: List[str] = []

    if not os.getenv("LASTFM_API_KEY", ""):
        error_messages.append("LASTFM_API_KEY is missing or not loaded.")

    if artist_reference and latest_request:
        error_messages.append(
            "MusicBrainz returned no usable newest artist recordings/releases."
        )

    if not error_messages:
        error_messages.append("No usable music candidates found.")

    output = {
        "domain": "music",
        "items": [],
        "source": "Last.fm/MusicBrainz no results",
        "provider": "Last.fm/MusicBrainz",
        "detected": detected_payload,
        "error": " ".join(error_messages),
    }

    debug_json("MUSIC TOOL OUTPUT", output)
    return output


if __name__ == "__main__":
    mcp.run()
