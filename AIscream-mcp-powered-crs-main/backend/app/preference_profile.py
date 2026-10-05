from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Domain = Literal['movies', 'music', 'books']

DOMAIN_KEYWORDS: dict[Domain, set[str]] = {
    'movies': {
        'movie', 'movies', 'film', 'films', 'cinema', 'director', 'actor', 'actress',
        'nolan', 'interstellar', 'arrival', 'blade runner', 'sci-fi movie'
    },
    'music': {
        'music', 'song', 'songs', 'album', 'albums', 'track', 'tracks', 'artist',
        'spotify', 'band', 'singer', 'playlist', 'soundtrack'
    },
    'books': {
        'book', 'books', 'novel', 'novels', 'author', 'read', 'reading', 'literature',
        'open library', 'google books'
    },
}

SEMANTIC_KEYWORDS = {
    'emotional', 'philosophical', 'dark', 'funny', 'romantic', 'sad', 'uplifting',
    'slow-burn', 'slow burn', 'fast-paced', 'fast paced', 'cyberpunk', 'sci-fi',
    'science fiction', 'fantasy', 'mystery', 'thriller', 'horror', 'cozy', 'epic',
    'atmospheric', 'melancholic', 'adventurous', 'nostalgic', 'dreamy', 'introspective',
    'complex', 'nonlinear', 'non-linear', 'space', 'detective', 'crime', 'coming-of-age',
}

LIKE_PATTERNS = [
    re.compile(r"\bi like\s+([^.!?]+)", re.IGNORECASE),
    re.compile(r"\bi love\s+([^.!?]+)", re.IGNORECASE),
    re.compile(r"\bi enjoy\s+([^.!?]+)", re.IGNORECASE),
    re.compile(r"\bmy favorite(?:s)? (?:is|are)\s+([^.!?]+)", re.IGNORECASE),
]

REQUEST_PATTERNS = {
    'movies': re.compile(r"\b(recommend|suggest|find|give me).{0,40}\b(movies?|films?)\b|\b(movies?|films?).{0,40}\b(recommend|suggest|find)\b", re.IGNORECASE),
    'music': re.compile(r"\b(recommend|suggest|find|give me).{0,40}\b(music|songs?|albums?|tracks?|playlist)\b|\b(music|songs?|albums?|tracks?|playlist).{0,40}\b(recommend|suggest|find)\b", re.IGNORECASE),
    'books': re.compile(r"\b(recommend|suggest|find|give me).{0,40}\b(books?|novels?)\b|\b(books?|novels?).{0,40}\b(recommend|suggest|find)\b", re.IGNORECASE),
}


@dataclass
class PreferenceProfile:
    movie_preferences: list[str] = field(default_factory=list)
    music_preferences: list[str] = field(default_factory=list)
    book_preferences: list[str] = field(default_factory=list)
    shared_semantic_preferences: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            'movie_preferences': self.movie_preferences,
            'music_preferences': self.music_preferences,
            'book_preferences': self.book_preferences,
            'shared_semantic_preferences': self.shared_semantic_preferences,
        }


def _append_unique(items: list[str], value: str) -> None:
    value = value.strip(' ,;:.!?\n\t')
    if value and value.lower() not in {item.lower() for item in items}:
        items.append(value)


def detect_target_domains(text: str) -> list[Domain]:
    found: list[Domain] = []
    for domain, pattern in REQUEST_PATTERNS.items():
        if pattern.search(text):
            found.append(domain)  # type: ignore[arg-type]
    if found:
        return found

    lower = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            found.append(domain)
    return found or ['movies', 'music', 'books']


def build_preference_profile(history: list[dict[str, str]]) -> PreferenceProfile:
    profile = PreferenceProfile()

    for message in history:
        if message.get('role') != 'user':
            continue

        text = message.get('content', '')
        lower = text.lower()
        mentioned_domains = [
            domain for domain, keywords in DOMAIN_KEYWORDS.items()
            if any(keyword in lower for keyword in keywords)
        ]

        for keyword in SEMANTIC_KEYWORDS:
            if keyword in lower:
                _append_unique(profile.shared_semantic_preferences, keyword)

        extracted: list[str] = []
        for pattern in LIKE_PATTERNS:
            for match in pattern.findall(text):
                extracted.append(match)

        if not extracted:
            extracted = [text]

        for value in extracted:
            if not mentioned_domains:
                _append_unique(profile.shared_semantic_preferences, value)
            for domain in mentioned_domains:
                if domain == 'movies':
                    _append_unique(profile.movie_preferences, value)
                elif domain == 'music':
                    _append_unique(profile.music_preferences, value)
                elif domain == 'books':
                    _append_unique(profile.book_preferences, value)

    return profile


def preference_context(
    profile: PreferenceProfile,
    target_domains: list[Domain],
    *,
    cross_domain: bool,
) -> str:
    target = ', '.join(target_domains)

    if cross_domain:
        return (
            'User preference profile for this conversation:\n'
            f'{profile.as_dict()}\n\n'
            'Cross-domain transfer is ENABLED. You may use movie, music, book, and shared semantic '
            f'preferences to build retrieval queries for the requested target domain(s): {target}. '
            'Translate preferences semantically across domains. Example: emotional sci-fi movies can become '
            'philosophical science-fiction books or atmospheric space-themed music.'
        )

    allowed: dict[str, list[str]] = {}
    if 'movies' in target_domains:
        allowed['movie_preferences'] = profile.movie_preferences
    if 'music' in target_domains:
        allowed['music_preferences'] = profile.music_preferences
    if 'books' in target_domains:
        allowed['book_preferences'] = profile.book_preferences

    return (
        'User preference profile for this conversation:\n'
        f'{profile.as_dict()}\n\n'
        'Cross-domain transfer is DISABLED for this baseline. Use ONLY preferences that were explicitly '
        f'expressed in the requested target domain(s): {target}. Allowed target-domain preferences: {allowed}. '
        'Ignore preferences from other domains even if they seem semantically useful. If no target-domain '
        'preferences exist, retrieve generic/popular candidates for the target domain.'
    )
