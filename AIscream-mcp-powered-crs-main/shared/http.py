from __future__ import annotations

import asyncio
from typing import Any

import httpx


async def fetch_json(url: str, *, params: dict[str, Any] | None = None, timeout: float = 12.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def safe_fetch_json(url: str, *, params: dict[str, Any] | None = None, timeout: float = 12.0) -> dict[str, Any] | None:
    try:
        return await fetch_json(url, params=params, timeout=timeout)
    except (httpx.HTTPError, asyncio.TimeoutError):
        return None
