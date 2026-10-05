import ast
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    except Exception as exc:
        print(f"[ENV] Could not read {path}: {exc}")

    return values


def _with_project_env(base_env: dict[str, str]) -> dict[str, str]:
    """Pass API keys from both root .env and backend/.env to MCP child processes.

    The backend settings file reads backend/.env, while MCP servers historically
    read only the project-root .env. Loading both here prevents the servers from
    reporting missing TMDB/OMDb/Last.fm keys when the user placed keys in
    backend/.env. It also normalizes common lowercase/alternate key names.
    """
    env = dict(base_env)

    for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"):
        for key, value in _load_env_file_values(env_path).items():
            env.setdefault(key, value)
            env.setdefault(key.upper(), value)

    aliases = {
        "TMDB_API_KEY": ("tmdb_api_key", "TMDB_KEY"),
        "OMDB_API_KEY": ("omdb_api_key", "OMDB_KEY"),
        "LASTFM_API_KEY": ("lastfm_api_key", "LAST_FM_API_KEY", "last_fm_api_key"),
        "GOOGLE_BOOKS_API_KEY": ("google_books_api_key", "GOOGLE_BOOKS_KEY"),
        "OPENAI_API_KEY": ("openai_api_key",),
    }

    for canonical, names in aliases.items():
        current = env.get(canonical, "")
        if not current:
            for name in names:
                if env.get(name):
                    env[canonical] = env[name]
                    current = env[name]
                    break
        if current:
            for name in names:
                env.setdefault(name, current)

    return env


@dataclass
class MCPConnection:
    domain: str
    server_path: Path
    session: ClientSession | None = None
    read_stream: Any = None
    write_stream: Any = None
    client_context: Any = None
    session_context: Any = None
    enabled: bool = True
    reason: str = ""


class MCPRegistry:
    def __init__(self) -> None:
        self.server_specs: dict[str, Path] = {
            "movies": PROJECT_ROOT / "mcp_servers" / "movies" / "server.py",
            "music": PROJECT_ROOT / "mcp_servers" / "music" / "server.py",
            "books": PROJECT_ROOT / "mcp_servers" / "books" / "server.py",
        }

        self.connections: dict[str, MCPConnection] = {}
        self.tool_cache: list[dict[str, Any]] = []
        self.tool_to_domain: dict[str, str] = {}

    async def warmup(self) -> None:
        print("\n========== MCP WARMUP ==========")
        print("Project root:", PROJECT_ROOT)

        for domain, path in self.server_specs.items():
            print(f"{domain}: {path}")

        print("================================\n")

        try:
            await self.list_openai_tools()
        except Exception as exc:
            print(f"[MCP] Warmup failed: {type(exc).__name__}: {exc}")

    def domain_status(self) -> dict[str, str]:
        status: dict[str, str] = {}

        for domain in self.server_specs:
            conn = self.connections.get(domain)

            if not conn:
                status[domain] = "not_connected"
            elif conn.enabled:
                status[domain] = "connected"
            else:
                status[domain] = f"disabled: {conn.reason}"

        return status

    async def ensure_connected(self, domain: str) -> MCPConnection:
        if domain not in self.server_specs:
            raise ValueError(f"Unknown MCP domain: {domain}")

        existing = self.connections.get(domain)

        if existing and existing.enabled and existing.session is not None:
            return existing

        if existing:
            await self._close_connection(domain)

        server_path = self.server_specs[domain]

        if not server_path.exists():
            conn = MCPConnection(
                domain=domain,
                server_path=server_path,
                enabled=False,
                reason=f"Server file not found: {server_path}",
            )
            self.connections[domain] = conn
            raise FileNotFoundError(conn.reason)

        env = _with_project_env(os.environ.copy())
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        print(f"\n========== CONNECTING MCP SERVER: {domain} ==========")
        print("Server path:", server_path)
        print("Working dir:", PROJECT_ROOT)
        print("PYTHONPATH:", env.get("PYTHONPATH"))
        print("====================================================\n")

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        conn = MCPConnection(
            domain=domain,
            server_path=server_path,
        )

        try:
            client_context = stdio_client(params)
            read_stream, write_stream = await client_context.__aenter__()

            session_context = ClientSession(read_stream, write_stream)
            session = await session_context.__aenter__()

            await session.initialize()

            conn.client_context = client_context
            conn.session_context = session_context
            conn.read_stream = read_stream
            conn.write_stream = write_stream
            conn.session = session
            conn.enabled = True
            conn.reason = ""

            self.connections[domain] = conn

            print(f"[MCP] Connected: {domain}")
            return conn

        except Exception as exc:
            conn.enabled = False
            conn.reason = f"Failed to connect: {type(exc).__name__}: {exc}"
            self.connections[domain] = conn

            print(f"[MCP] Failed to connect {domain}: {conn.reason}")
            raise

    async def _close_connection(self, domain: str) -> None:
        conn = self.connections.pop(domain, None)

        if not conn:
            return

        try:
            if conn.session_context:
                await conn.session_context.__aexit__(None, None, None)
        except Exception:
            pass

        try:
            if conn.client_context:
                await conn.client_context.__aexit__(None, None, None)
        except Exception:
            pass

        conn.session = None
        conn.read_stream = None
        conn.write_stream = None
        conn.session_context = None
        conn.client_context = None
        conn.enabled = False

    async def close_all(self) -> None:
        domains = list(self.connections.keys())

        for domain in domains:
            await self._close_connection(domain)

        self.connections.clear()
        self.tool_cache = []
        self.tool_to_domain = {}

    async def list_openai_tools(self) -> list[dict[str, Any]]:
        if self.tool_cache:
            return self.tool_cache

        print("\n========== DISCOVERING MCP TOOLS ==========")

        tools: list[dict[str, Any]] = []
        self.tool_to_domain = {}

        for domain in self.server_specs:
            try:
                if domain in self.connections:
                    existing = self.connections[domain]
                    if getattr(existing, "enabled", True) is False:
                        self.connections.pop(domain, None)

                conn = await self.ensure_connected(domain)

                if conn.session is None:
                    raise RuntimeError(f"MCP session for domain {domain} is not available.")

                result = await conn.session.list_tools()

                for tool in result.tools:
                    print(f"[MCP TOOL] {tool.name} from {domain}")

                    self.tool_to_domain[tool.name] = domain

                    tools.append(
                        {
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        }
                    )

            except Exception as exc:
                print(f"[MCP] Skipping {domain}: {type(exc).__name__}: {exc}")

                if domain in self.connections:
                    self.connections[domain].enabled = False
                    self.connections[domain].reason = (
                        f"Discovery failed: {type(exc).__name__}: {exc}"
                    )

        self.tool_cache = tools

        print("Tool map:", self.tool_to_domain)
        print("===========================================\n")

        return tools

    def _decode_tool_result(self, result: Any, domain: str) -> dict[str, Any]:
        """Decode MCP CallToolResult into the plain dict returned by our servers.

        FastMCP/Pydantic versions differ here:
        - some expose the return value as result.structuredContent
        - some expose result.structured_content
        - some only put JSON inside TextContent.text

        The previous code returned the first truthy structured payload. In some
        environments that payload is a wrapper/empty dict, while the real items are
        still present in content[0].text. That produced logs like RAW MCP TOOL RESULT
        showing items, but DECODED MCP RESULT item_count: 0.
        """

        def parse_text_payload(text: str) -> dict[str, Any] | None:
            if not text:
                return None

            raw = text.strip()

            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            return None

        def unwrap_payload(payload: Any) -> dict[str, Any] | None:
            if payload is None:
                return None

            if isinstance(payload, str):
                return parse_text_payload(payload)

            if not isinstance(payload, dict):
                return None

            # Some MCP versions wrap the tool return value under a generic key.
            for key in ("result", "data", "payload", "structuredContent", "structured_content"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    return unwrap_payload(nested) or nested
                if isinstance(nested, str):
                    parsed = parse_text_payload(nested)
                    if parsed is not None:
                        return parsed

            return payload

        candidates: list[dict[str, Any]] = []

        for attr in ("structuredContent", "structured_content"):
            payload = unwrap_payload(getattr(result, attr, None))
            if payload is not None:
                candidates.append(payload)

        content = getattr(result, "content", None) or []
        first_unparseable_text = ""

        for item in content:
            text = getattr(item, "text", None)
            if not text:
                continue

            parsed = parse_text_payload(text)
            if parsed is not None:
                candidates.append(parsed)
            elif not first_unparseable_text:
                first_unparseable_text = str(text)

        # Prefer the candidate that actually contains items. This fixes the
        # "RAW result has items, decoded item_count is 0" problem.
        for candidate in candidates:
            if isinstance(candidate.get("items"), list) and candidate.get("items"):
                return candidate

        # Next prefer a normal MCP response shape, even if items is empty, because
        # it may contain useful provider/detected/error fields.
        for candidate in candidates:
            if "items" in candidate or "domain" in candidate or "provider" in candidate or "source" in candidate:
                if "items" not in candidate:
                    candidate = dict(candidate)
                    candidate["items"] = []
                return candidate

        if first_unparseable_text:
            return {
                "items": [],
                "source": "MCP text response",
                "provider": domain,
                "message": first_unparseable_text,
                "error": "Tool returned text but not parseable JSON/dict.",
            }

        return {
            "items": [],
            "source": "MCP empty response",
            "provider": domain,
            "error": "Tool returned no readable structured or text content.",
        }

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.tool_to_domain:
            await self.list_openai_tools()

        domain = self.tool_to_domain.get(name)

        if domain is None:
            self.tool_cache = []
            self.tool_to_domain = {}
            await self.list_openai_tools()
            domain = self.tool_to_domain.get(name)

        if domain is None:
            return {
                "items": [],
                "source": None,
                "provider": None,
                "error": (
                    f"Unknown or unavailable tool: {name}. "
                    f"Available tools: {list(self.tool_to_domain.keys())}. "
                    f"Domain status: {self.domain_status()}"
                ),
            }

        try:
            if domain in self.connections:
                existing = self.connections[domain]

                if getattr(existing, "enabled", True) is False:
                    self.connections.pop(domain, None)

            conn = await self.ensure_connected(domain)

            if conn.session is None:
                raise RuntimeError(f"MCP session for domain {domain} is not available.")

            print("\n========== CALLING MCP TOOL ==========", flush=True)
            print("Tool:", name, flush=True)
            print("Domain:", domain, flush=True)
            print("Arguments:", json.dumps(arguments, ensure_ascii=False, indent=2), flush=True)
            print("======================================\n", flush=True)

            result = await conn.session.call_tool(name, arguments)

            print("\n========== RAW MCP TOOL RESULT ==========", flush=True)
            print("result type:", type(result), flush=True)
            print("has structuredContent:", hasattr(result, "structuredContent"), flush=True)
            print("has structured_content:", hasattr(result, "structured_content"), flush=True)
            print("content:", getattr(result, "content", None), flush=True)
            print("=========================================\n", flush=True)

            decoded = self._decode_tool_result(result, domain)

            print("\n========== DECODED MCP RESULT ==========" , flush=True)
            print("source:", decoded.get("source"), flush=True)
            print("provider:", decoded.get("provider"), flush=True)
            items_for_debug = decoded.get("items") or []
            print("item_count:", len(items_for_debug), flush=True)
            for i, item in enumerate(items_for_debug[:10], start=1):
                print(f"{i}. {item.get('title')} | {item.get('subtitle')} | source={item.get('source')} | url={item.get('url')}", flush=True)
            if decoded.get("detected"):
                print("detected:", json.dumps(decoded.get("detected"), ensure_ascii=False, indent=2), flush=True)
            if decoded.get("error"):
                print("error:", decoded.get("error"), flush=True)
            print("========================================\n", flush=True)

            if "items" not in decoded:
                decoded["items"] = []

            if "provider" not in decoded or decoded.get("provider") is None:
                decoded["provider"] = domain

            return decoded

        except Exception as exc:
            try:
                conn = self.connections.pop(domain, None)

                if conn:
                    try:
                        conn.enabled = False
                        conn.reason = f"Tool call failed: {type(exc).__name__}: {exc}"
                    except Exception:
                        pass

                    try:
                        if conn.session_context:
                            await conn.session_context.__aexit__(None, None, None)
                    except Exception:
                        pass

                    try:
                        if conn.client_context:
                            await conn.client_context.__aexit__(None, None, None)
                    except Exception:
                        pass

            except Exception:
                pass

            return {
                "items": [],
                "source": None,
                "provider": None,
                "error": (
                    f"Tool call failed for {name} on domain {domain}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
