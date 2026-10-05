from __future__ import annotations

import json
from typing import Any

from .config import get_settings


class OpenAIService:
    """LLM service.

    Provider selection:
    - If OPENAI_API_KEY is set, use OpenAI Responses API.
    - If OPENAI_API_KEY is empty, use a local Ollama OpenAI-compatible endpoint.

    The class name is kept as OpenAIService to avoid changing the rest of the app,
    but it now supports both OpenAI and Ollama.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.provider = 'openai' if bool(settings.openai_api_key) else 'ollama'
        self.model = settings.openai_model if self.provider == 'openai' else settings.ollama_model
        self.enabled = True
        self.client = None

        try:
            from openai import AsyncOpenAI

            if self.provider == 'openai':
                self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            else:
                # Ollama exposes an OpenAI-compatible API at /v1.
                self.client = AsyncOpenAI(
                    api_key='ollama',
                    base_url=settings.ollama_base_url.rstrip('/'),
                )
        except ModuleNotFoundError:
            self.enabled = False
            self.client = None

    def _base_formatting_rules(self) -> str:
        return (
            'When presenting recommendations, use markdown: use **bold** for titles, bullet points (- ) '
            'for lists, and blank lines between sections. Use actual line breaks, not literal \\n characters. '
            'Each recommendation must be exactly one bullet point. Do not use numbered lists. '
            'For movies, use: - **Movie Title (Year)** — one short description sentence. _(Source: source/server name)_ '
            'For music, use: - **Song Title** (Artist: Artist Name) — one short description sentence. _(Source: source/server name)_ '
            'For books, use: - **Book Title** (Author: Author Name) — one short description sentence. _(Source: source/server name)_ '
            'Only include movie year, artist, or author when available. If missing, omit that part instead of inventing it. '
            'For each requested domain or category, provide 5 recommendations when enough candidates are available. '
        )

    def _system_instructions(
        self,
        *,
        use_mcp: bool,
        use_history: bool,
        cross_domain: bool,
        grounded_only: bool,
        mode_name: str,
    ) -> str:
        model_role = 'GPT-4o' if self.provider == 'openai' else 'The local Ollama model'
        instructions = (
            'You are an entertainment recommendation assistant specializing in movies, music, and books. '
            f'{model_role} handles conversation, preference understanding, semantic reasoning, ranking, and explanation. '
            'MCP servers handle external candidate retrieval from movie, music, and book APIs when tools are enabled. '
            f'Current LLM provider: {self.provider}. Current experiment mode: {mode_name}. '
        )

        if use_mcp:
            instructions += (
                'MCP/API tools are ENABLED. Use tools whenever the user asks for recommendations or discovery. '
                'Use the preference profile message to formulate concise tool queries. '
            )
        else:
            instructions += (
                'MCP/API tools are DISABLED for the pure LLM baseline. Answer directly from model knowledge. '
                'This baseline is allowed to recommend from memory, but it should not claim API grounding or freshness. '
                'For source, write _(Source: LLM baseline)_. '
            )

        if use_history:
            instructions += 'Conversation history is ENABLED. Use previous turns for personalization. '
        else:
            instructions += 'Conversation history is DISABLED. Use only the latest user request and ignore prior turns. '

        if cross_domain:
            instructions += (
                'Cross-domain transfer is ENABLED. You may translate preferences from one entertainment domain '
                'to another using semantic similarities in mood, theme, creator style, genre, and tone. '
            )
        else:
            instructions += (
                'Cross-domain transfer is DISABLED. Do not use movie preferences for book/music requests, '
                'music preferences for movie/book requests, or book preferences for movie/music requests. '
                'Use only preferences explicitly stated in the requested target domain. '
            )

        if grounded_only:
            instructions += (
                'Grounded recommendation constraint is ENABLED. Never make up titles, artists, authors, years, '
                'sources, or descriptions. Final recommendations must come ONLY from MCP tool results. '
                'If tools return too few results, recommend only the available returned items and say that fewer results were found. '
            )
        else:
            instructions += 'Grounded recommendation constraint is DISABLED for this baseline. '

        instructions += (
            self._base_formatting_rules()
            + 'If the user prompt is vague, ask one specific clarifying question. '
            + 'Only decline if the request is clearly unrelated to movies, music, or books.'
        )
        return instructions

    def _tools_for_chat_completions(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI Responses-style function tools to Chat Completions format.

        MCPRegistry returns tools as:
        {"type": "function", "name": ..., "description": ..., "parameters": ...}

        OpenAI-compatible chat endpoints, including Ollama, expect:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        """
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if 'function' in tool:
                converted.append(tool)
                continue
            converted.append(
                {
                    'type': 'function',
                    'function': {
                        'name': tool.get('name', ''),
                        'description': tool.get('description', ''),
                        'parameters': tool.get('parameters', {'type': 'object', 'properties': {}}),
                    },
                }
            )
        return converted

    async def _create_response_openai(
        self,
        input_messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = await self.client.responses.create(
            model=self.model,
            input=input_messages,
            tools=tools,
        )

        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in response.output:
            if item.type == 'message':
                for content in item.content:
                    if getattr(content, 'type', None) == 'output_text':
                        text_chunks.append(content.text)
            elif item.type == 'function_call':
                tool_calls.append(
                    {
                        'name': item.name,
                        'arguments': item.arguments,
                        'call_id': item.call_id,
                    }
                )
        return {'output_text': ''.join(text_chunks).strip(), 'tool_calls': tool_calls}

    async def _create_response_ollama(
        self,
        input_messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            'model': self.model,
            'messages': input_messages,
        }
        if tools:
            kwargs['tools'] = self._tools_for_chat_completions(tools)
            kwargs['tool_choice'] = 'auto'

        response = await self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        output_text = message.content or ''

        tool_calls: list[dict[str, Any]] = []
        for call in getattr(message, 'tool_calls', None) or []:
            function = getattr(call, 'function', None)
            if function is None:
                continue
            tool_calls.append(
                {
                    'name': function.name,
                    'arguments': function.arguments or '{}',
                    'call_id': getattr(call, 'id', None) or function.name,
                }
            )
        return {'output_text': output_text.strip(), 'tool_calls': tool_calls}

    async def create_response(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        *,
        use_mcp: bool = True,
        use_history: bool = True,
        cross_domain: bool = True,
        grounded_only: bool = True,
        mode_name: str = 'full',
    ) -> Any:
        if not self.enabled or self.client is None:
            return {
                'output_text': (
                    'No LLM client is configured. Install the openai Python package, or configure OpenAI/Ollama.'
                ),
                'tool_calls': [],
            }

        input_messages = [
            {
                'role': 'system',
                'content': self._system_instructions(
                    use_mcp=use_mcp,
                    use_history=use_history,
                    cross_domain=cross_domain,
                    grounded_only=grounded_only,
                    mode_name=mode_name,
                ),
            },
            *messages,
        ]

        try:
            if self.provider == 'openai':
                return await self._create_response_openai(input_messages, tools if use_mcp else [])
            return await self._create_response_ollama(input_messages, tools if use_mcp else [])
        except Exception as exc:
            provider_hint = (
                'Make sure Ollama is running and the configured model is pulled.'
                if self.provider == 'ollama'
                else 'Check OPENAI_API_KEY and OPENAI_MODEL.'
            )
            return {
                'output_text': f'LLM provider {self.provider} failed: {exc}. {provider_hint}',
                'tool_calls': [],
            }

    async def create_followup(
        self,
        messages: list[dict[str, str]],
        tool_results: list[dict[str, Any]],
        *,
        cross_domain: bool = True,
        grounded_only: bool = True,
        mode_name: str = 'full',
    ) -> str:
        if not self.enabled or self.client is None:
            return 'Here are grounded recommendations collected from the available domain tools.'

        tool_context = '\n'.join(str(item) for item in tool_results)
        transfer_rule = (
            'Cross-domain transfer is ENABLED: explanations may connect preferences across movies, music, and books. '
            if cross_domain
            else 'Cross-domain transfer is DISABLED: explanations must not use preferences from non-target domains. '
        )
        grounding_rule = (
            'Use ONLY the provided tool results. Do not invent titles, years, artists, authors, sources, or descriptions. '
            if grounded_only
            else 'You may use tool results plus model knowledge because grounding is disabled in this baseline. '
        )

        input_messages = [
            {
                'role': 'system',
                'content': (
                    'You are an entertainment recommendation assistant. '
                    f'Current LLM provider: {self.provider}. Current experiment mode: {mode_name}. '
                    + transfer_rule
                    + grounding_rule
                    + 'Every recommendation MUST include source. Never omit source. '
                    'Use item.source when available. If item.source is missing, use result.source. '
                    'If both are missing, write _(Source: unknown)_. '
                    'Use clean markdown only. Do not use backslashes or escaped quotes. '
                    + self._base_formatting_rules()
                    + 'Use subtitle as the movie release date/year, artist name, or author name when available. '
                    'For movies, if subtitle is a full date like 2004-06-25, use only 2004. '
                    'Keep each description short, maximum 25 words. '
                    'If no relevant results were found, say so politely.'
                ),
            },
            *messages,
            {'role': 'user', 'content': f'Tool results JSON:\n{tool_context}'},
        ]

        try:
            if self.provider == 'openai':
                response = await self.client.responses.create(model=self.model, input=input_messages)
                return response.output_text.strip()
            response = await self.client.chat.completions.create(model=self.model, messages=input_messages)
            return (response.choices[0].message.content or '').strip()
        except Exception as exc:
            provider_hint = (
                'Make sure Ollama is running and the configured model is pulled.'
                if self.provider == 'ollama'
                else 'Check OPENAI_API_KEY and OPENAI_MODEL.'
            )
            return f'LLM provider {self.provider} failed: {exc}. {provider_hint}'
        
    async def create_json(self, messages: list[dict[str, str]]) -> dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content or "{}"

        try:
            return json.loads(text)
        except Exception:
            return {}
