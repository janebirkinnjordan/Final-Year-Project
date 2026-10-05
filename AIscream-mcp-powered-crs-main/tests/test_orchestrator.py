import pytest

from backend.app.orchestrator import RecommendationOrchestrator


class StubRegistry:
    async def list_openai_tools(self):
        return []


class StubOpenAI:
    async def create_response(self, messages, tools, **kwargs):
        return {'output_text': 'Here are grounded recommendations.', 'tool_calls': []}


@pytest.mark.asyncio
async def test_orchestrator_streams_text_and_done():
    orchestrator = RecommendationOrchestrator(StubRegistry(), StubOpenAI())
    events = []
    async for item in orchestrator.stream_chat([{'role': 'user', 'content': 'Hi'}]):
        events.append(item)
    assert any('event: text' in event for event in events)
    assert any('event: done' in event for event in events)
