import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from supabase import Client

from .config import get_settings
from .conversation import ConversationStore
from .mcp_registry import MCPRegistry
from .openai_service import OpenAIService
from .orchestrator import RecommendationOrchestrator
from .schemas import ChatRequest, CreateSessionRequest, MessageResponse, SessionResponse
from .supabase_client import get_supabase

settings = get_settings()
registry = MCPRegistry()
openai_service = OpenAIService()
store = ConversationStore()
orchestrator = RecommendationOrchestrator(registry, openai_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await registry.warmup()
    yield
    await registry.shutdown()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health() -> dict:
    return {
        'ok': True,
        'llm_enabled': openai_service.enabled,
        'llm_provider': openai_service.provider,
        'llm_model': openai_service.model,
        'domains': registry.domain_status(),
    }


@app.get('/sessions', response_model=list[SessionResponse])
def list_sessions(db: Client = Depends(get_supabase)):
    sessions = store.list_sessions(db)
    return [SessionResponse(id=s['public_id'], title=s['title']) for s in sessions]


@app.post('/sessions', response_model=SessionResponse)
def create_session(payload: CreateSessionRequest, db: Client = Depends(get_supabase)):
    session = store.create_session(db, payload.title)
    return SessionResponse(id=session['public_id'], title=session['title'])


@app.get('/sessions/{session_id}/messages', response_model=list[MessageResponse])
def get_messages(session_id: str, db: Client = Depends(get_supabase)):
    session = store.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    return [MessageResponse(role=m['role'], content=m['content']) for m in store.get_messages(db, session_id)]


@app.delete('/sessions/{session_id}', status_code=204)
def delete_session(session_id: str, db: Client = Depends(get_supabase)):
    session = store.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    store.delete_session(db, session_id)


@app.post('/sessions/{session_id}/stream')
def stream_chat(session_id: str, payload: ChatRequest, db: Client = Depends(get_supabase)):
    print("REQUEST MODE:", payload.mode)
    session = store.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    store.add_message(db, session_id, 'user', payload.message)
    history = [{'role': m['role'], 'content': m['content']} for m in store.get_messages(db, session_id)]

    async def event_publisher():
        assistant_parts: list[str] = []
        async for event in orchestrator.stream_chat(history, payload.mode, session_id=session_id):
            if event.startswith('event: text'):
                data = event.split('data: ', 1)[1].split('\n', 1)[0]
                assistant_parts.append(data.strip().strip('"'))
            yield event
        if assistant_parts:
            store.add_message(db, session_id, 'assistant', ''.join(assistant_parts))

    return EventSourceResponse(event_publisher())
