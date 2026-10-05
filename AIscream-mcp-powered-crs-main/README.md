# MCP-Powered CRS

A conversational entertainment recommender for movies, music, and books powered by MCP and OpenAI.

## What it does

- Recommends movies, music, and books through natural conversation
- Asks clarifying questions when prompts are vague
- Persists conversations across different machines via Supabase
- Streams assistant responses in real time

## Stack

- Frontend: Next.js App Router
- Backend: FastAPI
- MCP servers: Python + MCP SDK (`FastMCP`)
- AI: OpenAI Responses API (`gpt-4o-mini`)
- Database: Supabase

## Project structure

```text
frontend/          ← Next.js frontend
backend/           ← FastAPI backend
mcp_servers/
  movies/          ← TMDB + OMDb
  music/           ← Last.fm + MusicBrainz
  books/           ← Open Library + Google Books
shared/            ← shared HTTP utilities
tests/             ← backend tests
```

## First time setup (do this once)

### 1) Get the `.env` file from the project owner and place it at:
```
backend/.env
```

### 2) Set up backend

**Mac:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

**Windows:**
```bash
cd backend
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cd ..
```

### 3) Set up frontend

```bash
cd frontend
npm install
cd ..
```

### 4) Mac only — make start script executable

```bash
chmod +x start.sh
```

---

## Every time after that

**Mac:**
```bash
./start.sh
```

**Windows:**
```bash
./start.bat
```

This will start the backend, frontend, and open your browser automatically at `http://localhost:3000`.

---

## Default behavior when API keys are missing

- OpenAI is required for chat orchestration
- Movies can still run with OMDb if TMDB is missing
- Books can still run with Open Library if Google Books key is missing
- Music can still run with MusicBrainz if Last.fm key is missing
- If both primary and backup are unavailable, that domain is disabled and the app continues running

## API endpoints

- `GET    /health`
- `GET    /sessions`
- `POST   /sessions`
- `GET    /sessions/{session_id}/messages`
- `DELETE /sessions/{session_id}`
- `POST   /sessions/{session_id}/stream`

## Test

```bash
cd backend
pytest -q
```
