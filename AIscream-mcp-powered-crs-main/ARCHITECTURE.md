# Architecture Overview

## Backend responsibilities

- persist sessions and messages in Supabase
- keep startup cheap and predictable
- lazily connect to MCP servers only when chat requests need tools
- send the selected LLM provider the MCP tool schemas
- execute tool calls against the correct MCP server
- stream assistant output to the frontend through SSE

## MCP responsibilities

Each MCP server owns one domain:

- movies: TMDB primary, OMDb backup
- books: Open Library primary, Google Books backup
- music: Last.fm primary, MusicBrainz backup

Each server exposes one recommendation tool and performs fallback internally.

## Frontend responsibilities

- create and switch chat sessions
- stream assistant events and render recommendation cards
- show actionable fetch errors instead of a vague failure state

## Data flow

```
User → Frontend (Next.js)
     → Backend (FastAPI)
     → LLM Provider (OpenAI if OPENAI_API_KEY exists, otherwise Ollama)
     → MCP Servers (movies / books / music)
     → External APIs (TMDB / LastFM / Google Books etc.)
     → Supabase (session + message storage)
```
