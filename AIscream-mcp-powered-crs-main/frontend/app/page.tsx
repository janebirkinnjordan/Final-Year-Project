'use client';

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';

import Header from '@/components/Header';
import RecommendationCard from '@/components/RecommendationCard';
import { apiFetch, BACKEND_URL } from '@/lib/api';

type Message = { role: 'user' | 'assistant'; content: string };
type Session = { id: string; title: string };
type ExperimentMode = 'full' | 'pure_gpt' | 'no_history' | 'no_cross_domain';

type Recommendation = {
  domain: string;
  title: string;
  subtitle?: string;
  summary?: string;
  source?: string;
  year?: string;
  artist?: string;
  author?: string;
};

function cleanAssistantMarkdown(value: string) {
  let section: 'movie' | 'music' | 'book' | undefined;

  const sectionAwareText = value
    .split(/\r?\n/)
    .map((line) => {
      const lower = line.toLowerCase();

      if (/movies?|films?/.test(lower) && !/^\s*[-*]/.test(line)) section = 'movie';
      if (/songs?|music|tracks?/.test(lower) && !/^\s*[-*]/.test(line)) section = 'music';
      if (/books?|novels?/.test(lower) && !/^\s*[-*]/.test(line)) section = 'book';

      if (section === 'music') {
        return line.replace(
          /^(\s*[-*]\s+\*\*[^*]+?\*\*)\s+by\s+([^—\n]+?)\s+[—-]\s+/,
          '$1 (Artist: $2) — '
        );
      }

      if (section === 'book') {
        return line.replace(
          /^(\s*[-*]\s+\*\*[^*]+?\*\*)\s+by\s+([^—\n]+?)\s+[—-]\s+/,
          '$1 (Author: $2) — '
        );
      }

      return line;
    })
    .join('\n');

  return sectionAwareText
    .replace(/\r\n/g, '\n')
    .replace(/\\n/g, '\n')
    .replace(/\\+"/g, '"')
    .replace(/\\+'/g, "'")
    .replace(/^[ \t]*\\+[ \t]*$/gm, '')
    .replace(/\\+[ \t]*$/gm, '')

    // Fix: ** — Title** -> **Title**
    .replace(/\*\*[ \t]*[—-][ \t]*/g, '**')

    // Fix: — **Title** -> - **Title**
    // This restores bullets if an earlier output used em dashes as list markers.
    .replace(/^([ \t]*)—[ \t]+(\*\*)/gm, '$1- $2')

    // Fix: - **Title**Description
    // into: - **Title** — Description
    .replace(
      /^([ \t]*[-*][ \t]+\*\*[^*]+?\*\*)([A-Z][^\n]*)/gm,
      '$1 — $2'
    )

    .replace(/^([ \t]*[-*][ \t]+)\*"([^"]+)"\*[ \t]*[—-][ \t]*/gm, '$1**$2** — ')
    .replace(/^([ \t]*[-*][ \t]+)\*'([^']+)'\*[ \t]*[—-][ \t]*/gm, '$1**$2** — ')
    .replace(/^([ \t]*[-*][ \t]+)\*([^*]+)\*[ \t]*[—-][ \t]*/gm, '$1**$2** — ')

    // Fix: **Title** A description...
    // into: **Title** — A description...
    .replace(/(\*\*[^*]+?\*\*)\s+((?:A|An|The|This|These|It|Its)\b)/g, '$1 — $2')

    // Normalize em dash spacing only.
    // Do NOT include normal hyphen "-" here, or markdown bullets will break.
    .replace(/[ \t]*—[ \t]*/g, ' — ')

    // Restore bullets again after dash spacing normalization.
    .replace(/^([ \t]*)—[ \t]+(\*\*)/gm, '$1- $2')

    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export default function HomePage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionId, setSessionId] = useState('');
  const sessionIdRef = useRef('');

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const inputRef = useRef('');

  const [cards, setCards] = useState<Recommendation[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<ExperimentMode>('full');

  const bootstrapDone = useRef(false);
  const pendingSubmit = useRef(false);

  function updateSessionId(id: string) {
    setSessionId(id);
    sessionIdRef.current = id;

    if (typeof window !== 'undefined') {
      if (id) {
        window.localStorage.setItem('activeSessionId', id);
      } else {
        window.localStorage.removeItem('activeSessionId');
      }
    }
  }

  function updateInput(val: string) {
    setInput(val);
    inputRef.current = val;
  }

  useEffect(() => {
    void bootstrap();
  }, []);

  async function bootstrap() {
    try {
      const response = await apiFetch('/sessions');
      const data: Session[] = await response.json();

      if (data.length === 0) {
        setSessions([]);
        updateSessionId('');
      } else {
        setSessions(data);

        const savedSessionId =
          typeof window !== 'undefined' ? window.localStorage.getItem('activeSessionId') : '';
        const sessionToLoad =
          savedSessionId && data.some((session) => session.id === savedSessionId)
            ? savedSessionId
            : data[0].id;

        await loadSession(sessionToLoad);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load sessions');
    } finally {
      bootstrapDone.current = true;

      if (pendingSubmit.current) {
        pendingSubmit.current = false;
        await sendMessage();
      }
    }
  }

  async function loadSession(id: string) {
    try {
      updateSessionId(id);

      const response = await apiFetch(`/sessions/${id}/messages`);
      const data: Message[] = await response.json();

      setMessages(data);
      setCards([]);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load session');
    }
  }

  async function createSession(title?: string): Promise<Session> {
    const created: Session = await apiFetch('/sessions', {
      method: 'POST',
      body: JSON.stringify({
        title: title ?? `Conversation ${sessions.length + 1}`,
      }),
    }).then((res) => res.json());

    setSessions((prev) => [created, ...prev]);
    updateSessionId(created.id);
    setMessages([]);
    setCards([]);

    return created;
  }

  async function deleteSession(id: string) {
    try {
      await apiFetch(`/sessions/${id}`, { method: 'DELETE' });

      const remaining = sessions.filter((s) => s.id !== id);
      setSessions(remaining);

      if (sessionIdRef.current === id) {
        if (remaining.length > 0) {
          await loadSession(remaining[0].id);
        } else {
          updateSessionId('');
          setMessages([]);
          setCards([]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete session');
    }
  }

  async function sendMessage() {
    const currentInput = inputRef.current;

    if (!currentInput.trim() || loading) return;

    setError('');
    setLoading(true);
    setCards([]);

    let activeSessionId = sessionIdRef.current;

    if (!activeSessionId) {
      try {
        const newSession = await createSession('Conversation 1');
        activeSessionId = newSession.id;
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Could not create session');
        setLoading(false);
        return;
      }
    }

    const sentInput = currentInput;
    updateInput('');

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: sentInput },
      { role: 'assistant', content: '' },
    ]);

    try {
      const response = await fetch(`${BACKEND_URL}/sessions/${activeSessionId}/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({ message: sentInput, mode }),
      });

      if (!response.ok || !response.body) {
        throw new Error((await response.text()) || 'Streaming request failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = '';
      let assistantText = '';

      const appendAssistantText = (text: string) => {
        assistantText += text;

        setMessages((prev) => {
          const next = [...prev];

          if (next.length === 0) return next;

          next[next.length - 1] = {
            role: 'assistant',
            content: assistantText,
          };

          return next;
        });
      };

      const processEvents = (raw: string) => {
        const events = raw.split('\n\n');

        for (const rawEvent of events) {
          const lines = rawEvent.split('\n');
          let eventType = 'message';
          const dataLines: string[] = [];

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataLines.push(line.slice(5).trim());
            }
          }

          if (dataLines.length === 0) continue;

          const rawData = dataLines.join('\n');

          let data: unknown;

          try {
            data = JSON.parse(rawData);
          } catch {
            data = rawData;
          }

          if (eventType === 'text') {
            appendAssistantText(String(data));
          } else if (eventType === 'tools') {
            const nextCards = (
              data as Array<{ result?: { items?: Recommendation[] } }>
            ).flatMap((item) => item.result?.items || []);

            setCards(nextCards);
          } else if (eventType === 'error') {
            throw new Error(typeof data === 'string' ? data : 'Stream error');
          }
        }
      };

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          const tail = decoder.decode();

          if (tail) buffer += tail;
          if (buffer.trim()) processEvents(buffer);

          break;
        }

        buffer += decoder.decode(value, { stream: true });
        buffer = buffer.replace(/\r\n/g, '\n');

        const splitAt = buffer.lastIndexOf('\n\n');

        if (splitAt !== -1) {
          const complete = buffer.slice(0, splitAt);
          buffer = buffer.slice(splitAt + 2);
          processEvents(complete);
        }
      }

      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId && /^Conversation \d+$/.test(s.title)
            ? {
                ...s,
                title: sentInput.slice(0, 40) + (sentInput.length > 40 ? '…' : ''),
              }
            : s
        )
      );

      await new Promise((resolve) => setTimeout(resolve, 500));
      await loadSession(activeSessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed');

      setMessages((prev) => {
        const next = [...prev];

        if (
          next.length > 0 &&
          next[next.length - 1].role === 'assistant' &&
          next[next.length - 1].content === ''
        ) {
          next[next.length - 1] = {
            role: 'assistant',
            content: 'Sorry, something went wrong while generating the response.',
          };
        }

        return next;
      });
    } finally {
      setLoading(false);
    }
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();

    if (!bootstrapDone.current) {
      pendingSubmit.current = true;
      return;
    }

    await sendMessage();
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void onSubmit(event as unknown as FormEvent);
    }
  }

  return (
    <div className="layout">
      <Header />

      <div className="grid">
        <aside className="surface sidebar col">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <strong>Conversations</strong>

            <button className="button secondary" onClick={() => void createSession()}>
              New
            </button>
          </div>

          {sessions.map((session) => (
            <div key={session.id} className="row" style={{ gap: '4px', alignItems: 'center' }}>
              <button
                className="button secondary"
                style={{
                  flex: 1,
                  textAlign: 'left',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
                onClick={() => void loadSession(session.id)}
              >
                {session.title}
              </button>

              <button
                className="button secondary"
                style={{ flexShrink: 0, padding: '4px 8px' }}
                title="Delete conversation"
                onClick={() => void deleteSession(session.id)}
              >
                ✕
              </button>
            </div>
          ))}

          {sessions.length === 0 && (
            <div className="subtle">No conversations yet. Send a message to start one.</div>
          )}
        </aside>

        <section className="surface panel col">
          <div className="chatLog surface" style={{ boxShadow: 'none' }}>
            {messages.length === 0 && (
              <div className="subtle">Start with mood, genre, tone, or a cross-domain request.</div>
            )}

            {messages.map((message, idx) => (
              <div
                key={`${message.role}-${idx}`}
                className={`bubble ${message.role === 'user' ? 'user' : 'assistant'}`}
              >
                {message.role === 'assistant' ? (
                  <div className="markdown">
                    <ReactMarkdown>
                      {cleanAssistantMarkdown(message.content || (loading ? 'Recommending...' : ''))}
                    </ReactMarkdown>
                  </div>
                ) : (
                  message.content
                )}
              </div>
            ))}
          </div>

          <form className="col" onSubmit={onSubmit}>
            <textarea
              className="textarea"
              value={input}
              onChange={(e) => updateInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="What are you in the mood for?"
            />

            {error && <div className="error">{error}</div>}

            <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="row" style={{ gap: '8px', alignItems: 'center' }}>
                <span className="subtle">Experiment</span>
                <select
                  className="button secondary"
                  value={mode}
                  onChange={(e) => setMode(e.target.value as ExperimentMode)}
                  disabled={loading}
                >
                  <option value="full">Full MCP + history + cross-domain</option>
                  <option value="pure_gpt">Pure GPT-4o baseline</option>
                  <option value="no_history">MCP without history</option>
                  <option value="no_cross_domain">MCP without cross-domain transfer</option>
                </select>
              </label>

              <button className="button" type="submit" disabled={loading}>
                Send
              </button>
            </div>
          </form>

          {cards.length > 0 && (
            <div className="col">
              <div className="row">
                <strong>Recommendation cards</strong>
              </div>

              <div className="cards">
                {cards.map((item, idx) => (
                  <RecommendationCard key={`${item.title}-${idx}`} item={item} />
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
