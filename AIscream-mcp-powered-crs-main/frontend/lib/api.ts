export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000';

export async function apiFetch(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers || {});
  headers.set('Content-Type', 'application/json');

  let response: Response;
  try {
    response = await fetch(`${BACKEND_URL}${path}`, { ...init, headers });
  } catch {
    throw new Error(`Cannot reach backend at ${BACKEND_URL}. Make sure FastAPI is running.`);
  }

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return response;
}