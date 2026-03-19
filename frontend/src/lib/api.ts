const BASE = '';

export async function api<T = unknown>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export interface StreamHandlers {
  onMeta?: (data: Record<string, unknown>) => void;
  onDelta?: (data: { text: string }) => void;
  onDone?: (data: Record<string, unknown>) => void;
  onError?: (data: { detail: string }) => void;
}

export async function streamApi(
  path: string,
  body: Record<string, unknown>,
  handlers: StreamHandlers,
): Promise<void> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Stream request failed: ${res.status}`);
  }
  const reader = res.body?.getReader();
  if (!reader) throw new Error('No stream body');
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    let currentEvent = '';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (currentEvent === 'meta') handlers.onMeta?.(data);
        else if (currentEvent === 'delta') handlers.onDelta?.(data);
        else if (currentEvent === 'done') handlers.onDone?.(data);
        else if (currentEvent === 'error') handlers.onError?.(data);
      }
    }
  }
}

// ---- Type definitions ----

export interface Stats {
  papers: number;
  embeddings: number;
  pdf_catalog: number;
  methodology_runs: number;
  [key: string]: unknown;
}

export interface Paper {
  entry_id: string;
  title: string;
  abstract?: string;
  authors?: string[];
  source?: string;
  published?: string;
  link?: string;
  pdf_url?: string;
  doi?: string;
  tags?: string[];
  [key: string]: unknown;
}

export interface SummaryResult {
  date: string;
  language: string;
  paper_count: number;
  summary: string;
  papers: Paper[];
  cached: boolean;
  path: string;
  selection_mode: string;
  topic: string;
  actions?: string[];
  imported_count?: number;
}
