// The calendar API (aiforme/dashboard/server.py). Times go both ways as local "YYYY-MM-DDTHH:mm" strings.

export type Kind = "meeting" | "shift" | "other";

export interface CalendarEvent {
  id: number;
  title: string;
  start: string;
  end: string;
  kind: Kind;
  notes: string;
}

export type EventInput = Omit<CalendarEvent, "id">;

export const KIND_LABELS: Record<Kind, string> = {
  meeting: "Встреча",
  shift: "Смена",
  other: "Другое",
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : res.statusText;
    throw new Error(detail);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

/** A Date as local wall-clock time, the way the calendar stores it (not toISOString, which is UTC). */
export function toLocal(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export const api = {
  list: (start: Date, end: Date) =>
    request<CalendarEvent[]>(`/api/events?start=${toLocal(start)}&end=${toLocal(end)}`),
  add: (event: EventInput) =>
    request<CalendarEvent>("/api/events", { method: "POST", body: JSON.stringify(event) }),
  update: (id: number, fields: Partial<EventInput>) =>
    request<CalendarEvent>(`/api/events/${id}`, { method: "PATCH", body: JSON.stringify(fields) }),
  remove: (id: number) => request<void>(`/api/events/${id}`, { method: "DELETE" }),
};

// --- commands and active learning (aiforme/active_learning) ------------------------------------------------

export type Slots = Record<string, string>;

export interface Label {
  intent: string;
  slots: Slots;
  verdict: "confirmed" | "corrected";
  version: number | null;          // the model version that trained on it; null while pending
  labelled_at: number;
}

export interface Command {
  id: number;
  heard_at: number;                // unix seconds
  text: string;
  intent: string;
  confidence: number;
  slots: Slots;
  best_guess: string | null;
  model: number;                   // the version that recognized it
  label: Label | null;
}

export interface Scores {
  accuracy: number;
  in_scope_accuracy: number | null;
  oos_recall: number | null;
}

export interface Version {
  id: number;
  created_at: number;
  embedder: string;
  accepted: boolean;
  new_labels: number;
  examples: number;
  metrics: {
    test: Scores;
    test_before?: Scores;
    real: Scores | null;
    real_before?: Scores;
    real_labels?: number;
    test_phrases_lost?: number;
    test_ok?: boolean;
    real_ok?: boolean;
  };
}

export interface Learning {
  version: number;
  embedder: string;
  intents: Record<string, string | null>;     // intent -> the slot it takes
  pending: { confirmed: number; corrected: number };
  retraining: boolean;
  error: string | null;
  versions: Version[];
  live: { model: number; labelled: number; correct: number }[];
}

export const INTENT_LABELS: Record<string, string> = {
  open_app: "Открыть приложение",
  close_app: "Закрыть приложение",
  find_file: "Найти файл",
  dictate: "Надиктовать текст",
  web_search: "Поиск в интернете",
  volume_up: "Громче",
  volume_down: "Тише",
  mute: "Без звука",
  media_play_pause: "Пауза / продолжить",
  media_next: "Следующий трек",
  screenshot: "Скриншот",
  lock_screen: "Заблокировать экран",
  tell_time: "Который час",
  cancel: "Отмена",
  set_mode: "Сменить режим",
  agenda: "Планы на сегодня",
  out_of_scope: "Не команда",
};

export const SLOT_LABELS: Record<string, string> = {
  app: "Приложение",
  query: "Что искать",
  text: "Текст",
  mode: "Режим",
};

export const learningApi = {
  commands: (limit = 100) => request<Command[]>(`/api/commands?limit=${limit}`),
  label: (id: number, intent: string, slots: Slots) =>
    request<Label>(`/api/commands/${id}/label`, { method: "PUT", body: JSON.stringify({ intent, slots }) }),
  unlabel: (id: number) => request<void>(`/api/commands/${id}/label`, { method: "DELETE" }),
  state: () => request<Learning>("/api/learning"),
  retrain: () => request<{ retraining: boolean }>("/api/learning/retrain", { method: "POST" }),
};
