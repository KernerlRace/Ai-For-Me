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
