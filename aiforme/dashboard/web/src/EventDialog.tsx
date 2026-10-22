import { useEffect, useRef, useState } from "react";
import { KIND_LABELS, type EventInput, type Kind } from "./api";

interface Props {
  initial: EventInput;
  isNew: boolean;
  onSave: (event: EventInput) => Promise<void>;
  onDelete?: () => Promise<void>;
  onClose: () => void;
}

/** Create or edit one event. Esc or a click outside closes it without saving. */
export function EventDialog({ initial, isNew, onSave, onDelete, onClose }: Props) {
  const [form, setForm] = useState<EventInput>(initial);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const title = useRef<HTMLInputElement>(null);

  useEffect(() => {
    title.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const set = <K extends keyof EventInput>(key: K, value: EventInput[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <div className="backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <form
        className="dialog"
        onSubmit={(e) => {
          e.preventDefault();
          run(() => onSave({ ...form, title: form.title.trim() }));
        }}
      >
        <h2>{isNew ? "Новое событие" : "Событие"}</h2>

        <label>
          Название
          <input ref={title} value={form.title} onChange={(e) => set("title", e.target.value)} required
                 maxLength={200} placeholder="Например, созвон с командой" />
        </label>

        <fieldset className="kinds">
          <legend>Тип</legend>
          {(Object.keys(KIND_LABELS) as Kind[]).map((kind) => (
            <label key={kind} className={`kind kind-${kind}`}>
              <input type="radio" name="kind" checked={form.kind === kind} onChange={() => set("kind", kind)} />
              {KIND_LABELS[kind]}
            </label>
          ))}
        </fieldset>

        <div className="row">
          <label>
            Начало
            <input type="datetime-local" value={form.start} onChange={(e) => set("start", e.target.value)} required />
          </label>
          <label>
            Конец
            <input type="datetime-local" value={form.end} onChange={(e) => set("end", e.target.value)} required />
          </label>
        </div>

        <label>
          Заметки
          <textarea value={form.notes} onChange={(e) => set("notes", e.target.value)} rows={3} maxLength={2000}
                    placeholder={form.kind === "shift" ? "Ставка, место — пригодится финансовому трекеру" : ""} />
        </label>

        {error && <p className="error">{error}</p>}

        <div className="actions">
          {onDelete && (
            <button type="button" className="danger" disabled={busy} onClick={() => run(onDelete)}>
              Удалить
            </button>
          )}
          <span className="spacer" />
          <button type="button" onClick={onClose} disabled={busy}>Отмена</button>
          <button type="submit" className="primary" disabled={busy}>{isNew ? "Создать" : "Сохранить"}</button>
        </div>
      </form>
    </div>
  );
}
