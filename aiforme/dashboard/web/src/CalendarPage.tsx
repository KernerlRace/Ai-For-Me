import { useCallback, useRef, useState } from "react";
import FullCalendar from "@fullcalendar/react";
import dayGridPlugin from "@fullcalendar/daygrid";
import timeGridPlugin from "@fullcalendar/timegrid";
import interactionPlugin from "@fullcalendar/interaction";
import ruLocale from "@fullcalendar/core/locales/ru";
import type {
  DateSelectArg, EventChangeArg, EventClickArg, EventInput as FcEvent, EventSourceFuncArg,
} from "@fullcalendar/core";
import { api, toLocal, type CalendarEvent, type EventInput } from "./api";
import { EventDialog } from "./EventDialog";

type Editing = { id: number | null; initial: EventInput };

/** Week view by default; drag on empty time to create, drag or stretch an event to move it, click to edit. */
export function CalendarPage() {
  const calendar = useRef<FullCalendar>(null);
  const [editing, setEditing] = useState<Editing | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refetch = () => calendar.current?.getApi().refetchEvents();

  const load = useCallback(async (info: EventSourceFuncArg): Promise<FcEvent[]> => {
    try {
      const events = await api.list(info.start, info.end);
      setError(null);
      return events.map(toFc);
    } catch (e) {
      setError(`Не удалось загрузить календарь: ${e instanceof Error ? e.message : e}`);
      return [];
    }
  }, []);

  function onSelect(sel: DateSelectArg) {
    sel.view.calendar.unselect();
    let end = sel.end;
    if (sel.allDay) {                     // a day clicked in the month view: suggest 10:00–11:00 that day
      const start = new Date(sel.start);
      start.setHours(10, 0, 0, 0);
      end = new Date(start.getTime() + 60 * 60 * 1000);
      sel = { ...sel, start };
    }
    setEditing({ id: null, initial: { title: "", start: toLocal(sel.start), end: toLocal(end), kind: "meeting", notes: "" } });
  }

  function onClick(click: EventClickArg) {
    const ev = click.event.extendedProps.source as CalendarEvent;
    setEditing({ id: ev.id, initial: { title: ev.title, start: ev.start, end: ev.end, kind: ev.kind, notes: ev.notes } });
  }

  async function onChange(change: EventChangeArg) {
    const { start, end } = change.event;
    if (!start || !end) return change.revert();
    try {
      await api.update(Number(change.event.id), { start: toLocal(start), end: toLocal(end) });
      refetch();
    } catch (e) {
      change.revert();
      setError(`Не удалось перенести: ${e instanceof Error ? e.message : e}`);
    }
  }

  return (
    <section className="page">
      {error && <p className="banner">{error}</p>}
      <FullCalendar
        ref={calendar}
        plugins={[timeGridPlugin, dayGridPlugin, interactionPlugin]}
        locale={ruLocale}
        initialView="timeGridWeek"
        headerToolbar={{ left: "prev,next today", center: "title", right: "timeGridDay,timeGridWeek,dayGridMonth" }}
        height="100%"
        nowIndicator
        selectable
        selectMirror
        editable
        allDaySlot={false}
        slotMinTime="06:00:00"
        scrollTime="08:00:00"
        firstDay={1}
        events={load}
        select={onSelect}
        eventClick={onClick}
        eventChange={onChange}
      />
      {editing && (
        <EventDialog
          initial={editing.initial}
          isNew={editing.id === null}
          onClose={() => setEditing(null)}
          onSave={async (event) => {
            if (editing.id === null) await api.add(event);
            else await api.update(editing.id, event);
            setEditing(null);
            refetch();
          }}
          onDelete={editing.id === null ? undefined : async () => {
            await api.remove(editing.id!);
            setEditing(null);
            refetch();
          }}
        />
      )}
    </section>
  );
}

function toFc(ev: CalendarEvent): FcEvent {
  return {
    id: String(ev.id), title: ev.title, start: ev.start, end: ev.end,
    classNames: [`kind-${ev.kind}`], extendedProps: { source: ev },
  };
}
