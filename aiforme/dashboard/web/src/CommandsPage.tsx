import { useCallback, useEffect, useState } from "react";
import {
  INTENT_LABELS, SLOT_LABELS, learningApi,
  type Command, type CommandOrder, type Learning, type Slots, type Version,
} from "./api";
import { AccuracyChart, type AccuracyPoint } from "./AccuracyChart";
import { t } from "./i18n";

const intentName = (intent: string) => INTENT_LABELS[intent] ?? intent;
const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
const time = (s: number) => new Date(s * 1000).toLocaleString(t.locale, {
  day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
});

/** What the assistant heard, my verdict on each command, and the retraining those verdicts feed. */
export function CommandsPage() {
  const [commands, setCommands] = useState<Command[]>([]);
  const [learning, setLearning] = useState<Learning | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [order, setOrder] = useState<CommandOrder>("uncertain");

  const load = useCallback(async () => {
    try {
      const [cmds, state] = await Promise.all([learningApi.commands(order), learningApi.state()]);
      setCommands(cmds);
      setLearning(state);
      setError(null);
    } catch (e) {
      setError(t.commands.loadFailed(e instanceof Error ? e.message : String(e)));
    }
  }, [order]);

  useEffect(() => {
    load();
    const timer = window.setInterval(load, learning?.retraining ? 2_000 : 10_000);
    return () => window.clearInterval(timer);
  }, [load, learning?.retraining]);

  async function act(action: () => Promise<unknown>) {
    try {
      await action();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  if (!learning) return <section className="page scroll">{error && <p className="banner">{error}</p>}</section>;
  const pending = learning.pending.confirmed + learning.pending.corrected;

  return (
    <section className="page scroll">
      {error && <p className="banner">{error}</p>}
      <div className="cards">
        <Overview learning={learning} pending={pending} onRetrain={() => act(learningApi.retrain)} />
        <div className="card wide">
          <h3>{t.chart.title}</h3>
          <AccuracyChart points={chartPoints(learning)} />
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h3>{t.commands.heading}</h3>
          <div className="toggle" role="group" aria-label={t.commands.order}>
            {(["uncertain", "recent"] as const).map((o) => (
              <button key={o} className={o === order ? "active" : ""} aria-pressed={o === order}
                      onClick={() => setOrder(o)}>
                {o === "uncertain" ? t.commands.uncertain : t.commands.recent}
              </button>
            ))}
          </div>
        </div>
        {order === "uncertain" && commands.length > 0 && (
          <p className="muted small hint">{t.commands.hint}</p>
        )}
        {commands.length === 0 && (
          <p className="muted">{t.commands.empty} <code>{t.commands.emptyExample}</code>.</p>
        )}
        <ul className="commands">
          {commands.map((c) => (
            <CommandRow key={c.id} command={c} intents={learning.intents}
                        onLabel={async (intent, slots) => { await learningApi.label(c.id, intent, slots); await load(); }}
                        onUndo={async () => { await learningApi.unlabel(c.id); await load(); }} />
          ))}
        </ul>
      </div>

      {learning.versions.length > 0 && <History versions={learning.versions} />}
    </section>
  );
}

function chartPoints(learning: Learning): AccuracyPoint[] {
  const accepted = learning.versions.filter((v) => v.accepted);
  const ids = new Set([0, ...accepted.map((v) => v.id), ...learning.live.map((l) => l.model)]);
  return [...ids].sort((a, b) => a - b).map((id) => ({
    version: id,
    test: accepted.find((v) => v.id === id)?.metrics.test.accuracy ?? null,
    live: learning.live.find((l) => l.model === id) ?? null,
  }));
}

function Overview({ learning, pending, onRetrain }: { learning: Learning; pending: number; onRetrain: () => void }) {
  const last = learning.versions.at(-1);
  const live = learning.live.find((l) => l.model === learning.version);
  return (
    <div className="card">
      <h3>{t.model.heading}</h3>
      <div className="stats">
        <div><strong>v{learning.version}</strong><span>{t.model.current(learning.embedder)}</span></div>
        <div>
          <strong>{live ? pct(live.correct / live.labelled) : "—"}</strong>
          <span>{live ? t.model.rightOnMine(live.correct, live.labelled) : t.model.rightOnMineBare}</span>
        </div>
        <div>
          <strong>{pending}</strong>
          <span>{t.model.newLabels(learning.pending.confirmed, learning.pending.corrected)}</span>
        </div>
      </div>
      <div className="actions">
        <button className="primary" disabled={pending === 0 || learning.retraining} onClick={onRetrain}>
          {learning.retraining ? t.model.retraining : t.model.retrain}
        </button>
        {pending === 0 && !learning.retraining && <span className="muted">{t.model.needLabel}</span>}
      </div>
      {learning.auto.every > 0 && !learning.retraining && (
        <p className="muted small">{t.model.auto(Math.max(learning.auto.every - learning.auto.new, 0),
          learning.auto.new, learning.auto.every)}</p>
      )}
      {learning.error && <p className="error">{t.model.failed(learning.error)}</p>}
      {last && !learning.retraining && <Outcome version={last} automatic={learning.automatic} />}
    </div>
  );
}

function Outcome({ version: v, automatic }: { version: Version; automatic: boolean }) {
  if (v.id === 0) return null;
  const m = v.metrics;
  const how = automatic ? t.model.automatic : "";
  const reasons = [];
  if (m.test_ok === false) reasons.push(t.model.lost(m.test_phrases_lost));
  if (m.real_ok === false) reasons.push(t.model.worse(pct(m.real_before?.accuracy), pct(m.real?.accuracy)));
  return v.accepted ? (
    <p className="outcome ok">{t.model.accepted(v.id, how, v.new_labels, pct(m.test_before?.accuracy),
      pct(m.test.accuracy), pct(m.real_before?.accuracy), pct(m.real?.accuracy))}</p>
  ) : (
    <p className="outcome bad">{t.model.rejected(v.id, how, reasons.join("; "))}</p>
  );
}

interface RowProps {
  command: Command;
  intents: Record<string, string | null>;
  onLabel: (intent: string, slots: Slots) => Promise<void>;
  onUndo: () => Promise<void>;
}

/** One heard command. Its actions report failures in the row itself: the page banner may be scrolled away. */
function CommandRow({ command: c, intents, onLabel, onUndo }: RowProps) {
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const label = c.label;

  async function run(action: () => Promise<void>): Promise<boolean> {
    setError(null);
    try {
      await action();
      return true;
    } catch (e) {
      setError(t.commands.notSaved(e instanceof Error ? e.message : String(e)));
      return false;
    }
  }
  const shown = label ?? { intent: c.intent, slots: c.slots };

  return (
    <li className={`command ${label ? `labelled ${label.verdict}` : ""}`}>
      <div className="said">
        <span className="when">{time(c.heard_at)}</span>
        <span className="text">{t.quote(c.text)}</span>
      </div>
      <div className="parse">
        {label?.verdict === "corrected" && (
          <span className="was">{describe(c.intent, c.slots)} →</span>
        )}
        <span className="intent">{describe(shown.intent, shown.slots)}</span>
        {!label && (
          <span className="muted">
            {Math.round(c.confidence * 100)}%{c.intent === "out_of_scope" && c.best_guess
              ? t.commands.looksLike(intentName(c.best_guess)) : ""}{rival(c)} · v{c.model}
          </span>
        )}
      </div>
      <div className="row-actions">
        {!label && !editing && (
          <>
            <button onClick={() => run(() => onLabel(c.intent, c.slots))}>{t.commands.right}</button>
            <button onClick={() => setEditing(true)}>{t.commands.fix}</button>
          </>
        )}
        {label && (label.version === null ? (
          <>
            <span className="muted">{label.verdict === "confirmed" ? t.commands.rightDone : t.commands.fixedDone}</span>
            <button onClick={() => run(onUndo)}>{t.commands.undo}</button>
          </>
        ) : (
          <span className="muted">{t.commands.inVersion(label.version)}</span>
        ))}
      </div>
      {editing && (
        <Correction command={c} intents={intents} onCancel={() => setEditing(false)}
                    onSave={async (intent, slots) => {
                      if (await run(() => onLabel(intent, slots))) setEditing(false);
                    }} />
      )}
      {error && <p className="error row-error">{error}</p>}
    </li>
  );
}

/** ", or “Close app” 40%" when the runner-up came close; nothing for older events or a clear win. */
function rival(c: Command) {
  // a rejected command whose rival is "not a command" itself already says so
  if (!c.runner_up || c.runner_up === c.intent || c.runner_up_confidence === null || c.intent_margin === null
      || c.intent_margin > CLOSE) {
    return "";
  }
  return t.commands.or(intentName(c.runner_up), Math.round(c.runner_up_confidence * 100));
}

const CLOSE = 0.2;    // a runner-up within this of the top class is worth showing

function describe(intent: string, slots: Slots) {
  const value = Object.values(slots)[0];
  return value ? `${intentName(intent)} ${t.quote(value)}` : intentName(intent);
}

function Correction({ command: c, intents, onSave, onCancel }: {
  command: Command;
  intents: Record<string, string | null>;
  onSave: (intent: string, slots: Slots) => Promise<void>;
  onCancel: () => void;
}) {
  const start = c.intent === "out_of_scope" && c.best_guess ? c.best_guess : c.intent;
  const [intent, setIntent] = useState(start);
  const slot = intents[intent] ?? null;
  const [value, setValue] = useState(Object.values(c.slots)[0] ?? "");

  return (
    <form className="correction" onSubmit={(e) => {
      e.preventDefault();
      onSave(intent, slot ? { [slot]: value.trim() } : {});
    }}>
      <label>
        {t.commands.whatWasIt}
        <select value={intent} onChange={(e) => setIntent(e.target.value)} autoFocus>
          {Object.keys(intents).map((i) => <option key={i} value={i}>{intentName(i)}</option>)}
        </select>
      </label>
      {slot && (
        <label>
          {t.commands.slotWords(SLOT_LABELS[slot] ?? slot)}
          <input value={value} onChange={(e) => setValue(e.target.value)} required
                 placeholder={c.text} />
        </label>
      )}
      <div className="actions">
        <button type="button" onClick={onCancel}>{t.commands.cancel}</button>
        <button type="submit" className="primary">{t.commands.save}</button>
      </div>
    </form>
  );
}

function History({ versions }: { versions: Version[] }) {
  return (
    <div className="card">
      <h3>{t.history.heading}</h3>
      <table className="history">
        <thead>
          <tr>{t.history.columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {[...versions].reverse().map((v) => (
            <tr key={v.id} className={v.accepted ? "" : "rejected"}>
              <td>v{v.id}</td>
              <td>{time(v.created_at)}</td>
              <td>{v.id === 0 ? t.history.initial : v.accepted ? t.history.accepted : t.history.rejected}</td>
              <td>{v.id === 0 ? "—" : v.new_labels}</td>
              <td>{v.metrics.test_before ? `${pct(v.metrics.test_before.accuracy)} → ` : ""}{pct(v.metrics.test.accuracy)}</td>
              <td>{v.metrics.real_before ? `${pct(v.metrics.real_before.accuracy)} → ${pct(v.metrics.real?.accuracy)}` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">{t.history.note}</p>
    </div>
  );
}
