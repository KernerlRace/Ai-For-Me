import { useCallback, useEffect, useState } from "react";
import {
  INTENT_LABELS, SLOT_LABELS, learningApi,
  type Command, type Learning, type Slots, type Version,
} from "./api";
import { AccuracyChart, type AccuracyPoint } from "./AccuracyChart";

const intentName = (intent: string) => INTENT_LABELS[intent] ?? intent;
const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
const time = (s: number) => new Date(s * 1000).toLocaleString("ru-RU", {
  day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
});

/** What the assistant heard, my verdict on each command, and the retraining those verdicts feed. */
export function CommandsPage() {
  const [commands, setCommands] = useState<Command[]>([]);
  const [learning, setLearning] = useState<Learning | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [cmds, state] = await Promise.all([learningApi.commands(), learningApi.state()]);
      setCommands(cmds);
      setLearning(state);
      setError(null);
    } catch (e) {
      setError(`Не удалось загрузить команды: ${e instanceof Error ? e.message : e}`);
    }
  }, []);

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
          <h3>Точность по версиям</h3>
          <AccuracyChart points={chartPoints(learning)} />
        </div>
      </div>

      <div className="card">
        <h3>Команды</h3>
        {commands.length === 0 && (
          <p className="muted">Пока пусто: скажите ассистенту что-нибудь или отправьте
            <code> python -m aiforme say "открой телеграм"</code>.</p>
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
      <h3>Модель</h3>
      <div className="stats">
        <div><strong>v{learning.version}</strong><span>текущая версия · {learning.embedder}</span></div>
        <div>
          <strong>{live ? pct(live.correct / live.labelled) : "—"}</strong>
          <span>верно на моих командах{live ? ` (${live.correct} из ${live.labelled})` : ""}</span>
        </div>
        <div>
          <strong>{pending}</strong>
          <span>новых меток: {learning.pending.confirmed} верно, {learning.pending.corrected} исправлено</span>
        </div>
      </div>
      <div className="actions">
        <button className="primary" disabled={pending === 0 || learning.retraining} onClick={onRetrain}>
          {learning.retraining ? "Переобучаю…" : "Переобучить"}
        </button>
        {pending === 0 && !learning.retraining && <span className="muted">Нужна хотя бы одна новая метка.</span>}
      </div>
      {learning.error && <p className="error">Переобучение упало: {learning.error}</p>}
      {last && !learning.retraining && <Outcome version={last} />}
    </div>
  );
}

function Outcome({ version: v }: { version: Version }) {
  if (v.id === 0) return null;
  const m = v.metrics;
  const reasons = [];
  if (m.test_ok === false) reasons.push(`теряет ${m.test_phrases_lost} тестовых фраз (можно не больше одной)`);
  if (m.real_ok === false) reasons.push(`на моих командах хуже: ${pct(m.real_before?.accuracy)} → ${pct(m.real?.accuracy)}`);
  return v.accepted ? (
    <p className="outcome ok">✓ Версия {v.id} принята: {v.new_labels} меток, тестовые фразы {pct(m.test_before?.accuracy)} →
      {" "}{pct(m.test.accuracy)}, мои команды {pct(m.real_before?.accuracy)} → {pct(m.real?.accuracy)}.</p>
  ) : (
    <p className="outcome bad">✕ Версия {v.id} отклонена: {reasons.join("; ")}. Метки остались —
      проверьте исправления и отмените ошибочные.</p>
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
      setError(`Не сохранилось: ${e instanceof Error ? e.message : e}`);
      return false;
    }
  }
  const shown = label ?? { intent: c.intent, slots: c.slots };

  return (
    <li className={`command ${label ? `labelled ${label.verdict}` : ""}`}>
      <div className="said">
        <span className="when">{time(c.heard_at)}</span>
        <span className="text">«{c.text}»</span>
      </div>
      <div className="parse">
        {label?.verdict === "corrected" && (
          <span className="was">{describe(c.intent, c.slots)} →</span>
        )}
        <span className="intent">{describe(shown.intent, shown.slots)}</span>
        {!label && (
          <span className="muted">
            {Math.round(c.confidence * 100)}%{c.intent === "out_of_scope" && c.best_guess
              ? `, похоже на «${intentName(c.best_guess)}»` : ""} · v{c.model}
          </span>
        )}
      </div>
      <div className="row-actions">
        {!label && !editing && (
          <>
            <button onClick={() => run(() => onLabel(c.intent, c.slots))}>✓ Верно</button>
            <button onClick={() => setEditing(true)}>✎ Исправить</button>
          </>
        )}
        {label && (label.version === null ? (
          <>
            <span className="muted">{label.verdict === "confirmed" ? "✓ верно" : "✎ исправлено"}</span>
            <button onClick={() => run(onUndo)}>Отменить</button>
          </>
        ) : (
          <span className="muted">в версии {label.version}</span>
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

function describe(intent: string, slots: Slots) {
  const value = Object.values(slots)[0];
  return value ? `${intentName(intent)} «${value}»` : intentName(intent);
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
        Что это было
        <select value={intent} onChange={(e) => setIntent(e.target.value)} autoFocus>
          {Object.keys(intents).map((i) => <option key={i} value={i}>{intentName(i)}</option>)}
        </select>
      </label>
      {slot && (
        <label>
          {SLOT_LABELS[slot] ?? slot} — слова из команды
          <input value={value} onChange={(e) => setValue(e.target.value)} required
                 placeholder={c.text} />
        </label>
      )}
      <div className="actions">
        <button type="button" onClick={onCancel}>Отмена</button>
        <button type="submit" className="primary">Сохранить</button>
      </div>
    </form>
  );
}

function History({ versions }: { versions: Version[] }) {
  return (
    <div className="card">
      <h3>История переобучений</h3>
      <table className="history">
        <thead>
          <tr><th>Версия</th><th>Когда</th><th>Итог</th><th>Новых меток</th><th>Тестовые фразы</th><th>Мои команды</th></tr>
        </thead>
        <tbody>
          {[...versions].reverse().map((v) => (
            <tr key={v.id} className={v.accepted ? "" : "rejected"}>
              <td>v{v.id}</td>
              <td>{time(v.created_at)}</td>
              <td>{v.id === 0 ? "исходная" : v.accepted ? "принята" : "отклонена"}</td>
              <td>{v.id === 0 ? "—" : v.new_labels}</td>
              <td>{v.metrics.test_before ? `${pct(v.metrics.test_before.accuracy)} → ` : ""}{pct(v.metrics.test.accuracy)}</td>
              <td>{v.metrics.real_before ? `${pct(v.metrics.real_before.accuracy)} → ${pct(v.metrics.real?.accuracy)}` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">«Мои команды» — новые метки этой версии: до — точность прежней модели, после — новой,
        проверенной на них вне обучения (по фолдам). Тестовые фразы в обучение не попадают никогда.</p>
    </div>
  );
}
