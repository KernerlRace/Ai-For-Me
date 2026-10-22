import { useEffect, useState } from "react";

type Mode = "normal" | "coding" | "gaming" | "meeting";

const LABELS: Record<Mode, string> = {
  normal: "Обычный режим",
  coding: "Кодинг",
  gaming: "Игровой режим",
  meeting: "Встреча",
};

/** The autopilot's current mode, refreshed every 10 s; hidden until the assistant has chosen one. */
export function ModeBadge() {
  const [mode, setMode] = useState<{ mode: Mode | null; reason?: string }>({ mode: null });

  useEffect(() => {
    let alive = true;
    const load = () =>
      fetch("/api/mode")
        .then((r) => r.json())
        .then((m) => alive && setMode(m))
        .catch(() => {});
    load();
    const timer = window.setInterval(load, 10_000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  if (!mode.mode) return null;
  return (
    <span className={`mode mode-${mode.mode}`} title={mode.reason}>
      ◆ {LABELS[mode.mode] ?? mode.mode}
    </span>
  );
}
