import { useState } from "react";

export interface AccuracyPoint {
  version: number;
  test: number | null;                         // share of the test phrases right
  live: { correct: number; labelled: number } | null;   // the labelled commands this version served
}

const SERIES = [
  { key: "test", name: "Тестовые фразы", color: "var(--series-1)" },
  { key: "live", name: "Мои команды", color: "var(--series-2)" },
] as const;

const W = 640, H = 220, PAD = { top: 14, right: 64, bottom: 28, left: 40 };

function value(p: AccuracyPoint, key: "test" | "live"): number | null {
  if (key === "test") return p.test;
  return p.live && p.live.labelled > 0 ? p.live.correct / p.live.labelled : null;
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

/** Centred over the version, but kept inside the chart near its edges. */
function tooltipAt(share: number): React.CSSProperties {
  const shift = share < 0.25 ? "0" : share > 0.75 ? "-100%" : "-50%";
  return { left: `${share * 100}%`, transform: `translateX(${shift})` };
}

/** Accuracy by model version, two lines on one percentage axis; hover shows both values at a version. */
export function AccuracyChart({ points }: { points: AccuracyPoint[] }) {
  const [hover, setHover] = useState<number | null>(null);

  const values = points.flatMap((p) => SERIES.map((s) => value(p, s.key))).filter((v): v is number => v !== null);
  if (!values.length) return <p className="muted">График появится, когда будут метки или первое переобучение.</p>;

  const lo = Math.max(0, Math.floor((Math.min(...values) - 0.05) * 10) / 10);
  const x = (i: number) => PAD.left + (points.length === 1 ? (W - PAD.left - PAD.right) / 2
    : (i / (points.length - 1)) * (W - PAD.left - PAD.right));
  const y = (v: number) => PAD.top + (1 - (v - lo) / (1 - lo)) * (H - PAD.top - PAD.bottom);
  const ticks = Array.from({ length: Math.round((1 - lo) * 10) + 1 }, (_, i) => lo + i / 10)
    .filter((_, i, all) => all.length <= 6 || i % 2 === 0);

  // end labels: beside the last point of each line (a line may stop before the last version, when no
  // commands of the newest one are labelled yet), nudged apart when two would overlap
  const ends = SERIES.map((s) => {
    const i = points.map((p) => value(p, s.key) !== null).lastIndexOf(true);
    return i < 0 ? null : { ...s, v: value(points[i], s.key)!, x: x(i) + 9, y: y(value(points[i], s.key)!) };
  }).filter((e) => e !== null);
  if (ends.length === 2 && ends[0].x === ends[1].x && Math.abs(ends[0].y - ends[1].y) < 14) {
    const [a, b] = ends[0].y <= ends[1].y ? [ends[0], ends[1]] : [ends[1], ends[0]];
    const mid = (a.y + b.y) / 2;
    a.y = mid - 7;
    b.y = mid + 7;
  }

  function onMove(e: React.PointerEvent<SVGRectElement>) {
    const rect = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0;
    points.forEach((_, i) => { if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i; });
    setHover(best);
  }

  const h = hover === null ? null : points[hover];
  return (
    <div className="chart">
      <div className="legend">
        {SERIES.map((s) => (
          <span key={s.key}><i style={{ background: s.color }} />{s.name}</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Точность по версиям модели">
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid" x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} />
            <text className="tick" x={PAD.left - 6} y={y(t)} dy="0.32em" textAnchor="end">{Math.round(t * 100)}%</text>
          </g>
        ))}
        {points.map((p, i) => (
          <text key={p.version} className="tick" x={x(i)} y={H - 8} textAnchor="middle">v{p.version}</text>
        ))}
        {h && <line className="crosshair" x1={x(hover!)} x2={x(hover!)} y1={PAD.top} y2={H - PAD.bottom} />}
        {SERIES.map((s) => {
          const pts = points.map((p, i) => [i, value(p, s.key)] as const).filter(([, v]) => v !== null);
          return (
            <g key={s.key}>
              <polyline fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round"
                        points={pts.map(([i, v]) => `${x(i)},${y(v!)}`).join(" ")} />
              {pts.map(([i, v]) => (
                <circle key={i} cx={x(i)} cy={y(v!)} r={hover === i ? 5 : 4} fill={s.color}
                        stroke="var(--surface)" strokeWidth={2} />
              ))}
            </g>
          );
        })}
        {ends.map((e) => (
          <text key={e.key} className="end-label" x={e.x} y={e.y} dy="0.32em">{pct(e.v)}</text>
        ))}
        <rect x={PAD.left - 10} y={0} width={W - PAD.left - PAD.right + 20} height={H} fill="transparent"
              onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {h && (
        <div className="tooltip" style={tooltipAt(x(hover!) / W)}>
          <b>Версия {h.version}</b>
          {SERIES.map((s) => {
            const v = value(h, s.key);
            return (
              <div key={s.key}>
                <i style={{ background: s.color }} />
                <strong>{v === null ? "—" : pct(v)}</strong> {s.name}
                {s.key === "live" && h.live ? <span className="muted"> ({h.live.correct} из {h.live.labelled})</span> : null}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
