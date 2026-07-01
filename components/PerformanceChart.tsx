import type { PerformanceSnapshot } from "@/lib/types";
import { parsePercent } from "@/lib/utils";

interface PerformanceChartProps {
  snapshots: PerformanceSnapshot[];
}

function buildPoints(
  snapshots: PerformanceSnapshot[],
  accessor: (snapshot: PerformanceSnapshot) => number,
): string {
  if (snapshots.length === 0) {
    return "";
  }

  const width = 640;
  const height = 220;
  const padding = 24;
  const values = snapshots.map(accessor);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  return snapshots
    .map((snapshot, index) => {
      const x =
        padding +
        (index / Math.max(snapshots.length - 1, 1)) * (width - padding * 2);
      const y =
        height -
        padding -
        ((accessor(snapshot) - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");
}

export function PerformanceChart({ snapshots }: PerformanceChartProps) {
  if (snapshots.length === 0) {
    return (
      <p className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] px-5 py-8 text-sm text-[var(--muted)]">
        No performance snapshots exported yet. Run `npm run sync-picks` to populate history.
      </p>
    );
  }

  const roiPoints = buildPoints(snapshots, (snapshot) =>
    parsePercent(snapshot.performance.roi),
  );
  const accuracyPoints = buildPoints(snapshots, (snapshot) =>
    parsePercent(snapshot.performance.accuracy),
  );

  return (
    <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] p-5">
      <h2 className="mb-4 text-lg font-semibold text-white">Track Record Trend</h2>
      <div className="overflow-x-auto">
        <svg viewBox="0 0 640 220" className="min-w-[640px] w-full">
          <rect x="0" y="0" width="640" height="220" fill="transparent" />
          <polyline
            fill="none"
            stroke="#22c55e"
            strokeWidth="3"
            points={roiPoints}
          />
          <polyline
            fill="none"
            stroke="#60a5fa"
            strokeWidth="3"
            points={accuracyPoints}
          />
        </svg>
      </div>
      <div className="mt-3 flex gap-6 text-sm text-[var(--muted)]">
        <span className="flex items-center gap-2">
          <span className="inline-block h-2 w-6 rounded bg-[var(--accent)]" />
          ROI
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block h-2 w-6 rounded bg-blue-400" />
          Accuracy
        </span>
      </div>
    </section>
  );
}
