import Link from "next/link";

import { getPickDates } from "@/lib/picks";

interface DateSidebarProps {
  activeDate?: string;
}

export function DateSidebar({ activeDate }: DateSidebarProps) {
  const dates = getPickDates();

  if (dates.length === 0) {
    return null;
  }

  return (
    <aside className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
        Pick Archive
      </h2>
      <ul className="space-y-2">
        {dates.map((date) => {
          const isActive = date === activeDate;
          return (
            <li key={date}>
              <Link
                href={`/picks/${date}`}
                className={`block rounded-lg px-3 py-2 text-sm transition ${
                  isActive
                    ? "bg-[var(--accent-muted)]/40 font-medium text-[var(--accent)]"
                    : "text-[var(--muted)] hover:bg-black/20 hover:text-white"
                }`}
              >
                {date}
              </Link>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
