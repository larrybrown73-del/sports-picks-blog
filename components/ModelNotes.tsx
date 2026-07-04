import Link from "next/link";

export function ModelNotes() {
  const notes = [
    {
      term: "Edge",
      detail: "Model probability minus market implied probability.",
    },
    {
      term: "Confidence",
      detail: "Internal 0–100 score from model edge and win probability — not a guarantee.",
    },
    {
      term: "Quarter-Kelly",
      detail: "Bankroll sizing formula (25% of full Kelly optimal stake).",
    },
    {
      term: "Props",
      detail: "Only shown when enough live market data is available.",
    },
  ];

  return (
    <section className="rounded-xl border border-[var(--card-border)] bg-[var(--card)] p-5">
      <h2 className="text-lg font-semibold text-white">Model Notes</h2>
      <ul className="mt-4 space-y-3 text-sm">
        {notes.map((note) => (
          <li key={note.term} className="leading-relaxed text-[var(--muted)]">
            <span className="font-medium text-white">{note.term}</span> — {note.detail}
          </li>
        ))}
      </ul>
      <Link href="/about" className="mt-4 inline-block text-sm text-[var(--accent)] hover:underline">
        Full methodology and disclaimer →
      </Link>
    </section>
  );
}
