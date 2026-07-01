import Link from "next/link";

export default function NotFound() {
  return (
    <div className="space-y-4">
      <h1 className="text-3xl font-bold text-white">Page not found</h1>
      <p className="text-[var(--muted)]">
        That pick date does not exist in the archive.
      </p>
      <Link href="/" className="text-[var(--accent)] hover:underline">
        ← Back to today&apos;s picks
      </Link>
    </div>
  );
}
