import Link from "next/link";

const navItems = [
  { href: "/", label: "Today's Picks" },
  { href: "/performance", label: "Performance" },
  { href: "/about", label: "About" },
];

export function SiteHeader() {
  return (
    <header className="border-b border-[var(--card-border)] bg-[var(--card)]/80 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <Link href="/" className="group">
          <p className="text-xs uppercase tracking-[0.2em] text-[var(--muted)]">
            Model Generated
          </p>
          <h1 className="text-lg font-semibold text-white group-hover:text-[var(--accent)]">
            Sports Picks Blog
          </h1>
        </Link>
        <nav className="flex items-center gap-4 sm:gap-6">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="text-sm text-[var(--muted)] transition hover:text-white"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
