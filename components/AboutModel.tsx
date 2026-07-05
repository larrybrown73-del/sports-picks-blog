export function AboutModel() {
  return (
    <div className="space-y-8 rounded-xl border border-[var(--card-border)] bg-[var(--card)] p-6 shadow-md">
      <section>
        <h2 className="mb-4 border-b border-[var(--card-border)] pb-2 text-2xl font-bold text-white">
          How the Model Works
        </h2>
        <p className="mb-6 leading-relaxed text-[var(--muted)]">
          This system predicts MLB game winners and player props by analyzing deep historical
          performance metrics, tracking baseline offensive and defensive strengths like home and
          away run distributions. To keep our predictions realistic, the engine uses advanced
          logarithmic scaling to flatten out extreme, high-scoring blowout games so flukes
          don&apos;t warp our baseline data. Finally, the system translates these refined run
          environments into exact win probabilities, compares them against real-world sportsbook
          lines to find market mispricings, and uses a disciplined Quarter-Kelly formula to
          calculate the exact optimal edge and wager sizing.
        </p>
      </section>

      <section>
        <h2 className="mb-4 border-b border-[var(--card-border)] pb-2 text-2xl font-bold text-white">
          Future Upgrades & Forecasting Plan
        </h2>
        <p className="mb-4 text-sm italic text-[var(--muted)]">
          We are actively testing adjustments to turn our current statistical edge into an even
          sharper predictive engine:
        </p>

        <div className="space-y-4">
          <div className="rounded-lg bg-[var(--background)] p-4">
            <h3 className="font-semibold text-[var(--accent)]">
              Phase 1: Real-Time Weather & Park Factors
            </h3>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Integrating live stadium analytics (temperature, wind speed/direction, and humidity).
              High-altitude or hot-weather environments drastically alter run production; factoring
              this in stops the model from over-adjusting to venue-specific blowouts.
            </p>
          </div>

          <div className="rounded-lg bg-[var(--background)] p-4">
            <h3 className="font-semibold text-[var(--accent)]">
              Phase 2: Bullpen Fatigue & Pitcher-Batter Matchups
            </h3>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Moving beyond team-level data to track live bullpen usage and pitch counts from
              consecutive days. This allows the model to forecast late-game pitching collapses and
              find massive closing value before the books adjust.
            </p>
          </div>

          <div className="rounded-lg bg-[var(--background)] p-4">
            <h3 className="font-semibold text-[var(--accent)]">
              Phase 3: Context-Aware Smart Scaling
            </h3>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Upgrading from a fixed scaling modifier to a dynamic model that automatically scales
              its &quot;diminishing returns&quot; filter based on specific division styles,
              defensive schemes, and league scoring environments.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
