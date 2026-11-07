"""
Basic pairs trading, built from the ideas we've covered.
Run:  python3 basic_pairs.py
Then change the numbers in the SETTINGS block and rerun to see what happens.
"""
import urllib.request
import numpy as np

# Use the operating system's trusted certificates (fixes SSL errors on some Macs,
# VPNs and university networks). Install with: pip3 install truststore
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# ---------------- SETTINGS (play with these) ----------------
SYMBOL_A = "KO"        # stock A
SYMBOL_B = "PEP"       # stock B
WINDOW = 60            # days used for beta, average and std dev of the spread
ENTRY_Z = 2.0          # open a trade when |z| is above this
EXIT_Z = 0.5           # close the trade when |z| falls below this
COST_BPS = 3.0         # trading cost per $ traded, in basis points (3 = 0.03%)
# -------------------------------------------------------------


def _download_stooq(sym):
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}.us&i=d"
    text = urllib.request.urlopen(url, timeout=15).read().decode()
    rows = [r.split(",") for r in text.strip().splitlines()[1:]]
    prices = {r[0]: float(r[4]) for r in rows if len(r) > 4 and r[4]}
    if len(prices) < 300:
        raise ValueError(f"Stooq returned no usable data for {sym}: {text[:100]!r}")
    return prices


def _download_yahoo(sym):
    import json, datetime
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?range=10y&interval=1d&events=history")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    res = data["chart"]["result"][0]
    stamps = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    prices = {datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d"): float(c)
              for t, c in zip(stamps, closes) if c is not None}
    if len(prices) < 300:
        raise ValueError(f"Yahoo returned too little data for {sym}")
    return prices


def load_prices():
    """Download daily closes (Stooq, then Yahoo). Fall back to fake data if both fail."""
    for name, fetch in (("Stooq", _download_stooq), ("Yahoo", _download_yahoo)):
        try:
            pa, pb = fetch(SYMBOL_A), fetch(SYMBOL_B)
            dates = sorted(set(pa) & set(pb))
            if len(dates) < 300:
                raise ValueError("not enough overlapping dates")
            a = np.array([pa[d] for d in dates])
            b = np.array([pb[d] for d in dates])
            return dates, a, b, f"REAL ({name})"
        except Exception as exc:
            print(f"[{name} download failed: {type(exc).__name__}: {exc}]")

    print("\n*** USING FAKE DATA: real prices could not be downloaded. ***")
    print("*** Results below say nothing about KO/PEP. ***\n")
    # Fake pair: B is a random walk, A follows B plus a gap that snaps back (the "leash")
    rng = np.random.default_rng(1)
    n = 1500
    log_b = np.cumsum(rng.normal(0.0003, 0.012, n)) + np.log(100)
    gap = np.zeros(n)
    for t in range(1, n):
        gap[t] = 0.95 * gap[t - 1] + rng.normal(0, 0.01)  # pulled back toward 0
    log_a = 0.8 * log_b + gap + 0.5
    dates = [str(i) for i in range(n)]
    return dates, np.exp(log_a), np.exp(log_b), "FAKE"


def run():
    dates, price_a, price_b, source = load_prices()
    log_a, log_b = np.log(price_a), np.log(price_b)   # logs = percentage-scale prices
    n = len(log_a)

    position = 0        # +1 = long A / short B, -1 = short A / long B, 0 = flat
    returns = np.zeros(n)
    trades = []
    prev_weights = np.zeros(2)
    zs = np.full(n, np.nan)      # z-score each day (for the plot)
    pos_hist = np.zeros(n)       # position each day (for the plot)

    for t in range(WINDOW, n):
        # --- 1. beta: slope of the best-fit line of log A against log B (last WINDOW days) ---
        ya = log_a[t - WINDOW:t]
        xb = log_b[t - WINDOW:t]
        beta = np.polyfit(xb, ya, 1)[0]

        # --- 2. spread: A minus beta * B, for each day in the window and for today ---
        window_spread = ya - beta * xb
        spread_today = log_a[t] - beta * log_b[t]

        # --- 3. z-score: how stretched is today's spread compared with its recent normal? ---
        z = (spread_today - window_spread.mean()) / window_spread.std()
        zs[t] = z

        # --- 4. today's return comes from the position we already held (decided yesterday) ---
        ret_a = log_a[t] - log_a[t - 1]
        ret_b = log_b[t] - log_b[t - 1]
        gross = 1 + abs(beta)
        weights = np.array([position / gross, -position * beta / gross])  # $ in A, $ in B
        returns[t] = prev_weights[0] * ret_a + prev_weights[1] * ret_b

        # --- 5. decide the position for tomorrow ---
        new_position = position
        if position == 0:
            if z > ENTRY_Z:
                new_position = -1    # spread high: A expensive -> short A, long B
            elif z < -ENTRY_Z:
                new_position = +1    # spread low: A cheap -> long A, short B
        elif abs(z) < EXIT_Z:
            new_position = 0         # gap has closed -> exit

        # --- 6. pay trading costs on whatever we changed ---
        new_weights = np.array([new_position / gross, -new_position * beta / gross])
        turnover = np.abs(new_weights - weights * (position != 0) * 0 - prev_weights).sum()
        if new_position != position:
            returns[t] -= turnover * COST_BPS / 10000
            trades.append((dates[t], "ENTER" if position == 0 else "EXIT", new_position, round(z, 2)))

        position = new_position
        pos_hist[t] = position
        prev_weights = new_weights

    equity = np.cumprod(1 + returns)
    active = returns[WINDOW:]
    ann_ret = active.mean() * 252
    ann_vol = active.std() * np.sqrt(252)
    drawdown = (equity / np.maximum.accumulate(equity) - 1).min()

    print(f"Data: {source} - {SYMBOL_A}/{SYMBOL_B}, {n} days")
    print(f"Trades (enter+exit): {len(trades)}")
    print(f"Annual return: {ann_ret:.2%}   Annual vol: {ann_vol:.2%}")
    print(f"Sharpe: {ann_ret / ann_vol if ann_vol else 0:.2f}   Max drawdown: {drawdown:.2%}")
    print(f"Final equity (start = 1.00): {equity[-1]:.3f}")
    print("First few trades:")
    for tr in trades[:8]:
        print("  ", tr)

    # ---- worst drawdown: when did it start and end? ----
    peak_curve = np.maximum.accumulate(equity)
    dd = equity / peak_curve - 1
    trough = int(np.argmin(dd))
    peak = int(np.argmax(equity[:trough + 1]))
    print(f"Worst drawdown {dd[trough]:.1%}: peak {dates[peak]} -> trough {dates[trough]}")

    make_plot(dates, log_a, log_b, zs, pos_hist, equity, dd, source, WINDOW)


def make_plot(dates, log_a, log_b, zs, pos_hist, equity, dd, source, start):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to get the chart: pip3 install matplotlib")
        return

    if "-" in dates[0]:
        x = np.array(dates, dtype="datetime64[D]")
    else:
        x = np.arange(len(dates))   # fake data has no real dates

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # Panel 1: how far each stock has moved since the start (in %)
    ax[0].plot(x, (np.exp(log_a - log_a[0]) - 1) * 100, label=SYMBOL_A)
    ax[0].plot(x, (np.exp(log_b - log_b[0]) - 1) * 100, label=SYMBOL_B)
    ax[0].set_title("1. Price change since start (%)  - do the two stocks stay together?")
    ax[0].legend()

    # Panel 2: the z-score, with the trading rules drawn on it
    ax[1].plot(x, zs, linewidth=0.8, color="black")
    for level, style in ((ENTRY_Z, "--"), (-ENTRY_Z, "--"), (EXIT_Z, ":"), (-EXIT_Z, ":")):
        ax[1].axhline(level, linestyle=style, color="grey")
    in_trade = pos_hist != 0
    ax[1].fill_between(x, -6, 6, where=in_trade, alpha=0.15, color="orange", label="in a trade")
    ax[1].set_ylim(-6, 6)
    ax[1].set_title("2. Z-score of the gap (dashed = entry, dotted = exit, orange = we hold a trade)")
    ax[1].legend(loc="upper right")

    # Panel 3: equity and drawdown
    ax[2].plot(x, equity, color="green")
    ax[2].set_ylabel("equity (start = 1)")
    ax2 = ax[2].twinx()
    ax2.fill_between(x, dd * 100, 0, color="red", alpha=0.25)
    ax2.set_ylabel("drawdown (%)")
    ax[2].set_title("3. Strategy equity (green) and drawdown (red)")

    fig.suptitle(f"{SYMBOL_A}/{SYMBOL_B} pairs trading - {source} data", fontsize=13)
    fig.tight_layout()
    fig.savefig("basic_chart.png", dpi=120)
    print("Saved chart: basic_chart.png")


if __name__ == "__main__":
    run()