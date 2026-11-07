"""
Basic pairs trading, built from the ideas we've covered.
Run:  python3 basic_pairs.py
Then change the numbers in the SETTINGS block and rerun to see what happens.
"""
import urllib.request
import numpy as np

# ---------------- SETTINGS (play with these) ----------------
SYMBOL_A = "KO"        # stock A
SYMBOL_B = "PEP"       # stock B
WINDOW = 60            # days used for beta, average and std dev of the spread
ENTRY_Z = 2.0          # open a trade when |z| is above this
EXIT_Z = 0.5           # close the trade when |z| falls below this
COST_BPS = 3.0         # trading cost per $ traded, in basis points (3 = 0.03%)
# -------------------------------------------------------------


def load_prices():
    """Try to download daily closes from Stooq. Fall back to fake data if blocked."""
    try:
        out = []
        for sym in (SYMBOL_A, SYMBOL_B):
            url = f"https://stooq.com/q/d/l/?s={sym.lower()}.us&i=d"
            text = urllib.request.urlopen(url, timeout=10).read().decode()
            rows = [r.split(",") for r in text.strip().splitlines()[1:]]
            out.append({r[0]: float(r[4]) for r in rows if len(r) > 4 and r[4]})
        dates = sorted(set(out[0]) & set(out[1]))
        if len(dates) < 300:
            raise ValueError("not enough data")
        a = np.array([out[0][d] for d in dates])
        b = np.array([out[1][d] for d in dates])
        return dates, a, b, "real"
    except Exception:
        # Fake pair: B is a random walk, A follows B plus a gap that snaps back (the "leash")
        rng = np.random.default_rng(1)
        n = 1500
        log_b = np.cumsum(rng.normal(0.0003, 0.012, n)) + np.log(100)
        gap = np.zeros(n)
        for t in range(1, n):
            gap[t] = 0.95 * gap[t - 1] + rng.normal(0, 0.01)  # pulled back toward 0
        log_a = 0.8 * log_b + gap + 0.5
        dates = [str(i) for i in range(n)]
        return dates, np.exp(log_a), np.exp(log_b), "synthetic"


def run():
    dates, price_a, price_b, source = load_prices()
    log_a, log_b = np.log(price_a), np.log(price_b)   # logs = percentage-scale prices
    n = len(log_a)

    position = 0        # +1 = long A / short B, -1 = short A / long B, 0 = flat
    returns = np.zeros(n)
    trades = []
    prev_weights = np.zeros(2)

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
        prev_weights = new_weights

    equity = np.cumprod(1 + returns)
    active = returns[WINDOW:]
    ann_ret = active.mean() * 252
    ann_vol = active.std() * np.sqrt(252)
    drawdown = (equity / np.maximum.accumulate(equity) - 1).min()

    print(f"Data: {source} {SYMBOL_A}/{SYMBOL_B}, {n} days")
    print(f"Trades (enter+exit): {len(trades)}")
    print(f"Annual return: {ann_ret:.2%}   Annual vol: {ann_vol:.2%}")
    print(f"Sharpe: {ann_ret / ann_vol if ann_vol else 0:.2f}   Max drawdown: {drawdown:.2%}")
    print(f"Final equity (start = 1.00): {equity[-1]:.3f}")
    print("First few trades:")
    for tr in trades[:8]:
        print("  ", tr)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4))
        plt.plot(equity)
        plt.title(f"Equity curve ({source} data)")
        plt.savefig("basic_equity.png", dpi=120, bbox_inches="tight")
    except ImportError:
        pass


if __name__ == "__main__":
    run()