"""
Basic pairs trading, built from the ideas we've covered.

Run:  python3 basic_pairs.py
It runs the strategy once for each stop level in STOP_VALUES, prints one comparison
table, and saves one chart per stop level. Change the SETTINGS block and rerun.
"""
import datetime
import json
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
STOP_VALUES = [None, 3.0, 4.0, 5.0]   # stop levels to compare. None = no stop.
                       # STOP: if |z| goes beyond this, we were wrong -> close at a loss and
                       # stay out until the gap is back to normal.
COST_BPS = 3.0         # trading cost per $ traded, in basis points (3 = 0.03%)
USE_ADJUSTED = True    # True = dividend-adjusted prices (dividends no longer look like
                       # sudden drops in the spread). False = raw closing prices.
# -------------------------------------------------------------

_CACHE = {}


def _download_stooq(sym):
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}.us&i=d"
    text = urllib.request.urlopen(url, timeout=15).read().decode()
    rows = [r.split(",") for r in text.strip().splitlines()[1:]]
    prices = {r[0]: float(r[4]) for r in rows if len(r) > 4 and r[4]}
    if len(prices) < 300:
        raise ValueError(f"Stooq returned no usable data for {sym}: {text[:100]!r}")
    return prices


def _download_yahoo(sym, adjusted):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?range=10y&interval=1d&events=history")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    res = data["chart"]["result"][0]
    stamps = res["timestamp"]
    if adjusted:
        try:
            closes = res["indicators"]["adjclose"][0]["adjclose"]   # dividend-adjusted
        except (KeyError, IndexError):
            raise ValueError(f"Yahoo response for {sym} has no adjusted close")
    else:
        closes = res["indicators"]["quote"][0]["close"]             # raw close
    prices = {datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d"): float(c)
              for t, c in zip(stamps, closes) if c is not None}
    if len(prices) < 300:
        raise ValueError(f"Yahoo returned too little data for {sym}")
    return prices


def load_prices():
    """Download daily prices (Yahoo, then Stooq). Fall back to fake data if both fail."""
    if "data" in _CACHE:
        return _CACHE["data"]

    if USE_ADJUSTED:
        sources = (("Yahoo, adjusted", lambda s: _download_yahoo(s, True)),
                   ("Stooq, UNADJUSTED", _download_stooq))
    else:
        sources = (("Yahoo, unadjusted", lambda s: _download_yahoo(s, False)),
                   ("Stooq, unadjusted", _download_stooq))

    for i, (name, fetch) in enumerate(sources):
        try:
            pa, pb = fetch(SYMBOL_A), fetch(SYMBOL_B)
            dates = sorted(set(pa) & set(pb))
            if len(dates) < 300:
                raise ValueError("not enough overlapping dates")
            a = np.array([pa[d] for d in dates])
            b = np.array([pb[d] for d in dates])
            if USE_ADJUSTED and i > 0:
                print("\n*** WARNING: adjusted prices were not available. Using UNADJUSTED prices. ***\n")
            _CACHE["data"] = (dates, a, b, f"REAL ({name})")
            return _CACHE["data"]
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
    _CACHE["data"] = (dates, np.exp(log_a), np.exp(log_b), "FAKE")
    return _CACHE["data"]


def simulate(data, stop_z):
    """Run the strategy once with the given stop level. Returns a dict of results."""
    dates, price_a, price_b, source = data
    log_a, log_b = np.log(price_a), np.log(price_b)   # logs = percentage-scale prices
    n = len(log_a)

    position = 0        # +1 = long A / short B, -1 = short A / long B, 0 = flat
    blocked = False     # True after a stop: no new trades until the gap is back to normal
    returns = np.zeros(n)
    trades = []         # (day index, date, label, new position, z)
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
        returns[t] = prev_weights[0] * ret_a + prev_weights[1] * ret_b

        # --- 5. decide the position for tomorrow ---
        new_position = position
        stopped_now = False
        too_extreme = stop_z is not None and abs(z) > stop_z   # gap is beyond the stop line

        # After a stop, stay out until the gap has come back to normal (|z| < EXIT_Z).
        if blocked and abs(z) < EXIT_Z:
            blocked = False

        if position == 0:
            # Only enter if the gap is stretched enough AND not so stretched that we'd
            # be stopped out straight away, and we're not waiting after a stop.
            if not blocked and not too_extreme:
                if z > ENTRY_Z:
                    new_position = -1    # spread high: A expensive -> short A, long B
                elif z < -ENTRY_Z:
                    new_position = +1    # spread low: A cheap -> long A, short B
        elif too_extreme:
            new_position = 0             # STOP: gap kept widening, we were wrong -> get out
            stopped_now = True
            blocked = True
        elif abs(z) < EXIT_Z:
            new_position = 0             # gap has closed -> exit

        # --- 6. pay trading costs on whatever we changed ---
        new_weights = np.array([new_position / gross, -new_position * beta / gross])
        if new_position != position:
            turnover = np.abs(new_weights - prev_weights).sum()
            returns[t] -= turnover * COST_BPS / 10000
            label = "ENTER" if position == 0 else ("STOP" if stopped_now else "EXIT")
            trades.append((t, dates[t], label, new_position, float(z)))

        position = new_position
        pos_hist[t] = position
        prev_weights = new_weights

    equity = np.cumprod(1 + returns)
    active = returns[WINDOW:]
    ann_ret = active.mean() * 252
    ann_vol = active.std() * np.sqrt(252)
    dd = equity / np.maximum.accumulate(equity) - 1
    trough = int(np.argmin(dd))
    peak = int(np.argmax(equity[:trough + 1]))
    return {
        "stop_z": stop_z, "trades": trades, "returns": returns, "equity": equity, "dd": dd,
        "zs": zs, "pos_hist": pos_hist,
        "n_trades": len(trades), "n_stops": sum(1 for tr in trades if tr[2] == "STOP"),
        "ann_ret": ann_ret, "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol else 0.0,
        "max_dd": dd.min(), "dd_peak": dates[peak], "dd_trough": dates[trough],
        "final_equity": equity[-1],
    }


def chart_name(stop_z):
    return "basic_chart_stop_off.png" if stop_z is None else f"basic_chart_stop{stop_z:g}.png"


def make_plot(data, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to get the charts: pip3 install matplotlib")
        return

    dates, price_a, price_b, source = data
    log_a, log_b = np.log(price_a), np.log(price_b)
    stop_z = res["stop_z"]
    zs, pos_hist, equity, dd = res["zs"], res["pos_hist"], res["equity"], res["dd"]

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
    if stop_z is not None:
        ax[1].axhline(stop_z, linestyle="-.", color="red", alpha=0.6, label=f"stop line (+/-{stop_z:g})")
        ax[1].axhline(-stop_z, linestyle="-.", color="red", alpha=0.6)
    in_trade = pos_hist != 0
    ax[1].fill_between(x, -6, 6, where=in_trade, alpha=0.15, color="orange", label="in a trade")
    stop_trades = [tr for tr in res["trades"] if tr[2] == "STOP"]
    if stop_trades:
        idx = [tr[0] for tr in stop_trades]
        ax[1].scatter(x[idx], np.clip([tr[4] for tr in stop_trades], -6, 6),
                      marker="x", color="red", s=70, zorder=5, label=f"stop triggered ({len(idx)})")
    ax[1].set_ylim(-6, 6)
    ax[1].set_title("2. Z-score of the gap (orange = we hold a trade, red X = stopped out)")
    ax[1].legend(loc="upper right")

    # Panel 3: equity and drawdown
    ax[2].plot(x, equity, color="green")
    ax[2].set_ylabel("equity (start = 1)")
    ax2 = ax[2].twinx()
    ax2.fill_between(x, dd * 100, 0, color="red", alpha=0.25)
    ax2.set_ylabel("drawdown (%)")
    ax[2].set_title(f"3. Strategy equity (green) and drawdown (red)  -  Sharpe {res['sharpe']:.2f}, "
                    f"worst drawdown {res['max_dd']:.1%}")

    stop_text = "no stop" if stop_z is None else f"stop {stop_z:g}"
    fig.suptitle(f"{SYMBOL_A}/{SYMBOL_B} pairs trading, {stop_text} - {source}", fontsize=13)
    fig.tight_layout()
    fig.savefig(chart_name(stop_z), dpi=120)
    plt.close(fig)


def main():
    data = load_prices()
    dates, _, _, source = data
    print(f"Data: {source} - {SYMBOL_A}/{SYMBOL_B}, {len(dates)} days")
    print(f"Settings: entry {ENTRY_Z}, exit {EXIT_Z}, window {WINDOW}, cost {COST_BPS} bps\n")

    results = [simulate(data, s) for s in STOP_VALUES]

    header = f"{'Stop':>6} {'Trades':>7} {'Stops':>6} {'Ann.ret':>8} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>8}  Worst drawdown (peak -> trough)"
    print(header)
    print("-" * len(header))
    for r in results:
        stop_text = "off" if r["stop_z"] is None else f"{r['stop_z']:g}"
        print(f"{stop_text:>6} {r['n_trades']:>7} {r['n_stops']:>6} {r['ann_ret']:>8.2%} {r['ann_vol']:>7.2%} "
              f"{r['sharpe']:>7.2f} {r['max_dd']:>8.2%}  {r['dd_peak']} -> {r['dd_trough']}")

    for r in results:
        make_plot(data, r)
    print("\nSaved charts: " + ", ".join(chart_name(s) for s in STOP_VALUES))


if __name__ == "__main__":
    main()
