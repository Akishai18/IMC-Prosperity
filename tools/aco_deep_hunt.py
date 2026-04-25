"""Deeper ACO alpha hunt. What we've learned:
- L1 OBI: +0.57 corr
- L1+L2 OBI: +0.62 corr
- Momentum (5-tick): -0.47 corr (mean reverting)
- L3 OBI: -0.55 corr (opposite sign!)
- Microprice: +0.50 corr

New questions to answer before coding:
1. Is deviation (|mid - FV|) itself predictive of reversion strength?
2. Is spread regime (narrow vs wide) a signal?
3. Does stacked z-score (mid-FV normalized by std) give better threshold?
4. Is there momentum AT L2 (L1+L2 mid changes)?
5. How often do take conditions trigger? What's the win rate per trigger?
6. Is L1+L2+L3 net (with L3 negative) better than L1+L2?
7. Volatility clustering — do high-vol periods predict high-vol?
"""
import pandas as pd
import numpy as np
from tools.backtester_harness import KEVIN_BACKTESTER_DIR

BASE_RESOURCES = KEVIN_BACKTESTER_DIR / 'prosperity4bt' / 'resources'

def load(rnd, day):
    p = pd.read_csv(BASE_RESOURCES / f'round{rnd}' / f'prices_round_{rnd}_day_{day}.csv', sep=';')
    t = pd.read_csv(BASE_RESOURCES / f'round{rnd}' / f'trades_round_{rnd}_day_{day}.csv', sep=';')
    p = p[p['product'] == 'ASH_COATED_OSMIUM'].copy().reset_index(drop=True)
    t = t[t['symbol'] == 'ASH_COATED_OSMIUM'].copy().reset_index(drop=True)
    return p, t

def prep(p):
    p = p.dropna(subset=['bid_price_1','ask_price_1','bid_volume_1','ask_volume_1']).reset_index(drop=True)
    bb, ba = p['bid_price_1'], p['ask_price_1']
    bv1, av1 = p['bid_volume_1'].fillna(0), p['ask_volume_1'].fillna(0)
    bv2, av2 = p['bid_volume_2'].fillna(0), p['ask_volume_2'].fillna(0)
    bv3, av3 = p['bid_volume_3'].fillna(0), p['ask_volume_3'].fillna(0)
    mid = (bb + ba) / 2
    spread = ba - bb
    obi1 = (bv1 - av1) / (bv1 + av1).replace(0, np.nan)
    obi12 = (bv1+bv2 - av1-av2) / (bv1+bv2+av1+av2).replace(0, np.nan)
    obi123 = (bv1+bv2+bv3 - av1-av2-av3) / (bv1+bv2+bv3+av1+av2+av3).replace(0, np.nan)
    # L3 negated
    obi_smart = (bv1+bv2 - av1-av2 - (bv3 - av3)) / (bv1+bv2+bv3+av1+av2+av3).replace(0, np.nan)
    dev = mid - 10000
    return p, bb, ba, bv1, av1, mid, spread, obi1, obi12, obi123, obi_smart, dev


print("=" * 90)
print("  Q1: Is |deviation| predictive of ABSOLUTE reversion speed?")
print("=" * 90)
for rnd, day in [(2,-1), (2,0)]:
    try: p, t = load(rnd, day)
    except FileNotFoundError: continue
    p, bb, ba, bv1, av1, mid, spread, obi1, obi12, obi123, obi_smart, dev = prep(p)
    ret = mid.diff().shift(-1)
    # Buckets of dev
    for lo, hi in [(-1,1),(-3,-1),(-5,-3),(-10,-5),(1,3),(3,5),(5,10)]:
        mask = (dev >= lo) & (dev < hi)
        if mask.sum() > 20:
            print(f"  R{rnd}d{day} dev∈[{lo:>3},{hi:>3}]: n={mask.sum():>5}, mean_next_ret={ret[mask].mean():+.3f}, std={ret[mask].std():.2f}")
    print()


print("=" * 90)
print("  Q2: Spread regime — does narrow spread predict different dynamics?")
print("=" * 90)
for rnd, day in [(2,-1), (2,0)]:
    try: p, t = load(rnd, day)
    except FileNotFoundError: continue
    p, bb, ba, bv1, av1, mid, spread, obi1, obi12, obi123, obi_smart, dev = prep(p)
    ret = mid.diff().shift(-1)
    print(f"  R{rnd}d{day}:")
    for s_val in sorted(spread.unique()):
        mask = spread == s_val
        if mask.sum() > 30:
            print(f"    spread={int(s_val):>2}: n={mask.sum():>5}, mean_dev={dev[mask].mean():+.2f}, "
                  f"std_dev={dev[mask].std():.2f}, |ret|={ret[mask].abs().mean():.2f}, OBI1={obi1[mask].mean():+.3f}")
    print()


print("=" * 90)
print("  Q3: L3-adjusted OBI vs L1+L2 — is 'smart' better?")
print("=" * 90)
for rnd, day in [(2,-1), (2,0), (1,-1), (1,0)]:
    try: p, t = load(rnd, day)
    except FileNotFoundError: continue
    p, bb, ba, bv1, av1, mid, spread, obi1, obi12, obi123, obi_smart, dev = prep(p)
    ret = mid.diff().shift(-1)
    c1 = obi1.corr(ret)
    c12 = obi12.corr(ret)
    c123 = obi123.corr(ret)
    csmart = obi_smart.corr(ret)
    print(f"  R{rnd}d{day}: L1={c1:+.3f}, L1+L2={c12:+.3f}, L1+L2+L3={c123:+.3f}, SMART(L1+L2-L3)={csmart:+.3f}")


print()
print("=" * 90)
print("  Q4: Combined signal regression — best linear model")
print("=" * 90)

for rnd, day in [(2,-1), (2,0)]:
    try: p, t = load(rnd, day)
    except FileNotFoundError: continue
    p, bb, ba, bv1, av1, mid, spread, obi1, obi12, obi123, obi_smart, dev = prep(p)
    ret = mid.diff().shift(-1)
    mom5 = mid.diff(5)
    mom3 = mid.diff(3)
    df = pd.DataFrame({
        'dev': dev, 'obi1': obi1, 'obi12': obi12, 'obi_smart': obi_smart,
        'mom3': mom3, 'mom5': mom5, 'spread': spread, 'ret': ret
    }).dropna()
    X_cols = ['dev', 'obi_smart', 'mom5']
    X = df[X_cols].values
    Xa = np.column_stack([X, np.ones(len(X))])
    y = df['ret'].values
    coef, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    pred = Xa @ coef
    r2 = np.corrcoef(pred, y)[0,1]**2
    print(f"  R{rnd}d{day}: R² = {r2:.4f}")
    for c, cf in zip(X_cols + ['intercept'], coef):
        print(f"    {c}: coef={cf:+.5f}")
    for c in X_cols:
        ru = df[c].corr(df['ret'])**2
        print(f"    univariate R²({c})={ru:.4f}")
    print()


print()
print("=" * 90)
print("  Q5: Volatility regime — does high |next_ret| cluster?")
print("=" * 90)
for rnd, day in [(2,-1),(2,0)]:
    try: p, t = load(rnd, day)
    except FileNotFoundError: continue
    p, bb, ba, bv1, av1, mid, spread, obi1, obi12, obi123, obi_smart, dev = prep(p)
    ret = mid.diff()
    # vol = rolling abs(ret)
    for w in [5, 10, 20]:
        vol = ret.abs().rolling(w).mean()
        fwd_vol = ret.abs().shift(-1)
        c = vol.corr(fwd_vol)
        print(f"  R{rnd}d{day} vol_w={w}: corr(rolling_vol, |next_ret|) = {c:+.3f}")
    print()


print()
print("=" * 90)
print("  Q6: Take condition profitability — when does (ba <= FV-k) actually pay?")
print("=" * 90)
for rnd, day in [(2,-1),(2,0)]:
    try: p, t = load(rnd, day)
    except FileNotFoundError: continue
    p, bb, ba, bv1, av1, mid, spread, obi1, obi12, obi123, obi_smart, dev = prep(p)
    ret = mid.diff().shift(-1)
    # "Buy at ba" then expected exit at mid (later)
    # How profitable is taking at ba when dev < -k?
    for k in [2, 3, 4, 5, 6, 7]:
        buy_cond = ba <= 10000 - k  # dev < -k
        if buy_cond.sum() > 30:
            # Forward P&L: buy at ba, hold N ticks, exit at mid
            for horizon in [5, 10, 20]:
                exit_mid = mid.shift(-horizon)
                pnl = exit_mid - ba
                avg_pnl = pnl[buy_cond].mean()
                print(f"  R{rnd}d{day} buy at ba when dev<-{k}: n={buy_cond.sum()}, avg_pnl_h{horizon}={avg_pnl:.2f}")
    print()


print()
print("=" * 90)
print("  Q7: Signal-weighted take size — does confidence matter?")
print("=" * 90)
for rnd, day in [(2,-1),(2,0)]:
    try: p, t = load(rnd, day)
    except FileNotFoundError: continue
    p, bb, ba, bv1, av1, mid, spread, obi1, obi12, obi123, obi_smart, dev = prep(p)
    ret = mid.diff().shift(-1)
    mom5 = mid.diff(5)
    # Combined signal: dev (negative = buy) + obi12 (positive = up) + momentum (negative = down coming)
    # For buy decision: want predicted return to be POSITIVE
    # predicted_ret ≈ -0.1*dev + 2*obi12 - 0.1*mom5  (rough weights)
    pred = -0.15*dev + 2.0*obi12 - 0.1*mom5
    # Quartile buckets
    for lo, hi in [(pred.quantile(0.95), 10), (pred.quantile(0.9), pred.quantile(0.95)),
                   (pred.quantile(0.75), pred.quantile(0.9)), (pred.quantile(0.5), pred.quantile(0.75))]:
        mask = (pred >= lo) & (pred < hi) if hi != 10 else pred >= lo
        if mask.sum() > 20:
            print(f"  R{rnd}d{day} pred∈[{lo:.2f},{hi if hi != 10 else '∞':.2f}]: "
                  f"n={mask.sum()}, mean_ret={ret[mask].mean():+.3f}")
    print()
