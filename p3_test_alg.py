"""
P3 backtester compatibility version of our algorithm.
Maps our tutorial-round logic to Prosperity 3 product names:
  EMERALDS -> RAINFOREST_RESIN  (fixed FV = 10,000)
  TOMATOES -> KELP               (slow random walk, inside-wall MM)
  [NEW]    -> SQUID_INK          (Olivia informed-trader following)
  [NEW]    -> PICNIC_BASKET1/2   (ETF statistical arbitrage)

Use this file only for backtesting on P3 data.
Submit tutorial-round-alg.py for actual P4 submissions.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import jsonpickle

# ── Position limits ──────────────────────────────────────────────────────────
RESIN_LIMIT   = 50
KELP_LIMIT    = 50
SQUID_LIMIT   = 50
BASKET1_LIMIT = 60
BASKET2_LIMIT = 100

# ── Rainforest Resin ─────────────────────────────────────────────────────────
RESIN_FAIR    = 10000

# ── Kelp ─────────────────────────────────────────────────────────────────────
EMA_ALPHA     = 0.3

# ── Olivia (Squid Ink) ───────────────────────────────────────────────────────
OLIVIA_LOT_SIZE        = 15   # Olivia always trades exactly this many lots
OLIVIA_PRICE_TOLERANCE = 2    # price must be within this of the daily extreme

# ── Basket arb ───────────────────────────────────────────────────────────────
# Constituent weights for each basket
BASKET1_WEIGHTS = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
BASKET2_WEIGHTS = {"CROISSANTS": 4, "JAMS": 2}

# Entry threshold on the premium-adjusted spread.
# Basket1 raw spread: mean≈70, stdev≈79  → threshold ~1 stdev after premium removal
# Basket2 raw spread: mean≈58, stdev≈52  → threshold ~1 stdev after premium removal
BASKET1_THRESHOLD = 75
BASKET2_THRESHOLD = 50

# Slow EMA to track the persistent basket premium (mean ≠ 0).
# Half-life ≈ 1/alpha ticks. 0.002 → half-life ~500 ticks (adapts over half a day).
BASKET_PREMIUM_ALPHA = 0.002


class Trader:

    def bid(self) -> int:
        return 15

    def run(self, state: TradingState):
        saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        ema_kelp = saved.get("ema_kelp", None)

        result: Dict[str, List[Order]] = {}

        # ── Single-product strategies ─────────────────────────────────────────
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            pos = state.position.get(product, 0)

            if product == "RAINFOREST_RESIN":
                result[product] = self._trade_resin(order_depth, pos)

            elif product == "KELP":
                result[product], ema_kelp = self._trade_kelp(order_depth, pos, ema_kelp)

            elif product == "SQUID_INK":
                squid_trades = state.market_trades.get("SQUID_INK", [])
                result[product], saved = self._trade_squid(
                    order_depth, squid_trades, pos, saved, state.timestamp
                )

        # ── Basket strategies (need multiple products' prices) ────────────────
        if "PICNIC_BASKET1" in state.order_depths:
            orders, saved = self._trade_basket(
                "PICNIC_BASKET1", BASKET1_WEIGHTS, BASKET1_LIMIT, BASKET1_THRESHOLD,
                state.order_depths, state.position, saved,
            )
            if orders:
                result["PICNIC_BASKET1"] = orders

        if "PICNIC_BASKET2" in state.order_depths:
            orders, saved = self._trade_basket(
                "PICNIC_BASKET2", BASKET2_WEIGHTS, BASKET2_LIMIT, BASKET2_THRESHOLD,
                state.order_depths, state.position, saved,
            )
            if orders:
                result["PICNIC_BASKET2"] = orders

        saved["ema_kelp"] = ema_kelp
        return result, 0, jsonpickle.encode(saved)

    # ── RAINFOREST RESIN ──────────────────────────────────────────────────────

    def _trade_resin(self, order_depth: OrderDepth, pos: int) -> List[Order]:
        orders: List[Order] = []
        buy_cap  = RESIN_LIMIT - pos
        sell_cap = RESIN_LIMIT + pos

        for price in sorted(order_depth.sell_orders.keys()):
            if price > RESIN_FAIR or buy_cap <= 0:
                break
            vol = -order_depth.sell_orders[price]
            qty = min(vol, buy_cap)
            orders.append(Order("RAINFOREST_RESIN", price, qty))
            buy_cap -= qty

        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if price < RESIN_FAIR or sell_cap <= 0:
                break
            vol = order_depth.buy_orders[price]
            qty = min(vol, sell_cap)
            orders.append(Order("RAINFOREST_RESIN", price, -qty))
            sell_cap -= qty

        skew        = round(pos / RESIN_LIMIT * 2)
        passive_bid = max(RESIN_FAIR - 7 - skew, RESIN_FAIR - 8)
        passive_ask = min(RESIN_FAIR + 7 - skew, RESIN_FAIR + 8)

        if buy_cap  > 0: orders.append(Order("RAINFOREST_RESIN", passive_bid,  buy_cap))
        if sell_cap > 0: orders.append(Order("RAINFOREST_RESIN", passive_ask, -sell_cap))
        return orders

    # ── KELP ──────────────────────────────────────────────────────────────────

    def _trade_kelp(self, order_depth: OrderDepth, pos: int, ema: float):
        orders: List[Order] = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, ema

        wall_bid = max(order_depth.buy_orders.keys())
        wall_ask = min(order_depth.sell_orders.keys())
        mid      = (wall_bid + wall_ask) / 2.0
        ema      = EMA_ALPHA * mid + (1.0 - EMA_ALPHA) * ema if ema is not None else mid

        buy_cap  = KELP_LIMIT - pos
        sell_cap = KELP_LIMIT + pos

        for price in sorted(order_depth.sell_orders.keys()):
            if price > round(ema) or buy_cap <= 0: break
            vol = -order_depth.sell_orders[price]
            qty = min(vol, buy_cap)
            orders.append(Order("KELP", price, qty))
            buy_cap -= qty

        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if price < round(ema) or sell_cap <= 0: break
            vol = order_depth.buy_orders[price]
            qty = min(vol, sell_cap)
            orders.append(Order("KELP", price, -qty))
            sell_cap -= qty

        skew        = round(pos / KELP_LIMIT * 1)
        passive_bid = wall_bid + 1 - skew
        passive_ask = wall_ask - 1 - skew

        if passive_bid >= passive_ask:
            passive_bid = round(ema) - 1
            passive_ask = round(ema) + 1

        passive_bid = min(passive_bid, wall_bid)
        passive_ask = max(passive_ask, wall_ask)

        if buy_cap  > 0: orders.append(Order("KELP", passive_bid,  buy_cap))
        if sell_cap > 0: orders.append(Order("KELP", passive_ask, -sell_cap))
        return orders, ema

    # ── SQUID INK — Olivia informed-trader following ──────────────────────────

    def _trade_squid(self, order_depth, market_trades, pos, saved, timestamp):
        orders: List[Order] = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, saved

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid      = (best_bid + best_ask) / 2.0

        # Day reset: timestamp rolls back to 0 at start of each new day
        prev_ts = saved.get("squid_prev_ts", -1)
        if timestamp <= prev_ts and prev_ts != -1 and timestamp < 1000:
            saved["squid_daily_min"] = mid
            saved["squid_daily_max"] = mid
            saved["olivia_signal"]   = None
        saved["squid_prev_ts"] = timestamp

        daily_min: float       = saved.get("squid_daily_min", mid)
        daily_max: float       = saved.get("squid_daily_max", mid)
        signal: Optional[str]  = saved.get("olivia_signal", None)

        new_min = min(daily_min, mid)
        new_max = max(daily_max, mid)

        # Invalidate if price makes a new extreme that contradicts our position
        if signal == "LONG"  and new_min < daily_min: signal = None
        if signal == "SHORT" and new_max > daily_max: signal = None

        # Aggregate tick volume by side to handle Olivia's split orders
        tick_buys:  Dict[int, int] = {}
        tick_sells: Dict[int, int] = {}

        for trade in market_trades:
            buyer  = getattr(trade, 'buyer',  '')
            seller = getattr(trade, 'seller', '')

            if buyer == "Olivia":
                tick_buys[trade.price]  = tick_buys.get(trade.price, 0)  + trade.quantity
            elif seller == "Olivia":
                tick_sells[trade.price] = tick_sells.get(trade.price, 0) + trade.quantity
            elif buyer == "" and seller == "":
                # Anonymized round — infer from price proximity to daily extreme
                if abs(trade.price - new_min) <= OLIVIA_PRICE_TOLERANCE:
                    tick_buys[trade.price]  = tick_buys.get(trade.price, 0)  + trade.quantity
                elif abs(trade.price - new_max) <= OLIVIA_PRICE_TOLERANCE:
                    tick_sells[trade.price] = tick_sells.get(trade.price, 0) + trade.quantity
            # else: known non-Olivia trader — ignore

        # Detect Olivia's signature: exactly OLIVIA_LOT_SIZE at the daily extreme
        for price, vol in tick_buys.items():
            if vol == OLIVIA_LOT_SIZE and abs(price - new_min) <= OLIVIA_PRICE_TOLERANCE:
                signal = "LONG"
        for price, vol in tick_sells.items():
            if vol == OLIVIA_LOT_SIZE and abs(price - new_max) <= OLIVIA_PRICE_TOLERANCE:
                signal = "SHORT"

        buy_cap  = SQUID_LIMIT - pos
        sell_cap = SQUID_LIMIT + pos

        if signal == "LONG":
            if buy_cap > 0:
                orders.append(Order("SQUID_INK", best_ask,  buy_cap))
        elif signal == "SHORT":
            if sell_cap > 0:
                orders.append(Order("SQUID_INK", best_bid, -sell_cap))
        else:
            # No signal — flatten any open position
            if pos > 0: orders.append(Order("SQUID_INK", best_bid, -pos))
            elif pos < 0: orders.append(Order("SQUID_INK", best_ask, abs(pos)))

        saved["squid_daily_min"] = new_min
        saved["squid_daily_max"] = new_max
        saved["olivia_signal"]   = signal
        return orders, saved

    # ── BASKET ETF ARBITRAGE ──────────────────────────────────────────────────
    #
    # The basket price mean-reverts to its synthetic value (weighted sum of
    # constituent mids). We trade the basket only — no constituent hedging —
    # since hedging reduces EV slightly due to spread costs.
    #
    # Key design choices (from Frankfurt Hedgehogs analysis):
    #   1. Fixed thresholds, not z-score (no theoretical basis for vol-scaling)
    #   2. Subtract a running premium: baskets persistently trade above synthetic
    #   3. Exit at zero crossing, not opposite threshold (reduces variance)
    #   4. Enter aggressively (cross spread), exit aggressively
    #
    # Parameters to re-tune for every new round's product:
    #   BASKET_THRESHOLD, BASKET_PREMIUM_ALPHA

    def _trade_basket(
        self,
        basket_name: str,
        weights: Dict[str, int],
        limit: int,
        threshold: float,
        order_depths: Dict,
        positions: Dict,
        saved: dict,
    ):
        orders: List[Order] = []

        basket_depth = order_depths.get(basket_name)
        if not basket_depth or not basket_depth.buy_orders or not basket_depth.sell_orders:
            return orders, saved

        # Require all constituents to have valid books
        for constituent in weights:
            d = order_depths.get(constituent)
            if not d or not d.buy_orders or not d.sell_orders:
                return orders, saved

        # ── Compute basket mid ────────────────────────────────────────────────
        basket_bid = max(basket_depth.buy_orders.keys())
        basket_ask = min(basket_depth.sell_orders.keys())
        basket_mid = (basket_bid + basket_ask) / 2.0

        # ── Compute synthetic value ───────────────────────────────────────────
        synthetic = 0.0
        for constituent, weight in weights.items():
            d     = order_depths[constituent]
            c_bid = max(d.buy_orders.keys())
            c_ask = min(d.sell_orders.keys())
            synthetic += weight * (c_bid + c_ask) / 2.0

        # ── Raw spread ────────────────────────────────────────────────────────
        raw_spread = basket_mid - synthetic

        # ── Running premium (slow EMA) ────────────────────────────────────────
        # Tracks the persistent bias so the adjusted spread is centred at zero.
        pkey    = f"{basket_name}_premium"
        premium = saved.get(pkey, raw_spread)   # initialise to first observation
        premium = BASKET_PREMIUM_ALPHA * raw_spread + (1 - BASKET_PREMIUM_ALPHA) * premium
        saved[pkey] = premium

        # ── Adjusted spread ───────────────────────────────────────────────────
        adj_spread = raw_spread - premium

        # ── Signal state ──────────────────────────────────────────────────────
        skey   = f"{basket_name}_signal"
        signal = saved.get(skey, None)   # "LONG", "SHORT", or None
        pos    = positions.get(basket_name, 0)

        # Exit when adjusted spread crosses back through zero
        if signal == "LONG"  and adj_spread >= 0: signal = None
        if signal == "SHORT" and adj_spread <= 0: signal = None

        # Enter on new signal only when flat
        if signal is None:
            if   adj_spread < -threshold: signal = "LONG"
            elif adj_spread >  threshold: signal = "SHORT"

        saved[skey] = signal

        # ── Execute ───────────────────────────────────────────────────────────
        buy_cap  = limit - pos
        sell_cap = limit + pos

        if signal == "LONG":
            # Basket is cheap vs synthetic — buy it
            if buy_cap > 0:
                orders.append(Order(basket_name, basket_ask,  buy_cap))
        elif signal == "SHORT":
            # Basket is expensive vs synthetic — sell it
            if sell_cap > 0:
                orders.append(Order(basket_name, basket_bid, -sell_cap))
        else:
            # No signal — flatten any residual position
            if pos > 0: orders.append(Order(basket_name, basket_bid, -pos))
            elif pos < 0: orders.append(Order(basket_name, basket_ask, abs(pos)))

        return orders, saved
