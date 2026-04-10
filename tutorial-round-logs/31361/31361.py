from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

EMERALDS_LIMIT = 80
TOMATOES_LIMIT = 80
EMERALDS_FAIR = 10000
EMA_ALPHA = 0.3


class Trader:

    def bid(self) -> int:

        return 15

    def run(self, state: TradingState):

        saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        ema_tomatoes = saved.get("ema_tomatoes", None)

        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            pos = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product] = self._trade_emeralds(order_depth, pos)
            elif product == "TOMATOES":
                result[product], ema_tomatoes = self._trade_tomatoes(
                    order_depth, pos, ema_tomatoes
                )

        trader_data = jsonpickle.encode({"ema_tomatoes": ema_tomatoes})
        return result, 0, trader_data

    def _trade_emeralds(self, order_depth: OrderDepth, pos: int) -> List[Order]:
        orders: List[Order] = []
        buy_cap = EMERALDS_LIMIT - pos 
        sell_cap = EMERALDS_LIMIT + pos 

        for price in sorted(order_depth.sell_orders.keys()):
            if price > EMERALDS_FAIR or buy_cap <= 0:
                break
            vol = -order_depth.sell_orders[price]  
            qty = min(vol, buy_cap)
            orders.append(Order("EMERALDS", price, qty))
            buy_cap -= qty

        # Sell to any bid at or above fair value (anomaly state B: bid=10000)
        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if price < EMERALDS_FAIR or sell_cap <= 0:
                break
            vol = order_depth.buy_orders[price]
            qty = min(vol, sell_cap)
            orders.append(Order("EMERALDS", price, -qty))
            sell_cap -= qty

        passive_bid = EMERALDS_FAIR - 7   # 9993 — always 1 tick inside bot's wall
        passive_ask = EMERALDS_FAIR + 7   # 10007 — always 1 tick inside bot's wall

        if buy_cap > 0:
            orders.append(Order("EMERALDS", passive_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order("EMERALDS", passive_ask, -sell_cap))

        return orders

    def _trade_tomatoes(
        self, order_depth: OrderDepth, pos: int, ema: float
    ):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, ema

        wall_bid = max(order_depth.buy_orders.keys())
        wall_ask = min(order_depth.sell_orders.keys())
        mid = (wall_bid + wall_ask) / 2.0

        ema = EMA_ALPHA * mid + (1.0 - EMA_ALPHA) * ema if ema is not None else mid

        buy_cap = TOMATOES_LIMIT - pos
        sell_cap = TOMATOES_LIMIT + pos

        # Passive quotes around EMA with inventory skew.
        # EMA lag provides implicit contrarian bias: buys dips, sells spikes.
        # ±5 offset: 1 tick wider than the historical EMA-6 participant, earns
        # 5 ticks per fill while retaining exclusive best-bid status.
        skew = round(pos / TOMATOES_LIMIT * 3)
        bid_price = round(ema - 5 - skew)
        ask_price = round(ema + 5 - skew)

        if bid_price >= ask_price:
            bid_price = round(ema) - 1
            ask_price = round(ema) + 1

        if buy_cap > 0:
            orders.append(Order("TOMATOES", bid_price, buy_cap))
        if sell_cap > 0:
            orders.append(Order("TOMATOES", ask_price, -sell_cap))

        return orders, ema