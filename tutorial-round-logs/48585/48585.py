from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

EMERALDS_LIMIT = 80
TOMATOES_LIMIT = 80
EMERALDS_FAIR = 10000


class Trader:

    def bid(self) -> int:
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            pos = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product] = self._trade_emeralds(order_depth, pos)
            elif product == "TOMATOES":
                result[product] = self._trade_tomatoes(order_depth, pos)

        return result, 0, ""

    def _trade_emeralds(self, order_depth: OrderDepth, pos: int) -> List[Order]:
        orders: List[Order] = []
        buy_cap = EMERALDS_LIMIT - pos
        sell_cap = EMERALDS_LIMIT + pos

        # Take any ask at or below fair value (includes anomaly ask=10000)
        for price in sorted(order_depth.sell_orders.keys()):
            if price > EMERALDS_FAIR or buy_cap <= 0:
                break
            vol = -order_depth.sell_orders[price]
            qty = min(vol, buy_cap)
            orders.append(Order("EMERALDS", price, qty))
            buy_cap -= qty

        # Take any bid at or above fair value (includes anomaly bid=10000)
        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if price < EMERALDS_FAIR or sell_cap <= 0:
                break
            vol = order_depth.buy_orders[price]
            qty = min(vol, sell_cap)
            orders.append(Order("EMERALDS", price, -qty))
            sell_cap -= qty

        # Passive MM: 1 tick inside bot's ±8 wall
        if buy_cap > 0:
            orders.append(Order("EMERALDS", 9993, buy_cap))
        if sell_cap > 0:
            orders.append(Order("EMERALDS", 10007, -sell_cap))

        return orders

    def _trade_tomatoes(self, order_depth: OrderDepth, pos: int) -> List[Order]:
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        buy_cap = TOMATOES_LIMIT - pos
        sell_cap = TOMATOES_LIMIT + pos

        # Penny-jump: always quote 1 tick inside current L1
        # Adapts to both normal (bot at ±6.7) and tight-spread events
        # Inventory skew shifts both quotes to rebalance position
        skew = round(pos / TOMATOES_LIMIT * 3)
        bid_price = best_bid + 1 - skew
        ask_price = best_ask - 1 - skew

        # Safety: prevent crossed quotes
        if bid_price >= ask_price:
            bid_price = round(mid) - 1
            ask_price = round(mid) + 1

        if buy_cap > 0:
            orders.append(Order("TOMATOES", bid_price, buy_cap))
        if sell_cap > 0:
            orders.append(Order("TOMATOES", ask_price, -sell_cap))

        return orders