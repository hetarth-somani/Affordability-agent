"""Currency conversion using the fixed, dated exchange rates supplied in the dataset."""

from __future__ import annotations

import logging
from collections import deque
from datetime import date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CurrencyConverter:
    """Converts amounts between currencies using a directed rate graph per date.

    Builds a graph of exchange rates for a given date and finds the shortest
    conversion path via BFS.  Automatically derives inverse rates so that a
    single USD->INR row also enables INR->USD conversion.  Falls back to the
    nearest available date if no exact match exists.
    """

    def __init__(self, exchange_rates: Dict[Tuple[date, str, str], float]) -> None:
        self._rates = exchange_rates
        self._available_dates: List[date] = sorted({key[0] for key in exchange_rates})

    def convert(self, amount: float, from_currency: str, to_currency: str, on_date: date) -> float:
        if from_currency == to_currency:
            return amount

        graph = self._build_graph(self._resolve_date(on_date))
        path = self._shortest_path(graph, from_currency, to_currency)
        if path is None:
            raise ValueError(
                f"No conversion path from {from_currency} to {to_currency} on {on_date}"
            )

        result = amount
        for i in range(len(path) - 1):
            result *= graph[path[i]][path[i + 1]]
        return result

    def _resolve_date(self, on_date: date) -> date:
        if any(d == on_date for d in self._available_dates):
            return on_date
        nearest = min(self._available_dates, key=lambda d: abs((d - on_date).days))
        logger.warning(
            "No exchange rate row for %s; falling back to nearest available date %s",
            on_date,
            nearest,
        )
        return nearest

    def _build_graph(self, on_date: date) -> Dict[str, Dict[str, float]]:
        graph: Dict[str, Dict[str, float]] = {}
        for (rate_date, from_ccy, to_ccy), rate in self._rates.items():
            if rate_date != on_date:
                continue
            graph.setdefault(from_ccy, {})[to_ccy] = rate
            graph.setdefault(to_ccy, {})[from_ccy] = 1.0 / rate
        return graph

    @staticmethod
    def _shortest_path(
        graph: Dict[str, Dict[str, float]], start: str, goal: str
    ) -> Optional[List[str]]:
        if start not in graph or goal not in graph:
            return None
        visited = {start}
        queue: deque[List[str]] = deque([[start]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == goal:
                return path
            for neighbour in graph.get(node, {}):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(path + [neighbour])
        return None
