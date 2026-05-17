import asyncio
from typing import Callable, Awaitable


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = {}

    def subscribe(self, event: str, handler: Callable[..., Awaitable]) -> None:
        self._handlers.setdefault(event, []).append(handler)

    async def publish(self, event: str, **payload) -> None:
        for handler in self._handlers.get(event, []):
            try:
                await handler(**payload)
            except Exception as exc:
                pass  # handlers must not crash the bus

# rev 20260517212220-050d7eb6
