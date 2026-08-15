"""进程内 pub/sub — 审批结果推送给前端

单进程 uvicorn 场景够用(与 _ABORT_EVENT/STREAM_QUEUE 同构)。
生产多进程部署需换 Redis pub/sub。
"""
import asyncio
from collections import defaultdict
from typing import Dict, Set

_subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)


def subscribe(thread_id: str) -> asyncio.Queue:
    """订阅某线程的推送,返回接收队列。"""
    q: asyncio.Queue = asyncio.Queue()
    _subscribers[thread_id].add(q)
    return q


def unsubscribe(thread_id: str, q: asyncio.Queue) -> None:
    """取消订阅并清理空集合。"""
    _subscribers.get(thread_id, set()).discard(q)
    if not _subscribers.get(thread_id):
        _subscribers.pop(thread_id, None)


def publish(thread_id: str, event: dict) -> None:
    """向某线程的所有订阅者广播一个事件({event, data})。"""
    for q in list(_subscribers.get(thread_id, ())):
        try:
            q.put_nowait(event)
        except Exception:
            pass
