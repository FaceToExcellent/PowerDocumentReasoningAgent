"""指标采集 — 轻量实现：内存计数器 + JSON 快照导出（本机），生产可换 Prometheus
"""
import time
import json
import threading
from pathlib import Path
from typing import Dict


class MetricsCollector:
    """进程内指标计数 + 定时落盘 JSON（可被 Prometheus textfile / Grafana 读取）"""

    def __init__(self):
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
        self._lock = threading.Lock()
        self._start = time.time()

    def incr(self, name: str, value: float = 1, labels: dict = None):
        key = name
        if labels:
            suffix = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            key = f"{name}{{{suffix}}}"
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value: float, labels: dict = None):
        """记录观测值（延迟等），进入直方图"""
        key = name
        if labels:
            suffix = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            key = f"{name}{{{suffix}}}"
        with self._lock:
            self._histograms.setdefault(key, []).append(value)
            # 限制内存：只保留最近 2000 个采样
            if len(self._histograms[key]) > 2000:
                self._histograms[key] = self._histograms[key][-2000:]

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self._counters)
            hist = {}
            for k, v in self._histograms.items():
                if v:
                    hist[k] = {
                        "count": len(v),
                        "sum": round(sum(v), 2),
                        "avg": round(sum(v) / len(v), 2),
                        "p95": self._percentile(v, 0.95),
                        "max": round(max(v), 2),
                    }
        return {
            "uptime_seconds": round(time.time() - self._start, 1),
            "counters": counters,
            "histograms": hist,
        }

    @staticmethod
    def _percentile(values, p):
        s = sorted(values)
        idx = min(len(s) - 1, int(len(s) * p))
        return round(s[idx], 2)

    def to_prometheus_text(self) -> str:
        """导出 Prometheus text 格式（生产对接 /metrics）"""
        snap = self.snapshot()
        lines = ["# power_agent enterprise metrics"]
        for k, v in snap["counters"].items():
            lines.append(f"agent_counter_{k.replace('{','_').replace('}','').replace(',','_').replace('=','_')} {v}")
        for k, v in snap["histograms"].items():
            clean = k.replace("{", "_").replace("}", "").replace(",", "_").replace("=", "_")
            lines.append(f"agent_histogram_{clean}_avg {v['avg']}")
        return "\n".join(lines)


metrics = MetricsCollector()
