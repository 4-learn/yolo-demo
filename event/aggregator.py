"""
事件聚合器 (Event Aggregator)

提供事件聚合與統計功能。
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from .schema import PPEDetectionEvent
from .filters import EventFilter, TrueFilter


@dataclass
class AggregateResult:
    """聚合結果"""
    group_key: str
    count: int = 0
    events: list[PPEDetectionEvent] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class EventAggregator:
    """事件聚合器"""

    def __init__(
        self,
        group_by: Callable[[PPEDetectionEvent], str],
        filter: Optional[EventFilter] = None
    ):
        """
        Args:
            group_by: 分組函數，返回分組鍵
            filter: 過濾器（可選）
        """
        self.group_by = group_by
        self.filter = filter
        self.groups: dict[str, list[PPEDetectionEvent]] = defaultdict(list)

    def add(self, event: PPEDetectionEvent) -> None:
        """加入事件"""
        if self.filter and not self.filter.match(event):
            return

        key = self.group_by(event)
        self.groups[key].append(event)

    def add_batch(self, events: list[PPEDetectionEvent]) -> None:
        """批次加入事件"""
        for event in events:
            self.add(event)

    def get_results(self) -> list[AggregateResult]:
        """取得聚合結果"""
        results = []

        for key, events in sorted(self.groups.items()):
            confidences = [e.confidence for e in events]

            results.append(AggregateResult(
                group_key=key,
                count=len(events),
                events=events,
                stats={
                    "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
                    "min_confidence": min(confidences) if confidences else 0,
                    "max_confidence": max(confidences) if confidences else 0,
                }
            ))

        return results

    def get_counts(self) -> dict[str, int]:
        """取得各分組的事件數量"""
        return {key: len(events) for key, events in self.groups.items()}

    def clear(self) -> None:
        """清空聚合器"""
        self.groups.clear()


# === 常用分組函數 ===

def group_by_object(event: PPEDetectionEvent) -> str:
    """依物件類別分組"""
    return event.object


def group_by_source(event: PPEDetectionEvent) -> str:
    """依來源（攝影機）分組"""
    return event.source or "unknown"


def group_by_hour(event: PPEDetectionEvent) -> str:
    """依小時分組"""
    return event.timestamp.strftime("%Y-%m-%d %H:00")


def group_by_date(event: PPEDetectionEvent) -> str:
    """依日期分組"""
    return event.timestamp.strftime("%Y-%m-%d")


def group_by_interval(interval_minutes: int = 10) -> Callable:
    """依時間區間分組"""
    def _group(event: PPEDetectionEvent) -> str:
        ts = event.timestamp
        interval_start = ts.replace(
            minute=(ts.minute // interval_minutes) * interval_minutes,
            second=0,
            microsecond=0
        )
        return interval_start.strftime("%Y-%m-%d %H:%M")
    return _group


def group_by_zone(zone_map: dict) -> Callable:
    """依區域分組"""
    def _get_zone(bbox: list) -> str:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        for zone_name, regions in zone_map.items():
            for region in regions:
                if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
                    return zone_name
        return "unknown"

    def _group(event: PPEDetectionEvent) -> str:
        return _get_zone(event.bbox)

    return _group


# === 多維度聚合器 ===

class MultiDimensionAggregator:
    """多維度聚合器"""

    def __init__(
        self,
        dimensions: list[tuple[str, Callable]],
        filter: Optional[EventFilter] = None
    ):
        """
        Args:
            dimensions: [(維度名稱, 分組函數), ...]
            filter: 過濾器
        """
        self.dimensions = dimensions
        self.filter = filter
        self.data: dict[tuple, list[PPEDetectionEvent]] = defaultdict(list)

    def add(self, event: PPEDetectionEvent) -> None:
        if self.filter and not self.filter.match(event):
            return

        key = tuple(dim_func(event) for _, dim_func in self.dimensions)
        self.data[key].append(event)

    def add_batch(self, events: list[PPEDetectionEvent]) -> None:
        for event in events:
            self.add(event)

    def get_results(self) -> list[dict]:
        """取得聚合結果"""
        results = []

        for key, events in sorted(self.data.items()):
            result = {
                name: key[i]
                for i, (name, _) in enumerate(self.dimensions)
            }
            result["count"] = len(events)
            result["avg_confidence"] = sum(e.confidence for e in events) / len(events)
            results.append(result)

        return results

    def to_dataframe(self):
        """轉換為 Pandas DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.get_results())

    def clear(self) -> None:
        """清空聚合器"""
        self.data.clear()


# === 滑動視窗統計 ===

class SlidingWindowStats:
    """滑動視窗統計"""

    def __init__(
        self,
        window: timedelta,
        group_by: Optional[Callable] = None
    ):
        self.window = window
        self.group_by = group_by
        self.events: deque = deque()

    def add(self, event: PPEDetectionEvent) -> None:
        """加入事件"""
        self.events.append(event)
        self._cleanup(event.timestamp)

    def _cleanup(self, current_time: datetime) -> None:
        """清理過期事件"""
        cutoff = current_time - self.window
        while self.events and self.events[0].timestamp < cutoff:
            self.events.popleft()

    def get_count(self) -> int:
        """取得視窗內事件數量"""
        return len(self.events)

    def get_counts_by_group(self) -> dict[str, int]:
        """取得各分組的事件數量"""
        if not self.group_by:
            return {"total": len(self.events)}

        counts = defaultdict(int)
        for event in self.events:
            key = self.group_by(event)
            counts[key] += 1
        return dict(counts)

    def get_rate(self) -> float:
        """取得事件速率（每分鐘）"""
        if not self.events:
            return 0.0
        return len(self.events) / (self.window.total_seconds() / 60)

    def get_events(self) -> list[PPEDetectionEvent]:
        """取得視窗內所有事件"""
        return list(self.events)

    def clear(self) -> None:
        """清空視窗"""
        self.events.clear()


# === 使用範例 ===
if __name__ == "__main__":
    import random
    from datetime import timezone

    print("=== 事件聚合器示範 ===\n")

    # 區域定義
    zone_map = {
        "construction": [[0, 0, 400, 600]],
        "office": [[400, 0, 800, 600]],
    }

    # 產生測試事件
    base_time = datetime.now(timezone.utc)
    events = []

    for i in range(100):
        obj = random.choice(["no_helmet", "no_vest", "helmet", "vest"])
        conf = random.uniform(0.5, 0.99)
        x1 = random.uniform(0, 600)
        y1 = random.uniform(0, 400)

        events.append(PPEDetectionEvent(
            timestamp=base_time + timedelta(minutes=random.randint(0, 60)),
            object=obj,
            confidence=conf,
            bbox=[x1, y1, x1 + 100, y1 + 200],
            source=random.choice(["camera_01", "camera_02", "camera_03"])
        ))

    # 依物件類別聚合
    print("=== 依物件類別 ===")
    by_object = EventAggregator(group_by=group_by_object)
    by_object.add_batch(events)

    for result in by_object.get_results():
        print(f"  {result.group_key}: {result.count} 筆 (avg_conf={result.stats['avg_confidence']:.2f})")

    # 依區域聚合
    print("\n=== 依區域 ===")
    from .filters import ConfidenceFilter
    by_zone = EventAggregator(
        group_by=group_by_zone(zone_map),
        filter=ConfidenceFilter(min_conf=0.7)
    )
    by_zone.add_batch(events)

    for result in by_zone.get_results():
        print(f"  {result.group_key}: {result.count} 筆")

    # 依時間區間聚合（每 10 分鐘）
    print("\n=== 每 10 分鐘 ===")
    from .filters import ObjectFilter
    by_interval = EventAggregator(
        group_by=group_by_interval(10),
        filter=ObjectFilter(["no_helmet", "no_vest"])
    )
    by_interval.add_batch(events)

    for result in by_interval.get_results():
        bar = "█" * min(result.count, 20)
        print(f"  {result.group_key}: {bar} {result.count}")

    # 滑動視窗
    print("\n=== 滑動視窗（5 分鐘）===")
    window = SlidingWindowStats(
        window=timedelta(minutes=5),
        group_by=group_by_object
    )

    for event in events[:20]:  # 只加入前 20 筆
        window.add(event)

    print(f"  視窗內事件: {window.get_count()}")
    print(f"  分組統計: {window.get_counts_by_group()}")
    print(f"  事件速率: {window.get_rate():.1f} 次/分鐘")
