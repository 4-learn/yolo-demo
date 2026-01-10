"""
事件累積器 (Event Accumulator)

統計時間窗口內各類別的事件數量。
"""

from datetime import datetime, timedelta, timezone
from collections import defaultdict

from .schema import PPEDetectionEvent


class EventAccumulator:
    """
    事件累積器

    統計時間窗口內各類別的事件數量。

    Attributes:
        window: 時間窗口，只統計此時間內的事件
    """

    def __init__(self, window: timedelta = timedelta(minutes=5)):
        self.window = window
        self.events: list[PPEDetectionEvent] = []

    def add(self, event: PPEDetectionEvent) -> None:
        """
        加入事件

        Args:
            event: 要加入的事件
        """
        self.events.append(event)
        self._cleanup()

    def _cleanup(self) -> None:
        """清理過期事件"""
        now = datetime.now(timezone.utc)
        self.events = [
            e for e in self.events
            if now - e.timestamp < self.window
        ]

    def get_counts(self) -> dict[str, int]:
        """
        取得各類別的事件數量

        Returns:
            字典，key 為物件類別，value 為數量
        """
        self._cleanup()
        counts = defaultdict(int)
        for event in self.events:
            counts[event.object] += 1
        return dict(counts)

    def get_violation_count(self) -> int:
        """
        取得違規事件數量

        違規事件包含：no_helmet, no_vest

        Returns:
            違規事件總數
        """
        counts = self.get_counts()
        return counts.get("no_helmet", 0) + counts.get("no_vest", 0)

    def get_total_count(self) -> int:
        """
        取得所有事件數量

        Returns:
            事件總數
        """
        self._cleanup()
        return len(self.events)

    def clear(self) -> None:
        """清空所有事件"""
        self.events.clear()


# === 使用範例 ===
if __name__ == "__main__":
    from datetime import datetime, timezone

    # 建立累積器（5 分鐘窗口）
    accumulator = EventAccumulator(window=timedelta(minutes=5))

    # 加入事件
    events = [
        PPEDetectionEvent(object="no_helmet", confidence=0.9, bbox=[100, 50, 200, 300]),
        PPEDetectionEvent(object="no_vest", confidence=0.85, bbox=[150, 60, 250, 350]),
        PPEDetectionEvent(object="helmet", confidence=0.95, bbox=[200, 70, 300, 400]),
        PPEDetectionEvent(object="no_helmet", confidence=0.88, bbox=[300, 80, 400, 450]),
        PPEDetectionEvent(object="vest", confidence=0.92, bbox=[350, 90, 450, 500]),
    ]

    print("=== 事件累積器示範 ===\n")

    for event in events:
        accumulator.add(event)
        print(f"加入: {event.object}")

    print(f"\n各類別統計: {accumulator.get_counts()}")
    print(f"違規次數: {accumulator.get_violation_count()}")
    print(f"總事件數: {accumulator.get_total_count()}")
