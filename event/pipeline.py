"""
事件處理管線 (Event Pipeline)

整合去重、累積、計數，形成完整的事件處理流程。
"""

from datetime import timedelta

from .schema import PPEDetectionEvent
from .dedup import EventDeduplicator
from .accumulator import EventAccumulator
from .counter import ObjectCounter


class EventPipeline:
    """
    事件處理管線

    處理流程：
    YOLO Events → 去重 → 累積 → 計數 → 輸出

    Attributes:
        deduplicator: 事件去重器
        accumulator: 事件累積器
        counter: 物件計數器
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        dedup_window: timedelta = timedelta(seconds=2),
        accumulate_window: timedelta = timedelta(minutes=5),
        counter_timeout: timedelta = timedelta(seconds=5)
    ):
        self.deduplicator = EventDeduplicator(
            iou_threshold=iou_threshold,
            time_window=dedup_window
        )
        self.accumulator = EventAccumulator(
            window=accumulate_window
        )
        self.counter = ObjectCounter(
            timeout=counter_timeout
        )

    def process(self, event: PPEDetectionEvent) -> dict:
        """
        處理單一事件，返回當前狀態

        Args:
            event: 偵測事件

        Returns:
            {
                "is_new": bool,        # 是否為新事件（非重複）
                "violations": int,     # 過去 N 分鐘違規次數
                "person_count": int,   # 目前場域人數
                "event": PPEDetectionEvent | None  # 如果是新事件則返回
            }
        """
        # 1. 去重
        deduped = self.deduplicator.process(event)
        is_new = deduped is not None

        # 2. 如果是新事件，加入累積器
        if is_new:
            self.accumulator.add(event)

        # 3. 更新計數器（無論是否重複都要更新，因為要追蹤位置）
        if event.object == "person":
            self.counter.update(event)

        return {
            "is_new": is_new,
            "violations": self.accumulator.get_violation_count(),
            "person_count": self.counter.get_count(),
            "event": deduped,
        }

    def get_status(self) -> dict:
        """
        取得當前狀態摘要

        Returns:
            {
                "violations": int,
                "person_count": int,
                "violation_breakdown": dict[str, int]
            }
        """
        return {
            "violations": self.accumulator.get_violation_count(),
            "person_count": self.counter.get_count(),
            "violation_breakdown": self.accumulator.get_counts(),
        }

    def reset(self) -> None:
        """重置所有狀態"""
        self.deduplicator.clear()
        self.accumulator.clear()
        self.counter.clear()


# === 使用範例 ===
if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta

    # 建立管線
    pipeline = EventPipeline()

    print("=== 事件處理管線示範 ===\n")

    # 模擬事件流
    base_time = datetime.now(timezone.utc)
    events = [
        PPEDetectionEvent(
            timestamp=base_time,
            object="person",
            confidence=0.95,
            bbox=[100, 50, 200, 300]
        ),
        PPEDetectionEvent(
            timestamp=base_time + timedelta(milliseconds=100),
            object="no_helmet",
            confidence=0.9,
            bbox=[100, 50, 200, 300]
        ),
        PPEDetectionEvent(
            timestamp=base_time + timedelta(milliseconds=200),
            object="no_helmet",
            confidence=0.88,
            bbox=[101, 51, 201, 301]  # 重複
        ),
        PPEDetectionEvent(
            timestamp=base_time + timedelta(milliseconds=300),
            object="person",
            confidence=0.93,
            bbox=[400, 50, 500, 300]
        ),
        PPEDetectionEvent(
            timestamp=base_time + timedelta(milliseconds=400),
            object="no_vest",
            confidence=0.85,
            bbox=[400, 50, 500, 300]
        ),
    ]

    for event in events:
        result = pipeline.process(event)
        status = "新事件" if result["is_new"] else "重複"
        print(f"[{status}] {event.object}")
        print(f"    違規: {result['violations']}, 人數: {result['person_count']}")
        print()

    print("=== 最終狀態 ===")
    print(pipeline.get_status())
