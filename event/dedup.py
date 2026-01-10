"""
事件去重器 (Event Deduplicator)

使用 IoU 判斷兩個 bounding box 是否為同一物件，
在時間窗口內重複的事件會被過濾。
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from .schema import PPEDetectionEvent


def calculate_iou(box1: list, box2: list) -> float:
    """
    計算兩個 bounding box 的 IoU (Intersection over Union)

    Args:
        box1: [x1, y1, x2, y2] 左上角和右下角座標
        box2: [x1, y1, x2, y2] 左上角和右下角座標

    Returns:
        IoU 值 (0-1)
    """
    # 計算交集區域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 如果沒有交集
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 計算聯集
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class EventDeduplicator:
    """
    事件去重器

    在指定時間窗口內，相同類別且 IoU > 閾值的事件會被視為重複。

    Attributes:
        iou_threshold: IoU 閾值，超過此值視為同一物件
        time_window: 時間窗口，在此時間內的重複事件會被過濾
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        time_window: timedelta = timedelta(seconds=2)
    ):
        self.iou_threshold = iou_threshold
        self.time_window = time_window
        self.recent_events: list[PPEDetectionEvent] = []

    def is_duplicate(self, event: PPEDetectionEvent) -> bool:
        """
        檢查事件是否為重複

        Args:
            event: 要檢查的事件

        Returns:
            True 如果是重複事件，否則 False
        """
        now = event.timestamp

        # 清理過期事件
        self.recent_events = [
            e for e in self.recent_events
            if now - e.timestamp < self.time_window
        ]

        # 檢查是否與最近事件重複
        for recent in self.recent_events:
            # 不同類別的物件不算重複
            if recent.object != event.object:
                continue

            # 計算 IoU
            iou = calculate_iou(recent.bbox, event.bbox)
            if iou > self.iou_threshold:
                return True

        return False

    def process(self, event: PPEDetectionEvent) -> Optional[PPEDetectionEvent]:
        """
        處理事件

        如果事件不是重複，則返回該事件並加入追蹤列表。
        如果是重複事件，則返回 None。

        Args:
            event: 要處理的事件

        Returns:
            非重複事件返回原事件，重複事件返回 None
        """
        if self.is_duplicate(event):
            return None

        self.recent_events.append(event)
        return event

    def clear(self) -> None:
        """清空追蹤列表"""
        self.recent_events.clear()


# === 使用範例 ===
if __name__ == "__main__":
    from datetime import datetime, timezone

    # 建立去重器
    deduper = EventDeduplicator(iou_threshold=0.5, time_window=timedelta(seconds=2))

    # 模擬連續影格的偵測
    base_time = datetime.now(timezone.utc)
    events = [
        PPEDetectionEvent(
            timestamp=base_time,
            object="no_helmet",
            confidence=0.9,
            bbox=[100, 50, 200, 300]
        ),
        PPEDetectionEvent(
            timestamp=base_time + timedelta(milliseconds=33),
            object="no_helmet",
            confidence=0.88,
            bbox=[102, 51, 202, 301]  # 幾乎一樣，應該被過濾
        ),
        PPEDetectionEvent(
            timestamp=base_time + timedelta(milliseconds=66),
            object="no_helmet",
            confidence=0.87,
            bbox=[101, 50, 201, 300]  # 幾乎一樣，應該被過濾
        ),
        PPEDetectionEvent(
            timestamp=base_time + timedelta(milliseconds=100),
            object="no_helmet",
            confidence=0.85,
            bbox=[500, 50, 600, 300]  # 不同位置，新的人
        ),
    ]

    print("=== 事件去重示範 ===\n")
    for i, event in enumerate(events):
        result = deduper.process(event)
        if result:
            print(f"事件 {i+1}: ✓ 新事件 - {result.object} at {result.bbox}")
        else:
            print(f"事件 {i+1}: ✗ 重複，忽略")

    # 測試 IoU 計算
    print("\n=== IoU 計算示範 ===\n")
    box1 = [100, 50, 200, 300]
    box2 = [102, 51, 202, 301]
    box3 = [500, 50, 600, 300]

    print(f"box1 vs box2 IoU: {calculate_iou(box1, box2):.4f}")
    print(f"box1 vs box3 IoU: {calculate_iou(box1, box3):.4f}")
