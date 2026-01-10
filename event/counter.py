"""
物件計數器 (Object Counter)

追蹤場域內的物件數量。
"""

from datetime import datetime, timedelta, timezone

from .schema import PPEDetectionEvent


class ObjectCounter:
    """
    物件計數器

    追蹤場域內的物件數量。如果物件超過 timeout 沒有被偵測到，
    就視為已離開。

    Attributes:
        timeout: 超時時間，超過此時間沒偵測到就視為離開
    """

    def __init__(self, timeout: timedelta = timedelta(seconds=5)):
        self.timeout = timeout
        self.objects: dict[str, datetime] = {}  # object_id -> last_seen

    def update(self, event: PPEDetectionEvent) -> None:
        """
        更新物件狀態

        Args:
            event: 偵測事件
        """
        object_id = self._get_object_id(event.bbox)
        self.objects[object_id] = event.timestamp
        self._cleanup()

    def _get_object_id(self, bbox: list) -> str:
        """
        從 bbox 產生簡易 ID

        使用 bbox 中心點量化到格子作為 ID。
        實際應用會使用更複雜的 tracking 演算法（如 DeepSORT）。

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            物件 ID
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        # 量化到 50x50 格子，減少 ID 數量
        gx, gy = int(cx // 50), int(cy // 50)
        return f"{gx}_{gy}"

    def _cleanup(self) -> None:
        """清理超時的物件"""
        now = datetime.now(timezone.utc)
        self.objects = {
            oid: ts for oid, ts in self.objects.items()
            if now - ts < self.timeout
        }

    def get_count(self) -> int:
        """
        取得目前物件數量

        Returns:
            目前場域內的物件數量
        """
        self._cleanup()
        return len(self.objects)

    def get_object_ids(self) -> list[str]:
        """
        取得目前所有物件 ID

        Returns:
            物件 ID 列表
        """
        self._cleanup()
        return list(self.objects.keys())

    def clear(self) -> None:
        """清空所有物件"""
        self.objects.clear()


# === 使用範例 ===
if __name__ == "__main__":
    from datetime import datetime, timezone
    import time

    # 建立計數器（5 秒超時）
    counter = ObjectCounter(timeout=timedelta(seconds=5))

    print("=== 物件計數器示範 ===\n")

    # 模擬多人進入
    events = [
        PPEDetectionEvent(object="person", confidence=0.95, bbox=[100, 50, 200, 300]),
        PPEDetectionEvent(object="person", confidence=0.93, bbox=[400, 50, 500, 300]),
        PPEDetectionEvent(object="person", confidence=0.91, bbox=[700, 50, 800, 300]),
    ]

    for i, event in enumerate(events):
        counter.update(event)
        print(f"人員 {i+1} 進入，目前人數: {counter.get_count()}")

    print(f"\n物件 ID: {counter.get_object_ids()}")

    # 模擬同一人持續被偵測
    print("\n同一人持續被偵測（位置略有變化）...")
    event = PPEDetectionEvent(object="person", confidence=0.94, bbox=[102, 51, 202, 301])
    counter.update(event)
    print(f"目前人數: {counter.get_count()}")
