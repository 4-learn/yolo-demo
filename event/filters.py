"""
事件過濾器 (Event Filters)

提供可組合的事件過濾機制。
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Callable

from .schema import PPEDetectionEvent


class EventFilter(ABC):
    """事件過濾器基類"""

    @abstractmethod
    def match(self, event: PPEDetectionEvent) -> bool:
        """檢查事件是否符合過濾條件"""
        pass

    def __and__(self, other: "EventFilter") -> "AndFilter":
        """支援 & 運算符"""
        return AndFilter(self, other)

    def __or__(self, other: "EventFilter") -> "OrFilter":
        """支援 | 運算符"""
        return OrFilter(self, other)

    def __invert__(self) -> "NotFilter":
        """支援 ~ 運算符"""
        return NotFilter(self)


# === 基本過濾器 ===

class TrueFilter(EventFilter):
    """永遠匹配的過濾器"""

    def match(self, event: PPEDetectionEvent) -> bool:
        return True


class FalseFilter(EventFilter):
    """永遠不匹配的過濾器"""

    def match(self, event: PPEDetectionEvent) -> bool:
        return False


class ObjectFilter(EventFilter):
    """依物件類別過濾"""

    def __init__(self, objects: list[str]):
        self.objects = set(objects)

    def match(self, event: PPEDetectionEvent) -> bool:
        return event.object in self.objects


class ConfidenceFilter(EventFilter):
    """依信心分數過濾"""

    def __init__(self, min_conf: float = 0.0, max_conf: float = 1.0):
        self.min_conf = min_conf
        self.max_conf = max_conf

    def match(self, event: PPEDetectionEvent) -> bool:
        return self.min_conf <= event.confidence <= self.max_conf


class TimeRangeFilter(EventFilter):
    """依時間範圍過濾"""

    def __init__(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ):
        self.start = start
        self.end = end

    def match(self, event: PPEDetectionEvent) -> bool:
        if self.start and event.timestamp < self.start:
            return False
        if self.end and event.timestamp > self.end:
            return False
        return True


class SourceFilter(EventFilter):
    """依來源（攝影機）過濾"""

    def __init__(self, sources: list[str]):
        self.sources = set(sources)

    def match(self, event: PPEDetectionEvent) -> bool:
        return event.source in self.sources


class ZoneFilter(EventFilter):
    """依區域過濾"""

    def __init__(self, zones: list[str], zone_map: dict):
        self.zones = set(zones)
        self.zone_map = zone_map

    def _get_zone(self, bbox: list) -> Optional[str]:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        for zone_name, regions in self.zone_map.items():
            for region in regions:
                if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
                    return zone_name
        return None

    def match(self, event: PPEDetectionEvent) -> bool:
        zone = self._get_zone(event.bbox)
        return zone in self.zones


class LambdaFilter(EventFilter):
    """使用自定義函數過濾"""

    def __init__(self, func: Callable[[PPEDetectionEvent], bool]):
        self.func = func

    def match(self, event: PPEDetectionEvent) -> bool:
        return self.func(event)


# === 組合過濾器 ===

class AndFilter(EventFilter):
    """AND 組合過濾器"""

    def __init__(self, *filters: EventFilter):
        self.filters = filters

    def match(self, event: PPEDetectionEvent) -> bool:
        return all(f.match(event) for f in self.filters)


class OrFilter(EventFilter):
    """OR 組合過濾器"""

    def __init__(self, *filters: EventFilter):
        self.filters = filters

    def match(self, event: PPEDetectionEvent) -> bool:
        return any(f.match(event) for f in self.filters)


class NotFilter(EventFilter):
    """NOT 過濾器"""

    def __init__(self, filter: EventFilter):
        self.filter = filter

    def match(self, event: PPEDetectionEvent) -> bool:
        return not self.filter.match(event)


# === 過濾器工廠 ===

class FilterFactory:
    """過濾器工廠，從設定建立過濾器"""

    def __init__(self, zone_map: dict = None):
        self.zone_map = zone_map or {}

    def create(self, config: dict) -> EventFilter:
        """
        從設定建立過濾器

        Args:
            config: {
                "objects": ["no_helmet"],
                "confidence_min": 0.7,
                "confidence_max": 1.0,
                "sources": ["camera_01"],
                "zones": ["construction"],
                "time_start": "2024-01-15T00:00:00",
                "time_end": "2024-01-15T23:59:59"
            }

        Returns:
            組合後的過濾器
        """
        filters = []

        if "objects" in config:
            filters.append(ObjectFilter(config["objects"]))

        if "confidence_min" in config or "confidence_max" in config:
            filters.append(ConfidenceFilter(
                min_conf=config.get("confidence_min", 0.0),
                max_conf=config.get("confidence_max", 1.0)
            ))

        if "sources" in config:
            filters.append(SourceFilter(config["sources"]))

        if "zones" in config:
            filters.append(ZoneFilter(config["zones"], self.zone_map))

        if "time_start" in config or "time_end" in config:
            start = datetime.fromisoformat(config["time_start"]) if "time_start" in config else None
            end = datetime.fromisoformat(config["time_end"]) if "time_end" in config else None
            filters.append(TimeRangeFilter(start, end))

        if not filters:
            return TrueFilter()

        return AndFilter(*filters)


# === 使用範例 ===
if __name__ == "__main__":
    from datetime import timezone

    print("=== 事件過濾器示範 ===\n")

    # 區域定義
    zone_map = {
        "construction": [[0, 0, 400, 600]],
        "office": [[400, 0, 800, 600]],
    }

    # 建立測試事件
    events = [
        PPEDetectionEvent(
            object="no_helmet",
            confidence=0.85,
            bbox=[100, 100, 200, 300],
            source="camera_01"
        ),
        PPEDetectionEvent(
            object="no_helmet",
            confidence=0.55,
            bbox=[100, 100, 200, 300],
            source="camera_01"
        ),
        PPEDetectionEvent(
            object="helmet",
            confidence=0.9,
            bbox=[100, 100, 200, 300],
            source="camera_01"
        ),
        PPEDetectionEvent(
            object="no_vest",
            confidence=0.75,
            bbox=[500, 100, 600, 300],  # 辦公區
            source="camera_02"
        ),
    ]

    # 建立過濾器
    violation_filter = ObjectFilter(["no_helmet", "no_vest"])
    high_confidence = ConfidenceFilter(min_conf=0.7)
    construction_zone = ZoneFilter(["construction"], zone_map)

    # 組合過濾器
    critical_violations = violation_filter & high_confidence & construction_zone

    print("測試事件：")
    for i, event in enumerate(events):
        print(f"  {i+1}. {event.object} (conf={event.confidence:.2f}, bbox={event.bbox})")

    print("\n過濾條件: 違規 & 高信心 & 施工區")
    filtered = [e for e in events if critical_violations.match(e)]
    print(f"符合條件: {len(filtered)} 筆")
    for event in filtered:
        print(f"  - {event.object} (conf={event.confidence:.2f})")

    # 使用過濾器工廠
    print("\n=== 過濾器工廠 ===")
    factory = FilterFactory(zone_map)
    config = {
        "objects": ["no_helmet", "no_vest"],
        "confidence_min": 0.6,
    }
    filter_from_config = factory.create(config)
    filtered2 = [e for e in events if filter_from_config.match(e)]
    print(f"Config: {config}")
    print(f"符合條件: {len(filtered2)} 筆")
