"""
系統整合模組 (System Integration)

整合所有事件處理模組，提供完整的工安監控管線。
"""

import random
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable

from .schema import PPEDetectionEvent
from .dedup import EventDeduplicator
from .rules import RuleEngine, Rule, RuleConditions, Action, Severity
from .ml_classifier import ViolationClassifier
from .hybrid_classifier import HybridClassifier
from .filters import ObjectFilter
from .aggregator import (
    EventAggregator,
    SlidingWindowStats,
    group_by_object,
    group_by_zone,
)


# === 處理結果 ===

@dataclass
class ProcessingResult:
    """處理結果"""
    event: PPEDetectionEvent
    is_duplicate: bool
    action: Action
    severity: Optional[Severity]
    reason: str
    ml_probability: Optional[float] = None


# === 告警 ===

@dataclass
class Alert:
    """告警記錄"""
    id: str
    timestamp: datetime
    event: PPEDetectionEvent
    action: Action
    severity: Severity
    reason: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class AlertManager:
    """告警管理器"""

    def __init__(self, max_alerts: int = 1000):
        self.alerts: list[Alert] = []
        self.max_alerts = max_alerts
        self.alert_count = 0

    def create_alert(self, result: ProcessingResult) -> Alert:
        """建立告警"""
        self.alert_count += 1
        alert = Alert(
            id=f"alert_{self.alert_count:06d}",
            timestamp=datetime.now(timezone.utc),
            event=result.event,
            action=result.action,
            severity=result.severity or Severity.MEDIUM,
            reason=result.reason,
        )
        self.alerts.append(alert)

        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

        return alert

    def get_unacknowledged(self) -> list[Alert]:
        """取得未確認的告警"""
        return [a for a in self.alerts if not a.acknowledged]

    def acknowledge(self, alert_id: str, by: str) -> bool:
        """確認告警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = by
                alert.acknowledged_at = datetime.now(timezone.utc)
                return True
        return False

    def get_stats(self) -> dict:
        """取得告警統計"""
        by_severity = {}
        for alert in self.alerts:
            sev = alert.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total": len(self.alerts),
            "unacknowledged": len(self.get_unacknowledged()),
            "by_severity": by_severity,
        }

    def to_json(self, alert: Alert) -> str:
        """告警轉 JSON"""
        return json.dumps({
            "id": alert.id,
            "timestamp": alert.timestamp.isoformat(),
            "object": alert.event.object,
            "confidence": alert.event.confidence,
            "location": alert.event.bbox,
            "severity": alert.severity.value,
            "reason": alert.reason,
        }, ensure_ascii=False, indent=2)


# === YOLO 模擬器 ===

class YOLOSimulator:
    """模擬 YOLO 偵測輸出"""

    CLASS_NAMES = {
        0: "helmet",
        1: "no_helmet",
        2: "vest",
        3: "no_vest",
        4: "person",
    }

    def __init__(
        self,
        frame_width: int = 800,
        frame_height: int = 600,
        violation_rate: float = 0.3,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.violation_rate = violation_rate
        self.frame_count = 0

    def detect(self, num_objects: int = None) -> list[PPEDetectionEvent]:
        """模擬一個影格的偵測結果"""
        if num_objects is None:
            num_objects = random.randint(1, 5)

        self.frame_count += 1
        events = []

        for _ in range(num_objects):
            is_violation = random.random() < self.violation_rate
            if is_violation:
                obj = random.choice(["no_helmet", "no_vest"])
                confidence = random.uniform(0.6, 0.95)
            else:
                obj = random.choice(["helmet", "vest", "person"])
                confidence = random.uniform(0.7, 0.99)

            width = random.uniform(50, 150)
            height = random.uniform(100, 300)
            x1 = random.uniform(0, self.frame_width - width)
            y1 = random.uniform(0, self.frame_height - height)

            events.append(PPEDetectionEvent(
                timestamp=datetime.now(timezone.utc),
                object=obj,
                confidence=confidence,
                bbox=[x1, y1, x1 + width, y1 + height],
                source="camera_01",
                metadata={"frame_id": self.frame_count}
            ))

        return events


# === LLM 介面 ===

class LLMInterface(ABC):
    """LLM 介面（Ch06 實作）"""

    @abstractmethod
    def generate_alert_message(self, alert: Alert) -> str:
        pass

    @abstractmethod
    def describe_violation(self, event: PPEDetectionEvent) -> str:
        pass

    @abstractmethod
    def suggest_regulations(self, violation_type: str) -> list[str]:
        pass


class SimpleLLMPlaceholder(LLMInterface):
    """簡單的 LLM 佔位實作"""

    MESSAGES = {
        "no_helmet": "偵測到人員未配戴安全帽，請立即處理。",
        "no_vest": "偵測到人員未穿著反光背心，請注意安全。",
    }

    REGULATIONS = {
        "no_helmet": [
            "職業安全衛生設施規則 第 281 條",
            "營造安全衛生設施標準 第 11-1 條",
        ],
        "no_vest": [
            "職業安全衛生設施規則 第 21 條",
        ],
    }

    def generate_alert_message(self, alert: Alert) -> str:
        base_msg = self.MESSAGES.get(alert.event.object, "偵測到安全違規。")
        return f"[{alert.severity.value.upper()}] {base_msg}"

    def describe_violation(self, event: PPEDetectionEvent) -> str:
        return f"於 {event.timestamp.strftime('%H:%M:%S')} 偵測到 {event.object}，信心度 {event.confidence:.0%}"

    def suggest_regulations(self, violation_type: str) -> list[str]:
        return self.REGULATIONS.get(violation_type, [])


# === 整合管線 ===

class SafetyMonitoringPipeline:
    """工安監控整合管線"""

    def __init__(
        self,
        rules: list[Rule],
        zone_map: dict,
        ml_classifier: Optional[ViolationClassifier] = None,
        ml_threshold: float = 0.7,
        on_violation: Optional[Callable[[ProcessingResult], None]] = None,
        on_alert: Optional[Callable[[ProcessingResult], None]] = None,
    ):
        self.zone_map = zone_map

        # 去重器
        self.deduplicator = EventDeduplicator(
            iou_threshold=0.5,
            time_window=timedelta(seconds=2)
        )

        # 規則引擎
        self.rule_engine = RuleEngine(rules, zone_map)

        # 混合分類器
        self.hybrid_classifier = None
        if ml_classifier:
            self.hybrid_classifier = HybridClassifier(
                rule_engine=self.rule_engine,
                ml_classifier=ml_classifier,
                ml_threshold=ml_threshold
            )

        # 統計器
        self.stats_by_object = EventAggregator(group_by=group_by_object)
        self.stats_by_zone = EventAggregator(
            group_by=group_by_zone(zone_map),
            filter=ObjectFilter(["no_helmet", "no_vest"])
        )
        self.realtime_window = SlidingWindowStats(
            window=timedelta(minutes=5),
            group_by=group_by_object
        )

        # 回調
        self.on_violation = on_violation
        self.on_alert = on_alert

        # 計數器
        self.total_events = 0
        self.violations = 0
        self.alerts = 0

    def process(self, event: PPEDetectionEvent) -> ProcessingResult:
        """處理單一事件"""
        self.total_events += 1

        # 1. 去重
        deduped = self.deduplicator.process(event)
        if deduped is None:
            return ProcessingResult(
                event=event,
                is_duplicate=True,
                action=Action.IGNORE,
                severity=None,
                reason="重複事件"
            )

        # 2. 判斷
        if self.hybrid_classifier:
            result = self.hybrid_classifier.classify(event)
            action = result["action"]
            severity = result["rule_result"].severity if result["rule_result"].matched else None
            reason = result["reason"]
            ml_prob = result["ml_result"]["probability"] if result["ml_result"] else None
        else:
            rule_result = self.rule_engine.evaluate(event)
            action = rule_result.action if rule_result.matched else Action.IGNORE
            severity = rule_result.severity if rule_result.matched else None
            reason = rule_result.message if rule_result.matched else "無匹配規則"
            ml_prob = None

        # 3. 更新統計
        self.stats_by_object.add(event)
        self.stats_by_zone.add(event)
        self.realtime_window.add(event)

        # 4. 建立結果
        result = ProcessingResult(
            event=event,
            is_duplicate=False,
            action=action,
            severity=severity,
            reason=reason,
            ml_probability=ml_prob
        )

        # 5. 觸發回調
        if action == Action.VIOLATION:
            self.violations += 1
            if self.on_violation:
                self.on_violation(result)

        if action == Action.ALERT:
            self.alerts += 1
            if self.on_alert:
                self.on_alert(result)

        return result

    def process_batch(self, events: list[PPEDetectionEvent]) -> list[ProcessingResult]:
        """批次處理"""
        return [self.process(event) for event in events]

    def get_stats(self) -> dict:
        """取得統計資料"""
        return {
            "summary": {
                "total_events": self.total_events,
                "violations": self.violations,
                "alerts": self.alerts,
            },
            "by_object": [
                {"object": r.group_key, "count": r.count}
                for r in self.stats_by_object.get_results()
            ],
            "by_zone": [
                {"zone": r.group_key, "count": r.count, **r.stats}
                for r in self.stats_by_zone.get_results()
            ],
            "realtime_5min": {
                "counts": self.realtime_window.get_counts_by_group(),
                "rate": self.realtime_window.get_rate(),
            }
        }

    def reset_stats(self) -> None:
        """重置統計"""
        self.stats_by_object.clear()
        self.stats_by_zone.clear()
        self.realtime_window.clear()
        self.total_events = 0
        self.violations = 0
        self.alerts = 0


# === 使用範例 ===
if __name__ == "__main__":
    print("=== 工安監控系統整合示範 ===\n")

    # 1. 設定區域
    zone_map = {
        "construction": [[0, 0, 400, 600]],
        "office": [[400, 0, 700, 600]],
        "entrance": [[700, 0, 800, 600]],
    }

    # 2. 設定規則
    rules = [
        Rule(
            id="r1",
            name="施工區安全帽",
            conditions=RuleConditions(object="no_helmet", zone="construction", confidence_gte=0.6),
            action=Action.VIOLATION,
            severity=Severity.HIGH,
            message="施工區未配戴安全帽",
            priority=10
        ),
        Rule(
            id="r2",
            name="施工區反光背心",
            conditions=RuleConditions(object="no_vest", zone="construction", confidence_gte=0.6),
            action=Action.VIOLATION,
            severity=Severity.MEDIUM,
            message="施工區未穿著反光背心",
            priority=10
        ),
        Rule(
            id="r3",
            name="入口區安全帽",
            conditions=RuleConditions(object="no_helmet", zone="entrance", confidence_gte=0.7),
            action=Action.WARNING,
            severity=Severity.LOW,
            message="入口區建議配戴安全帽",
            priority=5
        ),
        Rule(
            id="r4",
            name="低信心忽略",
            conditions=RuleConditions(confidence_lte=0.5),
            action=Action.IGNORE,
            priority=100
        ),
    ]

    # 3. 建立告警管理器
    alert_manager = AlertManager()

    def on_violation(result: ProcessingResult):
        alert = alert_manager.create_alert(result)
        print(f"  [VIOLATION] {result.event.object} - {result.reason}")

    def on_alert(result: ProcessingResult):
        alert = alert_manager.create_alert(result)
        print(f"  [ALERT] {result.event.object} - {result.reason}")

    # 4. 建立管線
    pipeline = SafetyMonitoringPipeline(
        rules=rules,
        zone_map=zone_map,
        on_violation=on_violation,
        on_alert=on_alert,
    )

    # 5. 模擬 YOLO 輸入
    simulator = YOLOSimulator(
        frame_width=800,
        frame_height=600,
        violation_rate=0.4
    )

    print("模擬 10 個影格的偵測...\n")

    for frame in range(10):
        events = simulator.detect(num_objects=3)
        print(f"Frame {frame + 1}: {len(events)} 個偵測")

        for event in events:
            result = pipeline.process(event)

    # 6. 輸出統計
    print("\n=== 統計報告 ===")
    stats = pipeline.get_stats()

    print(f"\n總覽:")
    print(f"  總事件: {stats['summary']['total_events']}")
    print(f"  違規: {stats['summary']['violations']}")
    print(f"  告警: {stats['summary']['alerts']}")

    print(f"\n依物件類別:")
    for item in stats['by_object']:
        print(f"  {item['object']}: {item['count']}")

    print(f"\n依區域 (違規):")
    for item in stats['by_zone']:
        print(f"  {item['zone']}: {item['count']} (avg_conf={item.get('avg_confidence', 0):.2f})")

    print(f"\n即時統計 (5分鐘):")
    print(f"  {stats['realtime_5min']['counts']}")
    print(f"  速率: {stats['realtime_5min']['rate']:.1f} 次/分鐘")

    # 7. LLM 預覽
    print("\n=== LLM 介面預覽 ===")
    llm = SimpleLLMPlaceholder()

    unack_alerts = alert_manager.get_unacknowledged()
    if unack_alerts:
        alert = unack_alerts[0]
        print(f"\n告警訊息: {llm.generate_alert_message(alert)}")
        print(f"違規描述: {llm.describe_violation(alert.event)}")
        print(f"相關法規: {llm.suggest_regulations(alert.event.object)}")
