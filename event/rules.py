"""
規則引擎 (Rule Engine)

根據可配置的規則判斷事件應該採取什麼動作。
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from .schema import PPEDetectionEvent


class Severity(str, Enum):
    """嚴重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(str, Enum):
    """動作類型"""
    VIOLATION = "violation"  # 標記為違規
    WARNING = "warning"      # 標記為警告
    IGNORE = "ignore"        # 忽略（不處理）
    ALERT = "alert"          # 立即告警


class RuleConditions(BaseModel):
    """
    規則條件

    所有條件都是可選的，只有設定的條件才會被檢查。
    多個條件之間是 AND 關係。
    """
    object: Optional[str] = None           # 物件類別
    confidence_gte: Optional[float] = None # 信心分數 >=
    confidence_lte: Optional[float] = None # 信心分數 <=
    zone: Optional[str] = None             # 區域
    source: Optional[str] = None           # 攝影機來源
    time_start: Optional[str] = None       # 時間範圍開始 "HH:MM"
    time_end: Optional[str] = None         # 時間範圍結束 "HH:MM"


class Rule(BaseModel):
    """
    規則定義

    Attributes:
        id: 規則 ID
        name: 規則名稱
        conditions: 規則條件
        action: 匹配時的動作
        severity: 嚴重程度
        message: 訊息模板
        enabled: 是否啟用
        priority: 優先級（數字越大越優先）
    """
    id: str
    name: str
    conditions: RuleConditions
    action: Action = Action.VIOLATION
    severity: Severity = Severity.MEDIUM
    message: str = ""
    enabled: bool = True
    priority: int = 0


@dataclass
class RuleResult:
    """規則執行結果"""
    matched: bool
    rule: Optional[Rule] = None
    action: Optional[Action] = None
    severity: Optional[Severity] = None
    message: str = ""


class RuleMatcher:
    """
    規則比對器

    檢查事件是否符合規則條件
    """

    def __init__(self, zone_map: dict[str, list] = None):
        """
        Args:
            zone_map: 區域對應表 {zone_name: [[x1,y1,x2,y2], ...]}
        """
        self.zone_map = zone_map or {}

    def get_zone(self, bbox: list) -> Optional[str]:
        """
        根據 bbox 判斷所在區域

        使用 bbox 中心點判斷落在哪個區域。

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            區域名稱，如果不在任何區域則返回 None
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        for zone_name, regions in self.zone_map.items():
            for region in regions:
                if (region[0] <= cx <= region[2] and
                    region[1] <= cy <= region[3]):
                    return zone_name
        return None

    def match(self, event: PPEDetectionEvent, rule: Rule) -> bool:
        """
        檢查事件是否符合規則條件

        Args:
            event: 偵測事件
            rule: 規則

        Returns:
            True 如果所有條件都符合
        """
        cond = rule.conditions

        # 檢查物件類別
        if cond.object is not None and event.object != cond.object:
            return False

        # 檢查信心分數下限
        if cond.confidence_gte is not None and event.confidence < cond.confidence_gte:
            return False

        # 檢查信心分數上限
        if cond.confidence_lte is not None and event.confidence > cond.confidence_lte:
            return False

        # 檢查區域
        if cond.zone is not None:
            event_zone = self.get_zone(event.bbox)
            if event_zone != cond.zone:
                return False

        # 檢查來源
        if cond.source is not None and event.source != cond.source:
            return False

        # 檢查時間範圍
        if cond.time_start is not None and cond.time_end is not None:
            event_time = event.timestamp.strftime("%H:%M")
            # 處理跨日的時間範圍（如 18:00-06:00）
            if cond.time_start <= cond.time_end:
                if not (cond.time_start <= event_time <= cond.time_end):
                    return False
            else:
                # 跨日：18:00-06:00 表示 18:00-23:59 或 00:00-06:00
                if not (event_time >= cond.time_start or event_time <= cond.time_end):
                    return False

        return True


class RuleEngine:
    """
    規則引擎

    根據規則判斷事件應該採取什麼動作
    """

    def __init__(self, rules: list[Rule], zone_map: dict = None):
        """
        Args:
            rules: 規則列表
            zone_map: 區域對應表
        """
        # 按優先級排序（高優先級在前）
        self.rules = sorted(rules, key=lambda r: -r.priority)
        self.matcher = RuleMatcher(zone_map)

    def evaluate(self, event: PPEDetectionEvent) -> RuleResult:
        """
        評估事件，返回第一個匹配的規則結果

        使用此方法時，規則是互斥的（只返回最高優先級的匹配）。

        Args:
            event: 偵測事件

        Returns:
            RuleResult
        """
        for rule in self.rules:
            if not rule.enabled:
                continue

            if self.matcher.match(event, rule):
                return RuleResult(
                    matched=True,
                    rule=rule,
                    action=rule.action,
                    severity=rule.severity,
                    message=rule.message or f"違反規則: {rule.name}"
                )

        return RuleResult(matched=False)

    def evaluate_all(self, event: PPEDetectionEvent) -> list[RuleResult]:
        """
        評估事件，返回所有匹配的規則結果

        使用此方法時，規則可以疊加（返回所有匹配的規則）。

        Args:
            event: 偵測事件

        Returns:
            所有匹配的 RuleResult 列表
        """
        results = []
        for rule in self.rules:
            if not rule.enabled:
                continue

            if self.matcher.match(event, rule):
                results.append(RuleResult(
                    matched=True,
                    rule=rule,
                    action=rule.action,
                    severity=rule.severity,
                    message=rule.message or f"違反規則: {rule.name}"
                ))

        return results

    def add_rule(self, rule: Rule) -> None:
        """新增規則"""
        self.rules.append(rule)
        self.rules = sorted(self.rules, key=lambda r: -r.priority)

    def remove_rule(self, rule_id: str) -> bool:
        """移除規則"""
        original_len = len(self.rules)
        self.rules = [r for r in self.rules if r.id != rule_id]
        return len(self.rules) < original_len

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """取得規則"""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None


# === 使用範例 ===
if __name__ == "__main__":
    from datetime import timezone

    # 區域定義
    zone_map = {
        "construction": [[0, 0, 400, 600]],      # 施工區
        "office": [[400, 0, 800, 600]],          # 辦公區
        "entrance": [[350, 500, 450, 600]],      # 入口區
    }

    # 規則定義
    rules = [
        Rule(
            id="rule_001",
            name="施工區必須戴安全帽",
            conditions=RuleConditions(
                object="no_helmet",
                zone="construction",
                confidence_gte=0.7
            ),
            action=Action.VIOLATION,
            severity=Severity.HIGH,
            message="施工區域未配戴安全帽",
            priority=10
        ),
        Rule(
            id="rule_002",
            name="施工區必須穿反光背心",
            conditions=RuleConditions(
                object="no_vest",
                zone="construction",
                confidence_gte=0.7
            ),
            action=Action.VIOLATION,
            severity=Severity.MEDIUM,
            message="施工區域未穿著反光背心",
            priority=10
        ),
        Rule(
            id="rule_003",
            name="辦公區不強制安全帽",
            conditions=RuleConditions(
                object="no_helmet",
                zone="office"
            ),
            action=Action.IGNORE,
            message="辦公區不強制安全帽",
            priority=5
        ),
        Rule(
            id="rule_004",
            name="低信心偵測忽略",
            conditions=RuleConditions(
                confidence_lte=0.5
            ),
            action=Action.IGNORE,
            message="信心分數過低，忽略",
            priority=100  # 最高優先級
        ),
    ]

    # 建立規則引擎
    engine = RuleEngine(rules, zone_map)

    print("=== 規則引擎示範 ===\n")

    # 測試事件
    test_events = [
        ("施工區沒戴安全帽", PPEDetectionEvent(
            object="no_helmet",
            confidence=0.85,
            bbox=[100, 100, 200, 300],
            source="camera_01"
        )),
        ("辦公區沒戴安全帽", PPEDetectionEvent(
            object="no_helmet",
            confidence=0.9,
            bbox=[500, 100, 600, 300],
            source="camera_01"
        )),
        ("低信心偵測", PPEDetectionEvent(
            object="no_helmet",
            confidence=0.4,
            bbox=[100, 100, 200, 300],
            source="camera_01"
        )),
    ]

    for desc, event in test_events:
        result = engine.evaluate(event)
        print(f"測試: {desc}")
        if result.matched:
            print(f"  規則: {result.rule.name}")
            print(f"  動作: {result.action.value}")
            print(f"  嚴重: {result.severity.value}")
            print(f"  訊息: {result.message}")
        else:
            print(f"  無匹配規則")
        print()
