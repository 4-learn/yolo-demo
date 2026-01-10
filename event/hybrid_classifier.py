"""
混合分類器 (Hybrid Classifier)

結合 Rule-based 和 ML-based 判斷。
"""

from typing import Optional

from .schema import PPEDetectionEvent
from .rules import RuleEngine, RuleResult, Action
from .ml_classifier import ViolationClassifier


class HybridClassifier:
    """
    混合分類器

    結合 Rule-based 和 ML-based 判斷，策略：
    1. 先用規則判斷
    2. 如果規則說 IGNORE，直接忽略
    3. 如果規則說 VIOLATION，用 ML 確認
    4. ML 信心 < threshold 則降級為 WARNING
    """

    def __init__(
        self,
        rule_engine: RuleEngine,
        ml_classifier: ViolationClassifier,
        ml_threshold: float = 0.7
    ):
        """
        Args:
            rule_engine: 規則引擎
            ml_classifier: ML 分類器
            ml_threshold: ML 信心閾值
        """
        self.rule_engine = rule_engine
        self.ml_classifier = ml_classifier
        self.ml_threshold = ml_threshold

    def classify(self, event: PPEDetectionEvent) -> dict:
        """
        分類事件

        Args:
            event: 偵測事件

        Returns:
            {
                "action": Action,
                "rule_result": RuleResult,
                "ml_result": dict | None,
                "reason": str
            }
        """
        # 1. 規則判斷
        rule_result = self.rule_engine.evaluate(event)

        # 2. IGNORE 直接返回
        if rule_result.matched and rule_result.action == Action.IGNORE:
            return {
                "action": Action.IGNORE,
                "rule_result": rule_result,
                "ml_result": None,
                "reason": "規則判定忽略"
            }

        # 3. ALERT 直接返回（不需要 ML 確認）
        if rule_result.matched and rule_result.action == Action.ALERT:
            return {
                "action": Action.ALERT,
                "rule_result": rule_result,
                "ml_result": None,
                "reason": "規則判定立即告警"
            }

        # 4. VIOLATION 用 ML 確認
        if rule_result.matched and rule_result.action == Action.VIOLATION:
            ml_result = self.ml_classifier.predict(event)

            if ml_result["probability"] >= self.ml_threshold:
                return {
                    "action": Action.VIOLATION,
                    "rule_result": rule_result,
                    "ml_result": ml_result,
                    "reason": f"規則+ML 確認違規 (p={ml_result['probability']:.2f})"
                }
            else:
                return {
                    "action": Action.WARNING,
                    "rule_result": rule_result,
                    "ml_result": ml_result,
                    "reason": f"ML 信心不足，降級為警告 (p={ml_result['probability']:.2f})"
                }

        # 5. WARNING 也用 ML 確認
        if rule_result.matched and rule_result.action == Action.WARNING:
            ml_result = self.ml_classifier.predict(event)

            if ml_result["probability"] >= self.ml_threshold:
                return {
                    "action": Action.WARNING,
                    "rule_result": rule_result,
                    "ml_result": ml_result,
                    "reason": f"規則警告 + ML 確認 (p={ml_result['probability']:.2f})"
                }
            else:
                return {
                    "action": Action.IGNORE,
                    "rule_result": rule_result,
                    "ml_result": ml_result,
                    "reason": f"ML 信心不足，忽略 (p={ml_result['probability']:.2f})"
                }

        # 6. 規則未匹配，用 ML 判斷
        ml_result = self.ml_classifier.predict(event)
        if ml_result["probability"] >= self.ml_threshold:
            return {
                "action": Action.WARNING,
                "rule_result": rule_result,
                "ml_result": ml_result,
                "reason": f"規則未匹配，ML 偵測到可能違規 (p={ml_result['probability']:.2f})"
            }

        return {
            "action": Action.IGNORE,
            "rule_result": rule_result,
            "ml_result": ml_result,
            "reason": "無違規"
        }

    def classify_batch(self, events: list[PPEDetectionEvent]) -> list[dict]:
        """批次分類"""
        return [self.classify(event) for event in events]


# === 使用範例 ===
if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta
    import random
    import numpy as np

    from .rules import Rule, RuleConditions, Severity
    from .ml_classifier import prepare_dataset

    print("=== 混合分類器示範 ===\n")

    # 建立規則引擎
    zone_map = {
        "construction": [[0, 0, 400, 600]],
        "office": [[400, 0, 800, 600]],
    }

    rules = [
        Rule(
            id="rule_001",
            name="施工區必須戴安全帽",
            conditions=RuleConditions(
                object="no_helmet",
                zone="construction",
                confidence_gte=0.6
            ),
            action=Action.VIOLATION,
            severity=Severity.HIGH,
            message="施工區域未配戴安全帽",
            priority=10
        ),
        Rule(
            id="rule_002",
            name="辦公區不強制安全帽",
            conditions=RuleConditions(
                object="no_helmet",
                zone="office"
            ),
            action=Action.IGNORE,
            priority=5
        ),
        Rule(
            id="rule_003",
            name="低信心偵測忽略",
            conditions=RuleConditions(
                confidence_lte=0.5
            ),
            action=Action.IGNORE,
            priority=100
        ),
    ]

    rule_engine = RuleEngine(rules, zone_map)

    # 訓練 ML 分類器
    print("訓練 ML 分類器...")

    labeled_events = []
    base_time = datetime.now(timezone.utc)

    for i in range(500):
        confidence = random.uniform(0.5, 0.99)
        width = random.uniform(50, 200)
        height = random.uniform(100, 400)
        x1 = random.uniform(0, 300)  # 施工區
        y1 = random.uniform(0, 400)

        event = PPEDetectionEvent(
            timestamp=base_time,
            object="no_helmet",
            confidence=confidence,
            bbox=[x1, y1, x1 + width, y1 + height],
            source="camera_01"
        )

        is_violation = confidence > 0.7 and width > 80 and height > 150

        labeled_events.append({
            "event": event,
            "is_violation": is_violation
        })

    X_train, X_test, y_train, y_test = prepare_dataset(labeled_events)
    ml_classifier = ViolationClassifier()
    ml_classifier.train(X_train, y_train)
    print("ML 分類器訓練完成\n")

    # 建立混合分類器
    hybrid = HybridClassifier(
        rule_engine=rule_engine,
        ml_classifier=ml_classifier,
        ml_threshold=0.7
    )

    # 測試事件
    test_events = [
        ("施工區高信心違規", PPEDetectionEvent(
            object="no_helmet",
            confidence=0.9,
            bbox=[100, 100, 250, 400],
            source="camera_01"
        )),
        ("施工區低信心（ML 可能降級）", PPEDetectionEvent(
            object="no_helmet",
            confidence=0.65,
            bbox=[100, 100, 130, 200],  # 小 bbox
            source="camera_01"
        )),
        ("辦公區違規（規則忽略）", PPEDetectionEvent(
            object="no_helmet",
            confidence=0.9,
            bbox=[500, 100, 650, 400],
            source="camera_01"
        )),
        ("低信心偵測", PPEDetectionEvent(
            object="no_helmet",
            confidence=0.4,
            bbox=[100, 100, 250, 400],
            source="camera_01"
        )),
    ]

    for desc, event in test_events:
        result = hybrid.classify(event)
        print(f"測試: {desc}")
        print(f"  動作: {result['action'].value}")
        print(f"  原因: {result['reason']}")
        if result['ml_result']:
            print(f"  ML 信心: {result['ml_result']['probability']:.2f}")
        print()
