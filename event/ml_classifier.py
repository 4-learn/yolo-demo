"""
ML-based 事件分類器

使用 scikit-learn 訓練模型來判斷事件是否為真正的違規。
"""

import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from .schema import PPEDetectionEvent


# 特徵名稱
FEATURE_NAMES = [
    "confidence",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_ratio",
    "bbox_center_x",
    "bbox_center_y",
    "hour_of_day",
    "is_weekend",
]


def extract_features(event: PPEDetectionEvent) -> np.ndarray:
    """
    從事件中提取特徵

    Args:
        event: 偵測事件

    Returns:
        特徵向量
    """
    # bbox 相關特徵
    x1, y1, x2, y2 = event.bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    aspect_ratio = width / height if height > 0 else 0
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # 時間相關特徵
    hour = event.timestamp.hour
    is_weekend = 1 if event.timestamp.weekday() >= 5 else 0

    return np.array([
        event.confidence,
        width,
        height,
        area,
        aspect_ratio,
        center_x,
        center_y,
        hour,
        is_weekend,
    ])


def prepare_dataset(labeled_events: list[dict]) -> tuple:
    """
    準備訓練資料集

    Args:
        labeled_events: [{"event": Event, "is_violation": bool}, ...]

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = []
    y = []

    for item in labeled_events:
        features = extract_features(item["event"])
        X.append(features)
        y.append(1 if item["is_violation"] else 0)

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)


class ViolationClassifier:
    """
    違規事件分類器

    使用 Random Forest 判斷事件是否為真正的違規。
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        訓練模型

        Args:
            X: 特徵矩陣
            y: 標籤向量

        Returns:
            訓練結果統計
        """
        # 標準化特徵
        X_scaled = self.scaler.fit_transform(X)

        # 訓練模型
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # 計算訓練準確率
        train_acc = self.model.score(X_scaled, y)

        return {
            "train_accuracy": train_acc,
            "n_samples": len(y),
            "n_features": X.shape[1],
        }

    def predict(self, event: PPEDetectionEvent) -> dict:
        """
        預測事件是否為違規

        Args:
            event: 偵測事件

        Returns:
            {
                "is_violation": bool,
                "probability": float,
                "features": dict
            }
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未訓練")

        # 提取特徵
        features = extract_features(event)
        features_scaled = self.scaler.transform([features])

        # 預測
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        return {
            "is_violation": bool(prediction),
            "probability": float(probability[1]),  # 違規的機率
            "features": dict(zip(FEATURE_NAMES, features.tolist())),
        }

    def predict_batch(self, events: list[PPEDetectionEvent]) -> list[dict]:
        """批次預測"""
        return [self.predict(event) for event in events]

    def get_feature_importance(self) -> dict:
        """取得特徵重要性"""
        if not self.is_trained:
            raise RuntimeError("模型尚未訓練")

        importances = self.model.feature_importances_
        return dict(zip(FEATURE_NAMES, importances.tolist()))


def evaluate_model(
    classifier: ViolationClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    評估模型效能

    Args:
        classifier: 分類器
        X_test: 測試特徵
        y_test: 測試標籤

    Returns:
        評估指標字典
    """
    # 預測
    X_scaled = classifier.scaler.transform(X_test)
    y_pred = classifier.model.predict(X_scaled)

    # 計算指標
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# === 模型持久化 ===

def save_classifier(classifier: ViolationClassifier, path: str) -> None:
    """儲存分類器"""
    import joblib
    joblib.dump({
        "scaler": classifier.scaler,
        "model": classifier.model,
    }, path)


def load_classifier(path: str) -> ViolationClassifier:
    """載入分類器"""
    import joblib
    data = joblib.load(path)

    classifier = ViolationClassifier()
    classifier.scaler = data["scaler"]
    classifier.model = data["model"]
    classifier.is_trained = True

    return classifier


# === 使用範例 ===
if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta
    import random

    print("=== ML 分類器示範 ===\n")

    # 產生模擬資料
    print("產生模擬訓練資料...")

    labeled_events = []
    base_time = datetime.now(timezone.utc)

    for i in range(1000):
        # 隨機事件
        confidence = random.uniform(0.5, 0.99)
        width = random.uniform(50, 200)
        height = random.uniform(100, 400)
        x1 = random.uniform(0, 600)
        y1 = random.uniform(0, 400)

        event = PPEDetectionEvent(
            timestamp=base_time + timedelta(hours=random.randint(0, 23)),
            object="no_helmet",
            confidence=confidence,
            bbox=[x1, y1, x1 + width, y1 + height],
            source="camera_01"
        )

        # 標籤邏輯：高信心 + 合理大小 = 真違規
        is_violation = (
            confidence > 0.7 and
            width > 80 and
            height > 150 and
            random.random() > 0.1  # 加入一些噪音
        )

        labeled_events.append({
            "event": event,
            "is_violation": is_violation
        })

    # 準備資料集
    X_train, X_test, y_train, y_test = prepare_dataset(labeled_events)
    print(f"訓練樣本: {len(X_train)}, 測試樣本: {len(X_test)}")

    # 訓練模型
    print("\n訓練模型...")
    classifier = ViolationClassifier()
    train_result = classifier.train(X_train, y_train)
    print(f"訓練準確率: {train_result['train_accuracy']:.3f}")

    # 評估模型
    print("\n評估模型...")
    eval_result = evaluate_model(classifier, X_test, y_test)
    print(f"Accuracy:  {eval_result['accuracy']:.3f}")
    print(f"Precision: {eval_result['precision']:.3f}")
    print(f"Recall:    {eval_result['recall']:.3f}")
    print(f"F1 Score:  {eval_result['f1']:.3f}")

    # 特徵重要性
    print("\n特徵重要性:")
    importance = classifier.get_feature_importance()
    for name, score in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"  {name:20} {bar} {score:.3f}")

    # 預測單一事件
    print("\n單一事件預測:")
    test_event = PPEDetectionEvent(
        object="no_helmet",
        confidence=0.85,
        bbox=[100, 100, 200, 350],
        source="camera_01"
    )
    result = classifier.predict(test_event)
    print(f"  預測: {'違規' if result['is_violation'] else '非違規'}")
    print(f"  機率: {result['probability']:.3f}")
