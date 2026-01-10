"""
YOLO 輸出轉換器

將 YOLO 的偵測結果轉換成標準的 Event JSON 結構。
"""

from typing import Optional
from .schema import PPEDetectionEvent


# PPE 偵測類別對應表
PPE_CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "no_helmet",
    3: "vest",
    4: "no_vest"
}


def yolo_to_events(
    results,
    source: Optional[str] = None,
    class_names: dict = None,
    min_confidence: float = 0.0
) -> list[PPEDetectionEvent]:
    """
    將 YOLO 結果轉換成 PPEDetectionEvent 列表

    Args:
        results: YOLO 的 Results 物件
        source: 攝影機來源 ID
        class_names: 類別對應表（如果不提供，使用預設的 PPE_CLASS_NAMES）
        min_confidence: 最低信心閾值（預設 0，不過濾）

    Returns:
        PPEDetectionEvent 列表
    """
    if class_names is None:
        class_names = PPE_CLASS_NAMES

    events = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

        # 過濾低信心結果
        if conf < min_confidence:
            continue

        event = PPEDetectionEvent(
            source=source,
            object=class_names.get(cls_id, f"unknown_{cls_id}"),
            confidence=conf,
            bbox=bbox,
            metadata={"class_id": cls_id}
        )
        events.append(event)

    return events


def dict_to_events(
    detections: list[dict],
    source: Optional[str] = None,
    class_names: dict = None,
    min_confidence: float = 0.0
) -> list[PPEDetectionEvent]:
    """
    將字典格式的偵測結果轉換成 PPEDetectionEvent 列表

    這個函式用於測試或處理已經轉成字典的 YOLO 輸出。

    Args:
        detections: 偵測結果字典列表，每個字典包含：
            - class_id: int
            - confidence: float
            - bbox: list[float]
        source: 攝影機來源 ID
        class_names: 類別對應表
        min_confidence: 最低信心閾值

    Returns:
        PPEDetectionEvent 列表
    """
    if class_names is None:
        class_names = PPE_CLASS_NAMES

    events = []

    for detection in detections:
        cls_id = detection["class_id"]
        conf = detection["confidence"]
        bbox = detection["bbox"]

        # 過濾低信心結果
        if conf < min_confidence:
            continue

        event = PPEDetectionEvent(
            source=source,
            object=class_names.get(cls_id, f"unknown_{cls_id}"),
            confidence=conf,
            bbox=bbox,
            metadata={"class_id": cls_id}
        )
        events.append(event)

    return events


# === 使用範例 ===
if __name__ == "__main__":
    # 模擬 YOLO 輸出（字典格式）
    mock_detections = [
        {"class_id": 0, "confidence": 0.92, "bbox": [100, 50, 200, 300]},
        {"class_id": 2, "confidence": 0.78, "bbox": [150, 60, 220, 280]},
        {"class_id": 4, "confidence": 0.35, "bbox": [300, 100, 400, 350]},
    ]

    print("=== 所有偵測結果 ===")
    events = dict_to_events(mock_detections, source="camera_01")
    for event in events:
        print(event.model_dump_json(indent=2))
        print()

    print("=== 過濾後（confidence >= 0.5）===")
    filtered_events = dict_to_events(
        mock_detections,
        source="camera_01",
        min_confidence=0.5
    )
    print(f"共 {len(filtered_events)} 筆事件")
    for event in filtered_events:
        print(event.model_dump_json(indent=2))
        print()
