"""
Event Schema 示範

這個範例展示如何：
1. 使用 Pydantic 定義 Event 結構
2. 從 YOLO 輸出轉換成標準 Event
3. 過濾低信心的偵測結果
"""

from schema import PPEDetectionEvent
from converter import dict_to_events, PPE_CLASS_NAMES


def main():
    print("=" * 60)
    print("Event Schema 示範")
    print("=" * 60)

    # === 1. 直接建立 Event ===
    print("\n【1】直接建立 PPEDetectionEvent\n")

    event = PPEDetectionEvent(
        source="camera_01",
        object="no_helmet",
        confidence=0.87,
        bbox=[120.5, 80.3, 250.2, 320.1]
    )

    print(event.model_dump_json(indent=2))

    # === 2. 從 YOLO 輸出轉換 ===
    print("\n" + "=" * 60)
    print("【2】從 YOLO 輸出轉換成 Event")
    print("=" * 60)

    # 模擬 YOLO 輸出
    yolo_output = [
        {"class_id": 0, "confidence": 0.92, "bbox": [100, 50, 200, 300]},
        {"class_id": 2, "confidence": 0.78, "bbox": [150, 60, 220, 280]},
        {"class_id": 4, "confidence": 0.35, "bbox": [300, 100, 400, 350]},
    ]

    print("\n原始 YOLO 輸出：")
    for det in yolo_output:
        class_name = PPE_CLASS_NAMES.get(det["class_id"], "unknown")
        print(f"  - {class_name}: confidence={det['confidence']}")

    # 轉換（不過濾）
    print("\n轉換後（所有結果）：")
    events = dict_to_events(yolo_output, source="camera_01")
    for event in events:
        print(f"  - {event.object}: confidence={event.confidence}")

    # === 3. 過濾低信心結果 ===
    print("\n" + "=" * 60)
    print("【3】過濾低信心結果（threshold=0.5）")
    print("=" * 60)

    filtered_events = dict_to_events(
        yolo_output,
        source="camera_01",
        min_confidence=0.5
    )

    print(f"\n過濾後剩 {len(filtered_events)} 筆事件：")
    for event in filtered_events:
        print(f"\n{event.model_dump_json(indent=2)}")


if __name__ == "__main__":
    main()
