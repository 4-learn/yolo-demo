"""
Event Schema 定義

定義 YOLO 偵測事件的標準結構，讓下游系統可以統一處理。
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid


class DetectionEvent(BaseModel):
    """
    通用偵測事件結構

    Attributes:
        event_id: 唯一識別碼
        timestamp: 事件發生時間 (UTC)
        event_type: 事件類型
        source: 來源（例如攝影機 ID）
        object: 偵測到的物件名稱
        confidence: 信心分數 (0-1)
        bbox: 邊界框 [x1, y1, x2, y2]
        metadata: 額外資訊
    """

    event_id: str = Field(
        default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}",
        description="唯一識別碼"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="事件發生時間 (UTC)"
    )
    event_type: str = Field(
        default="detection",
        description="事件類型"
    )
    source: Optional[str] = Field(
        default=None,
        description="來源（例如攝影機 ID）"
    )
    object: str = Field(
        description="偵測到的物件名稱"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="信心分數 (0-1)"
    )
    bbox: List[float] = Field(
        min_length=4,
        max_length=4,
        description="邊界框 [x1, y1, x2, y2]"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="額外資訊"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "event_id": "evt_a1b2c3d4e5f6",
                "timestamp": "2024-01-15T10:30:45.123456Z",
                "event_type": "detection",
                "source": "camera_01",
                "object": "person",
                "confidence": 0.92,
                "bbox": [100.0, 50.0, 200.0, 300.0],
                "metadata": {"class_id": 0}
            }
        }
    }


class PPEDetectionEvent(DetectionEvent):
    """
    PPE (個人防護裝備) 偵測事件

    繼承自 DetectionEvent，event_type 固定為 "ppe_detection"
    """

    event_type: str = Field(
        default="ppe_detection",
        description="事件類型（固定為 ppe_detection）"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "event_id": "evt_a1b2c3d4e5f6",
                "timestamp": "2024-01-15T10:30:45.123456Z",
                "event_type": "ppe_detection",
                "source": "camera_01",
                "object": "no_helmet",
                "confidence": 0.87,
                "bbox": [120.5, 80.3, 250.2, 320.1],
                "metadata": {"class_id": 2}
            }
        }
    }


# === 使用範例 ===
if __name__ == "__main__":
    # 建立一個 PPE 偵測事件
    event = PPEDetectionEvent(
        source="camera_01",
        object="no_helmet",
        confidence=0.87,
        bbox=[120.5, 80.3, 250.2, 320.1],
        metadata={"class_id": 2}
    )

    # 輸出 JSON
    print("=== PPEDetectionEvent ===")
    print(event.model_dump_json(indent=2))
