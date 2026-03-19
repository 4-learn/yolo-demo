"""
產生模擬的偵測結果（100 筆），用於 KMeans 分群練習

三群分佈：
  低信心: 0.35~0.55（雜訊、誤判）
  中信心: 0.60~0.78（不確定）
  高信心: 0.80~0.95（可信偵測）
"""

import json
import random
from datetime import datetime, timezone, timedelta

random.seed(42)

images = [f"frame_{i:03d}.jpg" for i in range(1, 21)]
events = []

base_time = datetime(2026, 3, 19, 9, 0, 0, tzinfo=timezone.utc)

for i in range(100):
    # 隨機選一群
    group = random.choices(["low", "mid", "high"], weights=[20, 35, 45])[0]

    if group == "low":
        confidence = round(random.uniform(0.35, 0.55), 4)
    elif group == "mid":
        confidence = round(random.uniform(0.60, 0.78), 4)
    else:
        confidence = round(random.uniform(0.80, 0.95), 4)

    # head 或 helmet
    if random.random() < 0.35:
        event_type = "head_detected"
    else:
        event_type = "helmet_detected"

    events.append({
        "source_image": random.choice(images),
        "event_type": event_type,
        "confidence": confidence,
        "rule": "-",
        "action": "-",
        "severity": "-",
    })

with open("rule_results_large.json", "w") as f:
    json.dump(events, f, indent=2, ensure_ascii=False)

print(f"產生 {len(events)} 筆事件")

# 印出分佈
low = [e["confidence"] for e in events if e["confidence"] < 0.55]
mid = [e["confidence"] for e in events if 0.55 <= e["confidence"] < 0.80]
high = [e["confidence"] for e in events if e["confidence"] >= 0.80]
print(f"  低信心 (<0.55): {len(low)} 筆")
print(f"  中信心 (0.55~0.80): {len(mid)} 筆")
print(f"  高信心 (>=0.80): {len(high)} 筆")
