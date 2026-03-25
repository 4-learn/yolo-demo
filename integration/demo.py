"""
Demo：系統整合 — 新舊門檻比較

用 sklearn 算出的新門檻 vs 人工設的舊門檻，
跑同一批資料，比較結果差異。

執行方式：
  python demo.py

需要的檔案（來自前面章節）：
  ../rule_engine/sample_results.json  ← 偵測結果
  ../rule_engine/rules.yaml           ← 舊門檻（人工）
  ../sklearn_clustering/rules_suggested.yaml  ← 新門檻（sklearn）

如果沒有 rules_suggested.yaml，會先跑 KMeans 產生。
"""

import yaml
import json
import os
import pandas as pd
from sklearn.cluster import KMeans


# === 規則比對 ===

def load_rules(yaml_path):
    """讀取 YAML 規則檔"""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def check_event(event, rules):
    """用規則比對一筆事件"""
    for rule in rules:
        if "event_type" in rule:
            if event["event_type"] != rule["event_type"]:
                continue
        if "min_confidence" in rule:
            if event["confidence"] < rule["min_confidence"]:
                continue
        if "max_confidence" in rule:
            if event["confidence"] > rule["max_confidence"]:
                continue
        return rule
    return None


def process_events(events, config):
    """對所有事件跑規則比對"""
    rules = config["rules"]
    results = []
    for event in events:
        matched = check_event(event, rules)
        results.append({
            "source_image": event["source_image"],
            "event_type": event["event_type"],
            "confidence": event["confidence"],
            "rule": matched["name"] if matched else "無匹配",
            "action": matched["action"] if matched else "unknown",
            "severity": matched.get("severity", "-"),
        })
    return results


# === pandas 統計 ===

def summarize(results):
    """回傳統計摘要（不印出）"""
    df = pd.DataFrame(results)
    return {
        "total": len(df),
        "alert": int(df[df["action"] == "alert"].shape[0]),
        "warning": int(df[df["action"] == "warning"].shape[0]),
        "ok": int(df[df["action"] == "ok"].shape[0]),
        "ignore": int(df[df["action"] == "ignore"].shape[0]),
        "alert_images": df[df["action"] == "alert"]["source_image"].unique().tolist(),
    }


# === 產生 rules_suggested.yaml（如果不存在） ===

def generate_suggested_rules(events, output_path):
    """用 KMeans 產生建議門檻的 rules.yaml"""
    df = pd.DataFrame(events)
    X = df[["confidence"]]

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    centers = sorted(kmeans.cluster_centers_.flatten())

    threshold_low = round(float((centers[0] + centers[1]) / 2), 4)
    threshold_high = round(float((centers[1] + centers[2]) / 2), 4)

    rules = {
        "rules": [
            {"name": "低信心忽略", "max_confidence": threshold_low, "action": "ignore"},
            {"name": "沒戴安全帽（高信心）", "event_type": "head_detected",
             "min_confidence": threshold_high, "action": "alert", "severity": "high"},
            {"name": "沒戴安全帽（中信心）", "event_type": "head_detected",
             "action": "warning", "severity": "low"},
            {"name": "預設", "action": "ok"},
        ],
        "alert_threshold": 2,
    }

    with open(output_path, "w") as f:
        yaml.dump(rules, f, allow_unicode=True, default_flow_style=False)

    print(f"  KMeans 群中心: {[round(float(c), 4) for c in centers]}")
    print(f"  建議門檻: {threshold_low} / {threshold_high}")
    print(f"  已寫入 {output_path}\n")

    return threshold_low, threshold_high


# === 主程式：新舊門檻比較 ===

if __name__ == "__main__":
    print("=" * 55)
    print("  系統整合 — 新舊門檻比較")
    print("=" * 55)

    # 1. 載入偵測結果
    sample_path = "../rule_engine/sample_results.json"
    with open(sample_path) as f:
        events = json.load(f)
    print(f"\n讀取 {len(events)} 筆偵測事件\n")

    # 2. 如果沒有 rules_suggested.yaml，先跑 KMeans 產生
    suggested_path = "../sklearn_clustering/rules_suggested.yaml"
    if not os.path.exists(suggested_path):
        print("找不到 rules_suggested.yaml，用 KMeans 產生...")
        generate_suggested_rules(events, suggested_path)

    # 3. 載入兩份 rules
    old_config = load_rules("../rule_engine/rules.yaml")
    new_config = load_rules(suggested_path)

    # 4. 分別跑規則比對
    old_results = process_events(events, old_config)
    new_results = process_events(events, new_config)

    # 5. 統計
    old_stats = summarize(old_results)
    new_stats = summarize(new_results)

    # 6. 比較報告
    print("=" * 55)
    print("  比較結果")
    print("=" * 55)

    print(f"\n{'':>20} {'舊門檻':>10} {'新門檻':>10} {'差異':>10}")
    print("-" * 55)

    for key in ["alert", "warning", "ok", "ignore"]:
        old_val = old_stats[key]
        new_val = new_stats[key]
        diff = new_val - old_val
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{key:>20} {old_val:>10} {new_val:>10} {diff_str:>10}")

    print("-" * 55)
    print(f"{'total':>20} {old_stats['total']:>10} {new_stats['total']:>10}")

    # 7. alert 圖片比較
    print(f"\n舊門檻 alert 圖片: {old_stats['alert_images']}")
    print(f"新門檻 alert 圖片: {new_stats['alert_images']}")

    # 8. 結論
    old_alert = old_stats["alert"]
    new_alert = new_stats["alert"]

    print(f"\n--- 結論 ---")
    if new_alert < old_alert:
        print(f"  新門檻較嚴格：alert 從 {old_alert} 降到 {new_alert}，減少 {old_alert - new_alert} 筆誤報")
    elif new_alert > old_alert:
        print(f"  新門檻較寬鬆：alert 從 {old_alert} 增到 {new_alert}")
    else:
        print(f"  alert 數量相同（{old_alert} 筆），但門檻有依據了")

    print()
