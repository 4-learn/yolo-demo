"""
解答：系統整合 — 從偵測結果到分析報告

把前面學的模組串起來：
  偵測結果 → 規則比對 → pandas 分析 → 告警報告

執行方式：
  python solution.py
"""

import yaml
import json
import pandas as pd


# === 第一步：規則比對（來自 rule_engine） ===

def load_rules(yaml_path):
    """讀取 YAML 規則檔"""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def check_event(event, rules):
    """用規則比對一筆事件，回傳第一條符合的規則"""
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


# === 第二步：pandas 分析（來自 pandas_analysis） ===

def analyze(results):
    """用 pandas 做統計分析"""
    df = pd.DataFrame(results)

    print("=== 資料總覽 ===")
    print(f"總筆數: {len(df)}")
    print()

    # 各 action 數量
    print("=== 各 action 數量 ===")
    print(df["action"].value_counts())
    print()

    # 每張圖統計
    print("=== 每張圖的事件數 ===")
    print(df.groupby("source_image")["action"].count())
    print()

    # 違規事件
    violations = df[df["action"].isin(["alert", "warning"])]
    print("=== 違規事件 ===")
    print(violations[["source_image", "event_type", "confidence", "action"]])
    print(f"\n共 {len(violations)} 筆違規")
    print()

    # 信心度統計
    print("=== 各類型平均信心度 ===")
    print(df.groupby("event_type")["confidence"].mean())
    print()

    return df


# === 第三步：聚合告警（N 張圖門檻） ===

def aggregate_alerts(df, threshold):
    """統計有 alert 的圖片，判斷是否發佈"""
    alerts = df[df["action"] == "alert"]
    alert_images = alerts["source_image"].unique().tolist()

    should_alert = len(alert_images) >= threshold

    return {
        "alert_images": alert_images,
        "alert_image_count": len(alert_images),
        "should_alert": should_alert,
        "threshold": threshold,
    }


# === 第四步：產生報告 ===

def generate_report(df, agg):
    """產生最終報告"""
    print("=" * 50)
    print("          工安監控系統 — 分析報告")
    print("=" * 50)

    # 總覽
    total = len(df)
    alerts = len(df[df["action"] == "alert"])
    warnings = len(df[df["action"] == "warning"])
    ok = len(df[df["action"] == "ok"])

    print(f"\n總偵測事件: {total}")
    print(f"  alert:   {alerts}")
    print(f"  warning: {warnings}")
    print(f"  ok:      {ok}")

    # 圖片層級
    print(f"\n有 alert 的圖片: {agg['alert_images']}")
    print(f"門檻: {agg['threshold']} 張")

    if agg["should_alert"]:
        print(f"\n🚨 發佈 ALERT：{agg['alert_image_count']} 張圖有違規（>= {agg['threshold']}）")
    else:
        print(f"\n✅ 不發佈：只有 {agg['alert_image_count']} 張圖有 alert，未達門檻")

    # 建議
    print("\n--- 建議 ---")
    if alerts > 0:
        top_image = df[df["action"] == "alert"]["source_image"].value_counts().index[0]
        print(f"  最多違規的圖片: {top_image}")
        print(f"  建議優先處理該區域的安全帽佩戴狀況")
    else:
        print("  無違規，持續監控中")

    print()


# === 主程式：串起來 ===

if __name__ == "__main__":
    # 1. 載入原始偵測結果
    with open("../rule_engine/sample_results.json") as f:
        events = json.load(f)
    print(f"讀取 {len(events)} 筆偵測事件\n")

    # 2. 載入規則
    config = load_rules("../rule_engine/rules.yaml")
    print(f"載入 {len(config['rules'])} 條規則")
    print(f"聚合門檻: {config['alert_threshold']} 張圖\n")

    # 3. 規則比對
    results = process_events(events, config)
    print(f"完成規則比對: {len(results)} 筆\n")

    # 4. pandas 分析
    df = analyze(results)

    # 5. 聚合告警
    agg = aggregate_alerts(df, config["alert_threshold"])

    # 6. 產生報告
    generate_report(df, agg)
