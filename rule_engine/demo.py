import json
import yaml


def load_rules(path):
    with open(path) as f:
        return yaml.safe_load(f)


def check_event(event, rules):
    for rule in rules:
        if "event_type" in rule and event["event_type"] != rule["event_type"]:
            continue
        if "min_confidence" in rule and event["confidence"] < rule["min_confidence"]:
            continue
        if "max_confidence" in rule and event["confidence"] > rule["max_confidence"]:
            continue
        return rule
    return None


def process_events(events, config):
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
            "severity": matched.get("severity", "-") if matched else "-",
        })

    return results


def aggregate_alerts(results, threshold):
    image_stats = {}

    for r in results:
        img = r["source_image"]
        if img not in image_stats:
            image_stats[img] = {"alert": 0, "warning": 0, "ignore": 0, "ok": 0}

        action = r["action"]
        if action in image_stats[img]:
            image_stats[img][action] += 1

    alert_images = [img for img, stats in image_stats.items() if stats["alert"] > 0]

    return {
        "image_stats": image_stats,
        "alert_images": alert_images,
        "should_alert": len(alert_images) >= threshold,
        "threshold": threshold,
    }


if __name__ == "__main__":
    config = load_rules("rules.yaml")
    print(f"載入 {len(config['rules'])} 條規則")
    print(f"聚合門檻: {config['alert_threshold']} 張圖\n")

    with open("sample_results.json") as f:
        events = json.load(f)
    print(f"讀取 {len(events)} 筆事件\n")

    results = process_events(events, config)

    print("=== 規則比對結果 ===")
    for r in results:
        print(
            f"  {r['source_image']} | {r['event_type']} "
            f"({r['confidence']:.2f}) → {r['action']} [{r['rule']}]"
        )

    agg = aggregate_alerts(results, config["alert_threshold"])

    print("\n=== 聚合結果 ===")
    for img, stats in agg["image_stats"].items():
        print(f"  {img}: {stats}")

    print(f"\n有 alert 的圖片: {agg['alert_images']}")
    print(f"門檻: {agg['threshold']} 張")

    if agg["should_alert"]:
        print(f"\n發佈 ALERT：{len(agg['alert_images'])} 張圖偵測到違規（>= {agg['threshold']}）")
    else:
        print(f"\n不發佈：只有 {len(agg['alert_images'])} 張圖有 alert，未達門檻 {agg['threshold']}")

    with open("rule_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n已寫入 rule_results.json（{len(results)} 筆）")
