"""
解答：用 pandas 分析規則比對結果

讀取 rule_results.json，用 DataFrame 做統計分析。

執行方式：
  python solution.py
"""

import pandas as pd
import json


def load_data(json_path):
    """讀取 JSON 檔，回傳 DataFrame"""
    with open(json_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def basic_stats(df):
    """基本統計：總筆數、欄位、前幾筆"""
    print("=== 基本資訊 ===")
    print(f"總筆數: {len(df)}")
    print(f"欄位: {list(df.columns)}")
    print()
    print(df.head())
    print()


def count_by_action(df):
    """統計每種 action 的數量"""
    print("=== 各 action 數量 ===")
    counts = df["action"].value_counts()
    print(counts)
    print()
    return counts


def count_by_image(df):
    """統計每張圖的事件數和 alert 數"""
    print("=== 每張圖的統計 ===")

    # 每張圖的總事件數
    image_counts = df.groupby("source_image")["action"].count()
    print("事件數:")
    print(image_counts)
    print()

    # 每張圖的 alert 數
    alerts = df[df["action"] == "alert"]
    alert_counts = alerts.groupby("source_image")["action"].count()
    print("alert 數:")
    print(alert_counts)
    print()

    return image_counts, alert_counts


def confidence_stats(df):
    """信心度統計"""
    print("=== 信心度統計 ===")
    print(df["confidence"].describe())
    print()

    # 按 event_type 分組的平均信心度
    print("各 event_type 平均信心度:")
    avg = df.groupby("event_type")["confidence"].mean()
    print(avg)
    print()

    return avg


def find_violations(df):
    """找出所有違規事件（action 為 alert 或 warning）"""
    print("=== 違規事件 ===")
    violations = df[df["action"].isin(["alert", "warning"])]
    print(violations[["source_image", "event_type", "confidence", "action"]])
    print(f"\n共 {len(violations)} 筆違規")
    print()
    return violations


if __name__ == "__main__":
    # 1. 讀取資料
    df = load_data("rule_results.json")

    # 2. 基本統計
    basic_stats(df)

    # 3. 按 action 統計
    count_by_action(df)

    # 4. 按圖片統計
    count_by_image(df)

    # 5. 信心度統計
    confidence_stats(df)

    # 6. 找出違規
    find_violations(df)
