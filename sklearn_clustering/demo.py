"""
Demo：用 KMeans 找 confidence 門檻

這支程式示範完整流程：
  1. 讀取偵測結果
  2. 用 KMeans 分 3 群
  3. 算出門檻
  4. 跟原本的 rules.yaml 比較

執行方式：
  python demo.py
  python demo.py rule_results_small.json   ← 用 21 筆資料
  python demo.py rule_results_large.json   ← 用 100 筆資料
"""

import json
import sys
import pandas as pd
from sklearn.cluster import KMeans


# === 第一步：讀取資料 ===

def load_data(json_path):
    """讀取 JSON，回傳 DataFrame"""
    with open(json_path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


# === 第二步：KMeans 分群 ===

def cluster_confidence(df, n_clusters=3):
    """
    用 KMeans 對 confidence 分群

    參數：
      df: DataFrame，需要有 "confidence" 欄位
      n_clusters: 分幾群（預設 3：低/中/高）

    回傳：
      df: 加上 "cluster" 欄位的 DataFrame
      centers: 排序後的群中心
    """
    # 取出 confidence，轉成 KMeans 要的格式（二維陣列）
    X = df[["confidence"]]

    # 建立 KMeans，分成 n_clusters 群
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # 訓練 + 預測（fit_predict 同時做兩件事）
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X)

    # 群中心，排序
    centers = sorted(kmeans.cluster_centers_.flatten())

    return df, centers


# === 第三步：算門檻 ===

def calculate_thresholds(centers):
    """
    從群中心算出門檻（相鄰兩群中心的中點）

    例如中心 [0.45, 0.70, 0.87]
    → 門檻 [0.575, 0.785]
    → 低/中界線 = 0.575，中/高界線 = 0.785
    """
    thresholds = []
    for i in range(len(centers) - 1):
        mid = round((centers[i] + centers[i + 1]) / 2, 4)
        thresholds.append(mid)
    return thresholds


# === 第四步：印出結果 ===

def show_results(df, centers, thresholds):
    """印出分群結果和建議門檻"""

    print("=== 群中心 ===")
    labels = ["低信心", "中信心", "高信心"]
    for i, center in enumerate(centers):
        label = labels[i] if i < len(labels) else f"群{i}"
        count = len(df[df["cluster"] == i])
        print(f"  {label}: 中心 = {center:.4f}（{count} 筆）")

    print(f"\n=== 建議門檻 ===")
    print(f"  低/中界線: {thresholds[0]}")
    print(f"  中/高界線: {thresholds[1]}")

    print(f"\n=== 對照原本 rules.yaml ===")
    print(f"  原本: max_confidence=0.5, min_confidence=0.7")
    print(f"  建議: max_confidence={thresholds[0]}, min_confidence={thresholds[1]}")

    print(f"\n=== 建議的 rules.yaml ===")
    print(f"""rules:
  - name: 低信心忽略
    max_confidence: {thresholds[0]}
    action: ignore

  - name: 沒戴安全帽（高信心）
    event_type: head_detected
    min_confidence: {thresholds[1]}
    action: alert
    severity: high

  - name: 沒戴安全帽（中信心）
    event_type: head_detected
    action: warning
    severity: low

  - name: 預設
    action: ok

alert_threshold: 2""")


# === 主程式 ===

if __name__ == "__main__":
    # 選擇資料檔
    json_path = sys.argv[1] if len(sys.argv) > 1 else "rule_results_large.json"
    print(f"讀取: {json_path}\n")

    # 1. 讀取
    df = load_data(json_path)
    print(f"共 {len(df)} 筆事件")
    print(f"confidence 範圍: {df['confidence'].min():.4f} ~ {df['confidence'].max():.4f}\n")

    # 2. 分群
    df, centers = cluster_confidence(df, n_clusters=3)

    # 3. 算門檻
    thresholds = calculate_thresholds(centers)

    # 4. 印結果
    show_results(df, centers, thresholds)
