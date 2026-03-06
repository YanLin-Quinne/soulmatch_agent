#!/usr/bin/env python3
"""筛选旧金山 OkCupid 数据集中的高质量 profiles"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any

CSV_PATH = "/Users/quinne/Downloads/okcupid_profiles.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data/raw/sf_profiles_selected.json"

def calculate_essay_word_count(row: pd.Series) -> int:
    """计算 essay0-essay9 的总字数"""
    total = 0
    for i in range(10):
        essay = row.get(f"essay{i}")
        if pd.notna(essay) and isinstance(essay, str):
            total += len(essay.split())
    return total

def count_non_empty_essays(row: pd.Series) -> int:
    """计算非空 essay 数量"""
    count = 0
    for i in range(10):
        essay = row.get(f"essay{i}")
        if pd.notna(essay) and isinstance(essay, str) and len(essay.strip()) > 0:
            count += 1
    return count

def main():
    print(f"读取 CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"总记录数: {len(df)}")

    # 筛选条件
    print("\n应用筛选条件...")
    df['essay_word_count'] = df.apply(calculate_essay_word_count, axis=1)
    df['non_empty_essays'] = df.apply(count_non_empty_essays, axis=1)

    filtered = df[
        (df['age'] >= 22) &
        (df['age'] <= 50) &
        (df['non_empty_essays'] >= 3) &
        (df['essay_word_count'] >= 500)
    ].copy()

    print(f"筛选后记录数: {len(filtered)}")

    # 多样性采样
    print("\n按多样性采样...")

    # 年龄段分组
    filtered['age_group'] = pd.cut(filtered['age'], bins=[22, 28, 35, 42, 50],
                                    labels=['20s', '30s_early', '30s_late', '40s'])

    # 每个组合采样
    samples: List[pd.Series] = []

    # 性别 x 年龄段，每组取 2-3 个
    for sex in ['m', 'f']:
        for age_group in ['20s', '30s_early', '30s_late', '40s']:
            group = filtered[(filtered['sex'] == sex) & (filtered['age_group'] == age_group)]
            if len(group) > 0:
                n = min(3, len(group))
                samples.extend([row for _, row in group.nlargest(n, 'essay_word_count').iterrows()])

    # 去重并限制到 20 个
    seen_indices = set()
    unique_samples = []
    for sample in samples:
        if sample.name not in seen_indices:
            seen_indices.add(sample.name)
            unique_samples.append(sample)
            if len(unique_samples) >= 20:
                break

    print(f"最终选择: {len(unique_samples)} 个 profiles")

    # 转换为 JSON
    results = []
    for idx, row in enumerate(unique_samples):
        profile = {
            "profile_id": f"sf_{idx}",
            "original_index": int(row.name),
            "age": int(row['age']),
            "sex": row['sex'],
            "orientation": row['orientation'],
            "status": row['status'],
            "body_type": row['body_type'],
            "diet": row['diet'],
            "drinks": row['drinks'],
            "drugs": row['drugs'],
            "education": row['education'],
            "ethnicity": row['ethnicity'],
            "height": float(row['height']) if pd.notna(row['height']) else None,
            "income": int(row['income']) if pd.notna(row['income']) else -1,
            "job": row['job'],
            "location": row['location'],
            "offspring": row['offspring'],
            "pets": row['pets'],
            "religion": row['religion'],
            "sign": row['sign'],
            "smokes": row['smokes'],
            "speaks": row['speaks'],
            "essays": {
                f"essay{i}": row[f"essay{i}"] if pd.notna(row[f"essay{i}"]) else ""
                for i in range(10)
            },
            "essay_word_count": int(row['essay_word_count']),
            "non_empty_essays": int(row['non_empty_essays'])
        }
        results.append(profile)

    # 保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n保存到: {OUTPUT_PATH}")

    # 统计
    print("\n多样性统计:")
    result_df = pd.DataFrame(results)
    print(f"性别分布: {result_df['sex'].value_counts().to_dict()}")
    print(f"年龄范围: {result_df['age'].min()}-{result_df['age'].max()}")
    print(f"平均 essay 字数: {result_df['essay_word_count'].mean():.0f}")

if __name__ == "__main__":
    main()
